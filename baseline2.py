# evaluate_qwen_linear_fair.py
# -*- coding: utf-8 -*-
"""
一个“公平对照”的 Qwen2.5-VL 4分类评估脚本：
- 与 evaluate_qwen_lora.py 的 linear 评估逻辑保持一致（last-logits -> 4个label首token softmax）
- Baseline 与 LoRA 评估除了“是否加载 LoRA 适配器”这一项之外，其余全部一致（processor、prompt、label_token_ids、device/4bit设置、数据读取、统计方式）

用法示例：
  # 只跑 baseline（不加载 LoRA）
  python evaluate_qwen_linear_fair.py --mode baseline

  # 只跑 LoRA
  python evaluate_qwen_linear_fair.py --mode lora

  # 两个都跑（建议，输出两个报告文件）
  python evaluate_qwen_linear_fair.py --mode both
"""

import os
import re
import warnings
import argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


# ===================== 路径 & 配置（按需修改） =====================

# 本地基座路径（按你要求的格式）
BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

# test CSV
TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 图像根目录和后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"

# LoRA 输出目录（= 训练 OUTPUT_DIR；baseline 也会用它来加载 processor，以保证 chat_template/processor 完全一致）
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen_output"

# 输出目录
EVAL_OUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen_eval_fair"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# 列名
COL_IMAGE_ID = "image_id"
COL_TARGET = "dx"

# 4分类（无 bkl）
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]

# 与 evaluate_qwen_lora.py 保持一致：label_token_ids 取 tokenizer(cls) 的首 token（不加空格）
LABEL_TOKEN_ADD_SPACE = False

# 与 evaluate_qwen_lora.py 保持一致：默认 4bit
USE_4BIT = True
COMPUTE_DTYPE = torch.bfloat16

# 可选：限制视觉 token（两边都一样，确保公平）
MIN_PIXELS = None   # 例如 224*224
MAX_PIXELS = None   # 例如 448*448

# （重要）不要对多模态输入做 truncation='max_length'，否则可能截断 image token 引发 mismatch
# 如果你担心文本过长，请只截断 clinical_note 字符串
CLINICAL_NOTE_MAX_CHARS = 600


# ===================== 工具函数 =====================

def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    return device


def build_clinical_note_minimal(row: pd.Series) -> str:
    """
    这里保持和 evaluate_qwen_lora.py 一致：只做最小拼接（避免引入第二变量）。
    你如果希望使用更复杂的临床字段拼接，务必 baseline/LoRA 都同步改。
    """
    age = row.get("年龄", "")
    sex = row.get("性别", "")
    region = row.get("区域", "")
    clinical_note = f"{age}-year-old {sex}, lesion on {region}."
    if len(clinical_note) > CLINICAL_NOTE_MAX_CHARS:
        clinical_note = clinical_note[:CLINICAL_NOTE_MAX_CHARS] + "…"
    return clinical_note


def build_messages_for_qwen(clinical_note: str):
    """
    与 evaluate_qwen_lora.py 保持一致。
    Qwen2.5-VL 的关键：messages 里必须包含 {"type":"image"} 占位符，
    这样 apply_chat_template 生成的文本会含有 image token，避免 tokens=0 的报错。
    """
    system_text = (
        "You are a dermatology assistant. "
        "Given the clinical note and the skin lesion image, "
        "classify the lesion into one of the following classes: "
        "akiec, bcc, nev, mel. "
        "Always answer with exactly one lowercase class name from this set, with no explanations."
    )

    user_text = (
        f"Clinical note:\n{clinical_note}\n\n"
        "Based on the clinical note and the provided skin lesion image, "
        "predict the most likely disease class.\n"
        "Answer with only one class name:\n"
        "akiec, bcc, nev, mel.\n"
        "Final answer:"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    return messages


def load_processor_fair():
    """
    优先从本地 LoRA 输出目录加载 processor；
    如果该目录不包含 processor 文件，则退回到 BASE_MODEL。
    """
    if os.path.isdir(LORA_DIR):
        print(f"🔧 从本地 LORA_DIR 加载 processor: {LORA_DIR}")
        processor = AutoProcessor.from_pretrained(
            LORA_DIR,
            trust_remote_code=True,
            local_files_only=True,   # ⭐ 关键
        )
    else:
        print(f"⚠ LORA_DIR 不存在，退回 BASE_MODEL processor")
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

    processor.tokenizer.padding_side = "right"
    return processor



def load_model_fair(load_lora: bool):
    """
    唯一变量：load_lora=True 时加载 adapter；False 时不加载 adapter。
    其余（4bit、device_map、processor来源）保持一致。
    """
    device = get_device()
    processor = load_processor_fair()

    print("🔧 加载 Qwen2.5-VL 基座模型 ...")
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if device.type == "cuda":
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        else:
            dtype = torch.float32

        base_model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
            trust_remote_code=True,
        )
        base_model.to(device)

    if load_lora:
        if PeftModel is None:
            raise RuntimeError("当前环境无法导入 peft.PeftModel，但你选择了 --mode lora。请先安装 peft。")
        print("🔧 加载 LoRA 适配器 ...")
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
    else:
        print("🔧 Baseline：不加载 LoRA 适配器（只用基座）")
        model = base_model

    model.eval()
    return model, processor, device


def compute_label_token_ids(processor):
    label_token_ids = []
    for cls in ALLOWED_DX:
        text = (" " + cls) if LABEL_TOKEN_ADD_SPACE else cls
        ids = processor.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) == 0:
            raise ValueError(f"标签 {cls} tokenizer 结果为空")
        label_token_ids.append(ids[0])
    return torch.tensor(label_token_ids, device="cuda" if torch.cuda.is_available() else "cpu")


def run_eval(mode_name: str, load_lora: bool):
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"未找到测试集 CSV: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"📄 从 Test CSV 读取 {len(df)} 条样本: {TEST_CSV}")

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError(f"TEST_CSV 中缺少列：{COL_IMAGE_ID} 或 {COL_TARGET}")

    model, processor, device = load_model_fair(load_lora=load_lora)
    label_token_ids = compute_label_token_ids(processor)
    print("🔧 label_token_ids:", {c: int(t) for c, t in zip(ALLOWED_DX, label_token_ids.tolist())})

    # 与 evaluate_qwen_lora.py 一致：logit_bias=0
    logit_bias = torch.zeros((len(ALLOWED_DX),), device=label_token_ids.device)

    y_true, y_pred = [], []
    total, correct = 0, 0
    missing = 0

    for _, row in df.iterrows():
        image_id = str(row[COL_IMAGE_ID])
        true_label = normalize_dx(str(row[COL_TARGET]))
        if true_label not in ALLOWED_DX:
            continue

        img_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        if not os.path.exists(img_path):
            print(f"⚠ image_id={image_id} 对应图片不存在: {img_path}")
            missing += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠ 打开图片失败 image_id={image_id}: {e}")
            missing += 1
            continue

        clinical_note = build_clinical_note_minimal(row)
        messages = build_messages_for_qwen(clinical_note)
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 关键：不要 truncation='max_length'，避免 image token 被截断导致 mismatch
        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )

        # 与 evaluate_qwen_lora.py 保持一致：有 cuda 就放 cuda
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]  # (vocab,)
        logits_k = last_logits[label_token_ids] + logit_bias  # (4,)
        probs_k = torch.softmax(logits_k, dim=-1)
        pred_idx = int(torch.argmax(probs_k).item())
        pred_label = ALLOWED_DX[pred_idx]

        total += 1
        correct += int(pred_label == true_label)
        y_true.append(true_label)
        y_pred.append(pred_label)

        print(
            f"🩺 [{total}] image_id={image_id} | pred={pred_label} | true={true_label} "
            f"| {'✅' if pred_label == true_label else '❌'}"
        )

    print(f"\n====== 📊 {mode_name} 线性评估结果（4类） ======")
    print(f"有效样本数: {total}")
    print(f"缺少图片样本数: {missing}")
    if total == 0:
        print("没有有效样本，评估终止。")
        return

    acc = correct / total
    print(f"总体准确率: {acc:.2%}")

    classes = ALLOWED_DX
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    report = classification_report(y_true_arr, y_pred_arr, labels=classes)
    print("\n====== 📊 classification_report ======")
    print(report)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    print("\n====== 📊 混淆矩阵（rows=true, cols=pred） ======")
    print(classes)
    print(cm)

    # 保存文本报告（方便你论文/对照）
    out_txt = os.path.join(EVAL_OUT_DIR, f"{mode_name}_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"{mode_name} ACC: {acc:.6f}\n\n")
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(classes) + "\n")
        f.write(np.array2string(cm) + "\n")
    print(f"📁 报告已保存到: {out_txt}")

    # 混淆矩阵图
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(len(classes)))
    ax_cm.set_yticks(range(len(classes)))
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title(f"Confusion Matrix ({mode_name}, 4 classes, linear)")
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig_cm.tight_layout()
    cm_path = os.path.join(EVAL_OUT_DIR, f"{mode_name}_confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"📁 混淆矩阵图已保存到: {cm_path}")

    # 与 evaluate_qwen_lora.py 保持一致：ROC/PR 用 pseudo-score（one-hot 预测）
    y_true_bin = label_binarize(y_true_arr, classes=classes)
    scores = np.zeros_like(y_true_bin, dtype=float)
    for i, pred in enumerate(y_pred_arr):
        if pred in classes:
            scores[i, classes.index(pred)] = 1.0

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], scores[:, idx])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
        except ValueError:
            continue
    ax_roc.plot([0, 1], [0, 1], "k--", label="chance")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves ({mode_name}, 4 classes, pseudo-scores)")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(EVAL_OUT_DIR, f"{mode_name}_roc_curve.png")
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)
    print(f"📁 ROC 曲线图已保存到: {roc_path}")

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(classes):
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], scores[:, idx])
            ap = average_precision_score(y_true_bin[:, idx], scores[:, idx])
            ax_pr.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
        except ValueError:
            continue
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall ({mode_name}, 4 classes, pseudo-scores)")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(EVAL_OUT_DIR, f"{mode_name}_pr_curve.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 P-R 曲线图已保存到: {pr_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "lora", "both"], default="both",
                        help="baseline=不加载LoRA；lora=加载LoRA；both=两者都跑（推荐）")
    args = parser.parse_args()

    warnings.filterwarnings("once")
    set_seed(42)

    if args.mode in ["baseline", "both"]:
        run_eval(mode_name="baseline", load_lora=False)

    if args.mode in ["lora", "both"]:
        run_eval(mode_name="lora", load_lora=True)


if __name__ == "__main__":
    main()
