# evaluate_qwen_lora.py
# -*- coding: utf-8 -*-

"""
仿照 evaluate_medgamma.py 的评估格式，但用于 Qwen2.5-VL LoRA/QLoRA 皮肤病 4 分类：
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]  (无 bkl)

核心点：
1) Qwen2.5-VL 必须在 prompt 文本里出现 image placeholder（由 apply_chat_template 生成的 <image> 之类 token）
   否则会报：Image features and image tokens do not match: tokens: 0, features ...
2) 评估默认用“线性一跳”（取最后一个位置的 logits，对四个类别首 token 做 softmax 选最大），速度快、显存更省。
3) 支持 4bit 加载（推荐在 16GB 显存上 eval），只要安装 bitsandbytes。

你只需要改动下面的 BASE_MODEL / LORA_DIR / TEST_CSV / IMAGE_ROOT_DIR 等路径即可运行。
"""

import os
import re
import warnings

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from peft import PeftModel

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ===================== 路径 & 配置（与训练脚本一致） =====================

# 你的 Qwen2.5-VL 基座
BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"


# 训练脚本生成的 test CSV（4分类，无 bkl）
TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# LoRA/QLoRA 权重目录（= 你的训练 OUTPUT_DIR）
# 这里应当包含 adapter_config.json、adapter_model.safetensors，以及 tokenizer/processor 配置（你训练脚本 save_pretrained 的输出）
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen25vl_derm_lora2"

# 图像根目录和后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"  # 若是 jpg 改为 ".jpg"

# 评估结果保存目录
EVAL_OUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen_eval_results2"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# 列名（与你的 CSV 一致即可）
COL_IMAGE_ID = "image_id"
COL_TARGET = "dx"

# 只评估这 4 类（无 bkl）
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]

# 是否使用 4bit 加载（建议 True：更省显存）
USE_4BIT = True
# Qwen2.5-VL 推荐 bfloat16 作为 4bit 的 compute dtype（你的日志里也用 bf16）
COMPUTE_DTYPE = torch.bfloat16

# 可选：限制视觉 token 数（更省显存/更快，但可能影响精度）
# Qwen 的 processor 一般支持在 image_processor 上设置 min/max_pixels；
# 如果你的 transformers 版本不支持这两个属性，下面 try 会自动跳过。
MIN_PIXELS = None   # 例如 224*224
MAX_PIXELS = None   # 例如 448*448 或 512*512


# ===================== 小工具 =====================

def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def extract_dx_code(text: str) -> str:
    """从生成文本中抽取 dx code（兜底用）。"""
    if not isinstance(text, str):
        return "unknown"
    t = text.lower()
    m = re.search(r"\b(akiec|bcc|nev|mel)\b", t)
    if not m:
        return "unknown"
    code = normalize_dx(m.group(1))
    return code if code in ALLOWED_DX else "unknown"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    return device


def load_qwen_lora_model_and_processor():
    device = get_device()

    # 1) processor：优先从 LORA_DIR 读（这样保证 tokenizer/chat_template 与训练一致）
    processor = AutoProcessor.from_pretrained(LORA_DIR, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"

    # 2) 可选：设置 min/max_pixels（降低视觉 token 量 -> 省显存）
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            if MIN_PIXELS is not None:
                ip.min_pixels = int(MIN_PIXELS)
            if MAX_PIXELS is not None:
                ip.max_pixels = int(MAX_PIXELS)
            if MIN_PIXELS is not None or MAX_PIXELS is not None:
                print(f"🔧 已设置 image_processor: min_pixels={getattr(ip,'min_pixels',None)}, max_pixels={getattr(ip,'max_pixels',None)}")
    except Exception as e:
        print(f"⚠ 设置 min/max_pixels 失败（可忽略）: {e}")

    # 3) base model + adapter
    print("🔧 加载 Qwen2.5-VL 基座模型 + LoRA 适配器 ...")

    if USE_4BIT:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "你开启了 USE_4BIT=True，但当前环境无法导入 BitsAndBytesConfig。\n"
                "请先安装 bitsandbytes（Windows 可能需要特定版本/编译），或把 USE_4BIT=False。"
            ) from e

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
        # 非 4bit：用 bf16/fp16（更吃显存）
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

    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    # 注意：4bit + device_map="auto" 时，不要强行 model.to(device)
    # 非 4bit 时 base 已经 to(device)，adapter 会跟随

    return model, processor, device


# ===================== 核心评估：线性一跳（取 last_logits） =====================

def build_messages_for_qwen(clinical_note: str):
    """
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
        # ⭐ 这里 image placeholder 必须存在（不要写成 {"type":"image","image": image}）
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    return messages


def evaluate_qwen_lora_linear():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"未找到测试集 CSV: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"📄 从 Test CSV 读取 {len(df)} 条样本: {TEST_CSV}")

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError(f"TEST_CSV 中缺少列：{COL_IMAGE_ID} 或 {COL_TARGET}")

    model, processor, device = load_qwen_lora_model_and_processor()

    # ① 计算 4 个类别的首 token id（与训练“首 token 监督/线性头”一致）
    label_token_ids = []
    for cls in ALLOWED_DX:
        ids = processor.tokenizer(cls, add_special_tokens=False).input_ids
        if len(ids) == 0:
            raise ValueError(f"标签 {cls} tokenizer 结果为空")
        label_token_ids.append(ids[0])
    label_token_ids = torch.tensor(label_token_ids, device=device)  # (4,)

    # 可选：logit_bias（默认全 0；你也可以手动给某些类加偏置）
    logit_bias = torch.zeros((len(ALLOWED_DX),), device=device)

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

        # 你如果希望完全复用 MedGEMMA 的临床信息拼接，可把 evaluate_medgamma.py 里的 build_clinical_note() 拷过来
        # 这里最小化：如果你的 CSV 没有临床字段，就用空字符串也行
        clinical_note = ""
        if "年龄" in df.columns or "性别" in df.columns or "区域" in df.columns:
            # 兼容：如果你仍保留这些字段，可以简单拼接几项
            age = row.get("年龄", "")
            sex = row.get("性别", "")
            region = row.get("区域", "")
            clinical_note = f"{age}-year-old {sex}, lesion on {region}."
        else:
            clinical_note = "No additional clinical information."

        messages = build_messages_for_qwen(clinical_note)
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )
        # 4bit + device_map=auto 时，inputs 放 cuda:0 通常可行；但也可能需要跟随 model.hf_device_map
        # 简化：有 cuda 就放 cuda:0
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

    print("\n====== 📊 Qwen2.5-VL LoRA 线性评估结果（4类） ======")
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

    print("\n====== 📊 classification_report ======")
    print(classification_report(y_true_arr, y_pred_arr, labels=classes))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    print("\n====== 📊 混淆矩阵（rows=true, cols=pred） ======")
    print(classes)
    print(cm)

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
    ax_cm.set_title("Confusion Matrix (Qwen2.5-VL LoRA, 4 classes, linear)")
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
    cm_path = os.path.join(EVAL_OUT_DIR, "confusion_matrix_qwen_linear.png")
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"📁 混淆矩阵图已保存到: {cm_path}")

    # ROC & PR（用 one-hot 预测当 pseudo-score）
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
    ax_roc.set_title("ROC Curves (Qwen2.5-VL LoRA, 4 classes, pseudo-scores)")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(EVAL_OUT_DIR, "roc_curve_qwen_linear.png")
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
    ax_pr.set_title("Precision-Recall Curves (Qwen2.5-VL LoRA, 4 classes, pseudo-scores)")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(EVAL_OUT_DIR, "pr_curve_qwen_linear.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 P-R 曲线图已保存到: {pr_path}")


if __name__ == "__main__":
    warnings.filterwarnings("once")
    set_seed(42)
    evaluate_qwen_lora_linear()
