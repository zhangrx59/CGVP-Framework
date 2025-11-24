# evaluate_medgamma.py
# -*- coding: utf-8 -*-

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


# ========== 路径 & 配置（需与微调脚本一致） ==========

BASE_MODEL = "google/medgemma-4b-it"

# 原始 metadata CSV
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 微调脚本中 prepare_splits() 生成的 test CSV
TEST_CSV = METADATA_CSV.replace(".csv", "_test_5cls.csv")

# LoRA 权重所在目录（要和 LoRA_medgamma.py 里的 OUTPUT_DIR 一致）
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\medgemma_lora_derm_from_metadata"

# 图像根目录和后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"   # 如果是 .jpg 改成 ".jpg"

# 评估图像保存目录
LORA_PLOTS_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\eval_lora_linear"
os.makedirs(LORA_PLOTS_DIR, exist_ok=True)

# 列名（与微调脚本保持一致）
COL_IMAGE_ID    = "image_id"
COL_AGE         = "年龄"
COL_SEX         = "性别"
COL_FATHER_ORI  = "父籍贯"
COL_MOTHER_ORI  = "母籍贯"
COL_BIOPSY      = "是否活检"
COL_SMOKE       = "是否吸烟"
COL_DRINK       = "是否饮酒"
COL_PESTICIDE   = "农药"
COL_SKIN_CANCER = "皮肤癌病史"
COL_OTHER_CA    = "癌症病史"
COL_TAP_WATER   = "生活环境是否有自来水"
COL_SEWER       = "生活环境是否有下水道"
COL_PHOTOTYPE   = "皮肤光型"
COL_REGION      = "区域"
COL_D1          = "直径1"
COL_D2          = "直径2"
COL_PRURITUS    = "瘙痒"
COL_GROWTH      = "是否长大"
COL_PAIN        = "疼痛"
COL_MORPH_CHANGE= "形态变化"
COL_BLEEDING    = "出血"
COL_ELEVATED    = "是否隆起"

COL_TARGET      = "dx"   # 如果你的列名是“诊断标签”，这里改成 "诊断标签"

# 只评估这 5 类
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]


# ========== 一些工具函数（和 LoRA_medgamma.py 保持一致） ==========

def yn_str(v, yes: str, no: str, unk: str = "unknown") -> str:
    """
    把各种 True/False/空/NaN 归一化成 yes/no/unk
    """
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in ["TRUE", "T", "YES", "Y", "1"]:
            return yes
        if vs in ["FALSE", "F", "NO", "N", "0"]:
            return no
        if vs in ["UNK", "UNKNOWN", "NA", "NAN", "NONE", ""]:
            return unk
    if isinstance(v, (bool, int)):
        return yes if bool(v) else no
    if v != v:  # NaN
        return unk
    return str(v)


def build_clinical_note(row: pd.Series) -> str:
    """
    与训练脚本中的英文病历构造逻辑保持一致
    """
    age = row.get(COL_AGE, "")
    sex_raw = str(row.get(COL_SEX, "") or "").strip().lower()
    region = str(row.get(COL_REGION, "") or "").strip()
    father_ori = str(row.get(COL_FATHER_ORI, "") or "").strip()
    mother_ori = str(row.get(COL_MOTHER_ORI, "") or "").strip()

    # 性别英文化
    if sex_raw in ["男", "male", "m"]:
        sex_en = "male"
    elif sex_raw in ["女", "female", "f"]:
        sex_en = "female"
    else:
        sex_en = "unknown"

    skin_ca = yn_str(row.get(COL_SKIN_CANCER), "yes", "no")
    other_ca = yn_str(row.get(COL_OTHER_CA), "yes", "no")
    smoke = yn_str(row.get(COL_SMOKE), "yes", "no")
    drink = yn_str(row.get(COL_DRINK), "yes", "no")
    pesticide = yn_str(row.get(COL_PESTICIDE), "yes", "no")

    tap = yn_str(row.get(COL_TAP_WATER), "yes", "no")
    sewer = yn_str(row.get(COL_SEWER), "yes", "no")

    phototype = row.get(COL_PHOTOTYPE, "")
    d1 = row.get(COL_D1, "")
    d2 = row.get(COL_D2, "")

    pruritus = yn_str(row.get(COL_PRURITUS), "present", "absent")
    growth = yn_str(row.get(COL_GROWTH), "present", "absent")
    pain = yn_str(row.get(COL_PAIN), "present", "absent")
    morph_change = yn_str(row.get(COL_MORPH_CHANGE), "present", "absent")
    bleeding = yn_str(row.get(COL_BLEEDING), "present", "absent")
    elevated = yn_str(row.get(COL_ELEVATED), "present", "absent")

    region_en = region if region else "unknown region"

    size_str = ""
    if d1 and d2:
        size_str = f"Lesion size is about {d1} by {d2} mm."
    elif d1:
        size_str = f"Lesion maximum diameter is about {d1} mm."

    phototype_str = f"Fitzpatrick skin phototype: {phototype}." if phototype != "" else ""

    origin_str = ""
    if father_ori or mother_ori:
        origin_str = (
            f"The patient's father is from {father_ori or 'unknown'}, "
            f"and mother is from {mother_ori or 'unknown'}."
        )

    parts = []
    parts.append(f"{age}-year-old {sex_en} with a skin lesion on the {region_en}.")
    if size_str:
        parts.append(size_str)
    if origin_str:
        parts.append(origin_str)

    parts.append(
        f"Past history of skin cancer: {skin_ca}; "
        f"other malignancies: {other_ca}."
    )
    parts.append(
        f"Lifestyle: smoking {smoke}, alcohol {drink}, pesticide exposure {pesticide}."
    )
    parts.append(
        f"Living environment: tap water {tap}, sewer system {sewer}."
    )
    if phototype_str:
        parts.append(phototype_str)

    parts.append(
        "Current symptoms and signs: "
        f"pruritus {pruritus}, growth {growth}, pain {pain}, "
        f"morphologic change {morph_change}, bleeding {bleeding}, "
        f"elevation {elevated}."
    )

    note = " ".join(parts)
    return note


def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def extract_dx_code(text: str) -> str:
    """
    从模型输出文本中提取 5 类 dx code：
    - 支持 nv/nev，统一成 nev
    """
    if not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    m = re.search(r"\b(mel|bcc|bkl|nev|nv|akiec)\b", text_lower)
    if not m:
        return "unknown"
    code = m.group(1)
    code = normalize_dx(code)
    return code if code in ALLOWED_DX else "unknown"


# ========== 设备与 LoRA 模型加载 ==========

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    return device


def load_lora_model_and_processor():
    device = get_device()
    print("🔧 加载 MedGEMMA 基座模型 + LoRA 适配器 ...")

    if device.type == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if supports_bf16:
            dtype = torch.bfloat16
            print("🔧 GPU 支持 bfloat16，使用 dtype=torch.bfloat16")
        else:
            dtype = torch.float16
            print("🔧 GPU 不支持 bfloat16，使用 dtype=torch.float16")
    else:
        dtype = torch.float32
        print("🔧 使用 CPU，dtype=torch.float32")

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(LORA_DIR)
    processor.tokenizer.padding_side = "right"

    return model, processor, device


# ========== 逐样本线性推理评估 Test 集（方案 A：按 "Final answer:" 切） ==========

def evaluate_lora_linear():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"未找到测试集 CSV: {TEST_CSV}\n"
            f"请先运行微调脚本生成 *_test_5cls.csv。"
        )

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"📄 从 Test CSV 读取 {len(df)} 条样本: {TEST_CSV}")

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError("TEST_CSV 中缺少 image_id 或 dx 列")

    model, processor, device = load_lora_model_and_processor()

    y_true, y_pred = [], []
    total, correct = 0, 0
    missing_image = 0

    for _, row in df.iterrows():
        image_id = str(row[COL_IMAGE_ID])
        label_raw = normalize_dx(str(row[COL_TARGET]))
        if label_raw not in ALLOWED_DX:
            continue

        img_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        if not os.path.exists(img_path):
            print(f"⚠ image_id={image_id} 对应图片不存在: {img_path}")
            missing_image += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠ 打开图片失败 image_id={image_id}: {e}")
            missing_image += 1
            continue

        clinical_note = build_clinical_note(row)

        # prompt：和训练时保持一致，末尾有 "Final answer:"
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a dermatology assistant. "
                            "Given the clinical note and the skin lesion image, "
                            "your task is to classify the lesion into one of the following classes: "
                            "akiec, bcc, bkl, nev, mel. "
                            "Always answer with exactly one lowercase class name "
                            "from this set, with no explanations."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Clinical note:\n{clinical_note}\n\n"
                            "Based on the clinical note and the provided skin lesion image, "
                            "predict the most likely disease class.\n"
                            "Answer with only one class name:\n"
                            "akiec, bcc, bkl, nev, mel.\n"
                            "Final answer:"
                        ),
                    },
                    {"type": "image", "image": image},
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        # 方案 A：对完整 decode 的文本按 "Final answer:" 切分，只在答案部分做正则
        full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        low = full_text.lower()
        if "final answer:" in low:
            ans_part = low.split("final answer:")[-1]
        else:
            ans_part = low

        pred_label = extract_dx_code(ans_part)

        total += 1
        if pred_label == label_raw:
            correct += 1

        y_true.append(label_raw)
        y_pred.append(pred_label)

        print(
            f"🩺 [{total}] image_id={image_id} | pred={pred_label} | true={label_raw} "
            f"| {'✅' if pred_label == label_raw else '❌'} | ans_part={pred_label!r}"
        )

    print("\n====== 📊 LoRA 模型在线性推理下的 Test 集评估结果 ======")
    print(f"有效样本数: {total}")
    print(f"缺少图片样本数: {missing_image}")
    if total > 0:
        print(f"总体准确率: {correct/total:.2%}")
    else:
        print("没有有效样本")
        return

    # ===== 指标 + 混淆矩阵 + ROC/PR 曲线 =====
    classes = ALLOWED_DX
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    print("\n====== 📊 classification_report ======")
    print(classification_report(y_true_arr, y_pred_arr, labels=classes))

    # 混淆矩阵
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    print("\n====== 📊 混淆矩阵（rows=true, cols=pred） ======")
    print(classes)
    print(cm)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(len(classes)))
    ax_cm.set_yticks(range(len(classes)))
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("Confusion Matrix (LoRA, 5 classes, linear)")
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
    cm_path = os.path.join(LORA_PLOTS_DIR, "confusion_matrix_LoRA_linear.png")
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"📁 混淆矩阵图已保存到: {cm_path}")

    # ROC & PR（使用 one-hot 预测当作 score 近似）
    y_true_bin = label_binarize(y_true_arr, classes=classes)
    scores = np.zeros_like(y_true_bin, dtype=float)
    for i, pred in enumerate(y_pred_arr):
        if pred in classes:
            j = classes.index(pred)
            scores[i, j] = 1.0

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
    ax_roc.set_title("ROC Curves (LoRA, 5 classes, pseudo-scores, linear)")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(LORA_PLOTS_DIR, "roc_curve_LoRA_linear.png")
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)
    print(f"📁 ROC 曲线图已保存到: {roc_path}")

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(classes):
        try:
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, idx], scores[:, idx]
            )
            ap = average_precision_score(y_true_bin[:, idx], scores[:, idx])
            ax_pr.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
        except ValueError:
            continue

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (LoRA, 5 classes, pseudo-scores, linear)")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(LORA_PLOTS_DIR, "pr_curve_LoRA_linear.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 P-R 曲线图已保存到: {pr_path}")


# ========== 主入口：评估 LoRA 微调后模型（线性推理 + Final answer 切分） ==========

if __name__ == "__main__":
    warnings.filterwarnings("once")
    set_seed(42)
    evaluate_lora_linear()
