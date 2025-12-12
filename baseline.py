# evaluate_medgemma_baseline_testset.py
# -*- coding: utf-8 -*-

import os
import re
import warnings

import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoModelForImageTextToText, AutoProcessor

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

BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

# 原始 metadata CSV
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 微调脚本中 prepare_splits() 生成的 test CSV
TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 图像根目录和后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"   # 如果是 .jpg 改成 ".jpg"

# 评估图像保存目录（基座模型的结果单独放一个目录）
PLOTS_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\baseline2"
os.makedirs(PLOTS_DIR, exist_ok=True)

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


# ========== 一些工具函数（和微调脚本一致） ==========

def yn_str(v, yes="有", no="无", unk="不详"):
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in ["TRUE", "T", "YES", "Y"]:
            return yes
        if vs in ["FALSE", "F", "NO", "N"]:
            return no
        if vs in ["UNK", "UNKNOWN", "NA", "NAN", "NONE", ""]:
            return unk
    if isinstance(v, (bool, int)):
        return yes if bool(v) else no
    if v != v:  # NaN
        return unk
    return str(v)


def build_clinical_note(row: pd.Series) -> str:
    age = row.get(COL_AGE, "")
    sex = str(row.get(COL_SEX, "") or "").strip()
    region = str(row.get(COL_REGION, "") or "").strip()
    father_ori = str(row.get(COL_FATHER_ORI, "") or "").strip()
    mother_ori = str(row.get(COL_MOTHER_ORI, "") or "").strip()

    skin_ca = yn_str(row.get(COL_SKIN_CANCER))
    other_ca = yn_str(row.get(COL_OTHER_CA))
    smoke = yn_str(row.get(COL_SMOKE), yes="吸烟", no="不吸烟")
    drink = yn_str(row.get(COL_DRINK), yes="饮酒", no="不饮酒")
    pesticide = yn_str(row.get(COL_PESTICIDE), yes="有农药接触史", no="无农药接触史")

    tap = yn_str(row.get(COL_TAP_WATER), yes="有自来水", no="无自来水")
    sewer = yn_str(row.get(COL_SEWER), yes="有下水道", no="无下水道")

    phototype = row.get(COL_PHOTOTYPE, "")
    d1 = row.get(COL_D1, "")
    d2 = row.get(COL_D2, "")

    pruritus = yn_str(row.get(COL_PRURITUS))
    growth = yn_str(row.get(COL_GROWTH))
    pain = yn_str(row.get(COL_PAIN))
    morph_change = yn_str(row.get(COL_MORPH_CHANGE))
    bleeding = yn_str(row.get(COL_BLEEDING))
    elevated = yn_str(row.get(COL_ELEVATED))

    # 性别汉化
    if isinstance(sex, str) and sex.upper() in ["MALE", "M"]:
        sex_cn = "男性"
    elif isinstance(sex, str) and sex.upper() in ["FEMALE", "F"]:
        sex_cn = "女性"
    else:
        sex_cn = sex or "性别不详"

    region_cn = region or "部位不详"

    size_str = ""
    if d1 and d2:
        size_str = f"皮损约 {d1}×{d2} mm"
    elif d1:
        size_str = f"皮损最大径约 {d1} mm"

    photo_str = f"皮肤光型：{phototype} 型" if phototype != "" else ""

    origin_str = ""
    if father_ori or mother_ori:
        origin_str = f"父籍贯：{father_ori}，母籍贯：{mother_ori}。"

    parts = []
    parts.append(f"{age}岁{sex_cn}，{region_cn}皮肤病变。")
    if size_str:
        parts.append(size_str + "。")
    if origin_str:
        parts.append(origin_str)

    parts.append(f"既往皮肤癌病史：{skin_ca}；其他恶性肿瘤病史：{other_ca}。")
    parts.append(f"生活方式：{smoke}，{drink}，{pesticide}。")
    parts.append(f"居住环境：{tap}，{sewer}。")
    if photo_str:
        parts.append(photo_str + "。")

    parts.append(
        f"症状体征：瘙痒{pruritus}，是否长大{growth}，疼痛{pain}，"
        f"形态变化{morph_change}，出血{bleeding}，隆起{elevated}。"
    )

    return "".join(parts)


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
    m = re.search(r"\b(akiec|bcc|bkl|nev|nv|mel)\b", text_lower)
    if not m:
        return "unknown"
    code = m.group(1)
    code = normalize_dx(code)
    return code if code in ALLOWED_DX else "unknown"


# ========== 加载基座模型（不加载 LoRA） ==========

def load_baseline_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    print("🔧 加载 MedGEMMA 基座模型（未微调版本） ...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    return model, processor, device


# ========== 使用 Test 集评估基座模型 ==========

def evaluate_baseline_on_test():
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"未找到测试集 CSV: {TEST_CSV}\n"
            f"请先运行你的微调脚本（包含 prepare_splits()）生成 *_test_5cls.csv。"
        )

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"📄 从 Test CSV 读取 {len(df)} 条样本: {TEST_CSV}")

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError("TEST_CSV 中缺少 image_id 或 dx 列")

    model, processor, device = load_baseline_model_and_processor()

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

        # prompt：和微调时保持同一个任务定义，只是模型未微调
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a dermatology assistant. "
                            "Given the clinical note and the skin lesion image, "
                            "your task is to classify the skin lesion into one of the following dx codes: "
                            "akiec, bcc, bkl, nev, mel. "
                            "Always answer with exactly one lowercase code "
                            "(akiec/bcc/bkl/nev/mel), no explanations."
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
                            "临床病历摘要如下：\n"
                            f"{clinical_note}\n\n"
                            "请结合病历和下方的皮肤病变图像，判断该病变最可能属于哪一类，"
                            "并只输出一个英文小写 dx 代码（akiec/bcc/bkl/nev/mel），"
                            "不要输出任何其他字符："
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

        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[:, input_len:]
        gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_label = extract_dx_code(gen_text)

        total += 1
        if pred_label == label_raw:
            correct += 1

        y_true.append(label_raw)
        y_pred.append(pred_label)

        print(
            f"🩺 [{total}] image_id={image_id} | pred={pred_label} | true={label_raw} "
            f"| {'✅' if pred_label == label_raw else '❌'} | raw={gen_text!r}"
        )

    print("\n====== 📊 基座模型在 Test 集上的评估结果 ======")
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
    ax_cm.set_title("Confusion Matrix (Baseline, 5 classes)")
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
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"📁 混淆矩阵图已保存到: {cm_path}")

    # ROC & PR（同样使用 one-hot 预测当作 score 近似）
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
    ax_roc.set_title("ROC Curves (Baseline, 5 classes, pseudo-scores)")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
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
    ax_pr.set_title("Precision-Recall Curves (Baseline, 5 classes, pseudo-scores)")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, "pr_curve.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 P-R 曲线图已保存到: {pr_path}")


if __name__ == "__main__":
    warnings.filterwarnings("once")
    evaluate_baseline_on_test()
