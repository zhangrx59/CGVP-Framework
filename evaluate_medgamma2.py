# evaluate_medgamma_reasoning.py
# -*- coding: utf-8 -*-
"""
在原 evaluate_medgamma.py（4分类，无 bkl）的基础上做“最小侵入”的增强：
- 评估指标/混淆矩阵/ROC/PR：仍然使用你原来的“线性推理（取 final logits -> 4 个 label token）”逻辑，保证可复现。
- 额外输出：对每个样本再调用一次 generate()，生成“推理依据 + 治疗建议（含用药方向）”。

重要说明（医疗安全）：
本脚本生成的“治疗/用药建议”仅做技术演示与信息整理，不构成医疗处方或个体化治疗方案；
临床决策必须由有资质的医生结合病史、体检、皮肤镜/病理等结果给出。
"""

import os
import re
import json
import argparse
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
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize


# ========== 路径 & 配置（需与微调脚本一致） ==========

BASE_MODEL = "google/medgemma-4b-it"

# 原始 metadata CSV（不一定用得到，只是留档）
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 微调脚本中 prepare_splits() 生成的 test CSV
TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# LoRA 权重所在目录（要和微调脚本里的 OUTPUT_DIR 一致）
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lab5"

# 图像根目录和后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"  # 如果是 .jpg 改成 ".jpg"

# 评估图像保存目录
LORA_PLOTS_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lab5_results"
os.makedirs(LORA_PLOTS_DIR, exist_ok=True)

# 额外：逐样本推理输出保存目录
LORA_TEXT_DIR = os.path.join(LORA_PLOTS_DIR, "reasoning_outputs")
os.makedirs(LORA_TEXT_DIR, exist_ok=True)

# 列名（与微调脚本保持一致）
COL_IMAGE_ID = "image_id"
COL_AGE = "年龄"
COL_SEX = "性别"
COL_FATHER_ORI = "父籍贯"
COL_MOTHER_ORI = "母籍贯"
COL_BIOPSY = "是否活检"
COL_SMOKE = "是否吸烟"
COL_DRINK = "是否饮酒"
COL_PESTICIDE = "农药"
COL_SKIN_CANCER = "皮肤癌病史"
COL_OTHER_CA = "癌症病史"
COL_TAP_WATER = "生活环境是否有自来水"
COL_SEWER = "生活环境是否有下水道"
COL_PHOTOTYPE = "皮肤光型"
COL_REGION = "区域"
COL_D1 = "直径1"
COL_D2 = "直径2"
COL_PRURITUS = "瘙痒"
COL_GROWTH = "是否长大"
COL_PAIN = "疼痛"
COL_MORPH_CHANGE = "形态变化"
COL_BLEEDING = "出血"
COL_ELEVATED = "是否隆起"

COL_TARGET = "dx"

# 现在只评估这 4 类（注意顺序必须固定）
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]


# ========== 一些工具函数（和微调脚本保持一致） ==========

def yn_str(v, yes: str, no: str, unk: str = "unknown") -> str:
    """把各种 True/False/空/NaN 归一化成 yes/no/unk"""
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
    """与训练脚本中的英文病历构造逻辑保持一致"""
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
    """从模型输出文本中提取 4 类 dx code（备用函数）"""
    if not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    m = re.search(r"\b(mel|bcc|nev|nv|akiec)\b", text_lower)
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
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(LORA_DIR)
    processor.tokenizer.padding_side = "right"

    return model, processor, device


# ========== 新增：生成“依据 + 治疗建议”的推理文本 ==========

def build_reasoning_prompt(clinical_note: str) -> list:
    """
    生成专用于“解释 + 治疗建议”的对话 messages。
    注意：这里不要求模型输出仅一个类别了，而是输出中文结构化内容。
    """
    sys_text = (
        "你是一名皮肤科辅助医生（仅做信息整理与科普，不替代临床医生）。"
        "你会根据病历摘要与皮损图像给出：\n"
        "1) 诊断分类（必须是以下四类之一：akiec, bcc, nev, mel）\n"
        "2) 你的判断依据（列出图像/病史可能支持该分类的关键点）\n"
        "3) 下一步检查建议\n"
        "4) 治疗建议（可以给出常见用药/治疗方向，但不要给剂量与处方；强调需医生评估）\n\n"
        "输出格式必须严格为：\n"
        "诊断分类: <class>\n"
        "依据:\n- ...\n"
        "检查建议:\n- ...\n"
        "治疗建议:\n- ...\n"
        "风险提示:\n- ...\n"
    )

    user_text = (
        f"病历摘要:\n{clinical_note}\n\n"
        "请结合图像与病历，按要求输出。"
        "再次强调：不要给出具体剂量、疗程或处方；只给常见治疗方向，并提示就医。"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image"}]},
    ]
    return messages


@torch.no_grad()
def generate_reasoning_and_plan(
    model,
    processor,
    device,
    image: Image.Image,
    clinical_note: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """
    额外跑一次 generate() 输出“原因 + 治疗建议”。
    这一步不参与指标计算，仅用于可解释性展示/案例分析。
    """
    messages = build_reasoning_prompt(clinical_note)
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

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    out_text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # 尝试只截取 assistant 的最后输出（保守处理：找最后一次出现“诊断分类:”）
    idx = out_text.rfind("诊断分类")
    if idx != -1:
        out_text = out_text[idx:].strip()
    return out_text


# ========== 逐样本线性推理评估 Test 集（直接看 final logits） +（可选）生成解释/治疗建议 ==========

def evaluate_lora_linear(
    do_generate_reasoning: bool = True,
    reasoning_max_new_tokens: int = 256,
    reasoning_temperature: float = 0.2,
    reasoning_top_p: float = 0.9,
    save_reasoning_jsonl: bool = True,
):
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"未找到测试集 CSV: {TEST_CSV}\n"
            f"请先运行微调脚本生成 *_test.csv。"
        )

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    print(f"📄 从 Test CSV 读取 {len(df)} 条样本: {TEST_CSV}")

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError("TEST_CSV 中缺少 image_id 或 dx 列")

    # 仅保留 4 类，避免有旧的 bkl 干扰
    df["dx"] = df[COL_TARGET].apply(normalize_dx)
    df = df[df["dx"].isin(ALLOWED_DX)].reset_index(drop=True)
    print("📊 按 4 类过滤后的标签分布:", df["dx"].value_counts().to_dict())

    model, processor, device = load_lora_model_and_processor()

    y_true, y_pred = [], []
    total, correct = 0, 0
    missing_image = 0

    # === ① 计算 4 个类别的首 token id（与训练保持一致） ===
    label_token_ids = []
    for cls in ALLOWED_DX:
        ids = processor.tokenizer(cls, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            raise ValueError(f"标签 {cls} tokenizer 结果为空")
        label_token_ids.append(ids[0])
    label_token_ids = torch.tensor(label_token_ids, device=device)  # shape (4,)
    print("🔧 4 类标签的首 token id:", label_token_ids.tolist())

    # === ② 一层 logit bias + mel 阈值的超参（你原代码保留） ===
    # 顺序对应 ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]
    logit_bias = torch.tensor([0.2, 0.0, 0.2, -0.5], device=device)
    mel_idx = ALLOWED_DX.index("mel")
    mel_thresh = 0.6  # 只有当 mel 概率 >= 0.6 时才允许预测为 mel

    # 可选：把逐样本解释写到 jsonl，便于后处理
    jsonl_fp = None
    if save_reasoning_jsonl and do_generate_reasoning:
        jsonl_path = os.path.join(LORA_TEXT_DIR, "reasoning_outputs.jsonl")
        jsonl_fp = open(jsonl_path, "w", encoding="utf-8")
        print(f"📝 逐样本解释/建议将保存到: {jsonl_path}")

    for _, row in df.iterrows():
        image_id = str(row[COL_IMAGE_ID])
        label_raw = normalize_dx(str(row["dx"]))

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
        # （这里保持你原评估逻辑，方便和历史结果对齐）
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
                            "akiec, bcc, nev, mel. "
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
                            "akiec, bcc, nev, mel.\n"
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
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        logits_4 = last_logits[label_token_ids]  # (4,)

        # === ① 一层 logit bias 调整 ===
        logits_4 = logits_4 + logit_bias

        # === ② mel 阈值处理：如果 mel 概率不够高，则压一压 mel 再选一次 ===
        probs_4 = torch.softmax(logits_4, dim=-1)
        pred_idx = int(torch.argmax(probs_4).item())

        if pred_idx == mel_idx and probs_4[mel_idx] < mel_thresh:
            tmp_logits = logits_4.clone()
            tmp_logits[mel_idx] -= 1.0
            probs_4 = torch.softmax(tmp_logits, dim=-1)
            pred_idx = int(torch.argmax(probs_4).item())

        pred_label = ALLOWED_DX[pred_idx]

        total += 1
        is_ok = (pred_label == label_raw)
        if is_ok:
            correct += 1

        y_true.append(label_raw)
        y_pred.append(pred_label)

        probs_np = probs_4.detach().float().cpu().numpy().reshape(-1)
        probs_str = np.array2string(probs_np, precision=4, separator=" ", suppress_small=False)

        print(
            f"🩺 [{total}] image_id={image_id} | pred={pred_label} | true={label_raw} "
            f"| {'✅' if is_ok else '❌'} | probs={probs_str}"
        )

        # ===== 可选：额外生成“依据+治疗建议” =====
        reasoning_text = ""
        if do_generate_reasoning:
            try:
                reasoning_text = generate_reasoning_and_plan(
                    model=model,
                    processor=processor,
                    device=device,
                    image=image,
                    clinical_note=clinical_note,
                    max_new_tokens=reasoning_max_new_tokens,
                    temperature=reasoning_temperature,
                    top_p=reasoning_top_p,
                )
            except Exception as e:
                reasoning_text = f"[GEN_ERROR] {repr(e)}"

            # 也可以在控制台打印一小段，避免刷屏
            preview = reasoning_text.replace("\n", " ")[:180]
            print(f"   🧾 reasoning preview: {preview}...")

            if jsonl_fp is not None:
                rec = {
                    "image_id": image_id,
                    "true": label_raw,
                    "pred": pred_label,
                    "probs": probs_np.tolist(),
                    "clinical_note": clinical_note,
                    "reasoning": reasoning_text,
                }
                jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if jsonl_fp is not None:
        jsonl_fp.close()

    print("\n====== 📊 LoRA 模型在线性推理下的 Test 集评估结果（4 类） ======")
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

    print("📊 y_true 标签分布:", {c: int((y_true_arr == c).sum()) for c in classes})
    print("📊 y_pred 标签分布:", {c: int((y_pred_arr == c).sum()) for c in classes})

    print("\n====== 📊 classification_report ======")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=classes,
            target_names=classes,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    print("\n====== 📊 混淆矩阵（rows=true, cols=pred） ======")
    print(classes)
    print(cm)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        cmap=plt.cm.Blues,
        ax=ax_cm,
        values_format="d",
        colorbar=True,
    )
    ax_cm.set_title("Confusion Matrix (LoRA, 4 classes, linear)")
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    fig_cm.tight_layout()
    cm_path = os.path.join(LORA_PLOTS_DIR, "confusion_matrix_LoRA_linear_4cls.png")
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
    ax_roc.set_title("ROC Curves (LoRA, 4 classes, pseudo-scores, linear)")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(LORA_PLOTS_DIR, "roc_curve_LoRA_linear_4cls.png")
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
    ax_pr.set_title("Precision-Recall Curves (LoRA, 4 classes, pseudo-scores, linear)")
    ax_pr.legend(loc="lower left", fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(LORA_PLOTS_DIR, "pr_curve_LoRA_linear_4cls.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 P-R 曲线图已保存到: {pr_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", type=str, default=TEST_CSV)
    p.add_argument("--lora_dir", type=str, default=LORA_DIR)
    p.add_argument("--image_root", type=str, default=IMAGE_ROOT_DIR)
    p.add_argument("--image_ext", type=str, default=IMAGE_EXT)
    p.add_argument("--plots_dir", type=str, default=LORA_PLOTS_DIR)

    p.add_argument("--no_reasoning", action="store_true", help="不额外生成“依据+治疗建议”，只跑分类评估")
    p.add_argument("--reasoning_max_new_tokens", type=int, default=256)
    p.add_argument("--reasoning_temperature", type=float, default=0.2)
    p.add_argument("--reasoning_top_p", type=float, default=0.9)
    p.add_argument("--no_save_jsonl", action="store_true", help="不保存逐样本 jsonl 输出")
    return p.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("once")
    set_seed(42)

    args = parse_args()

    # 允许用命令行覆盖路径（不破坏原默认值）
    TEST_CSV = args.test_csv
    LORA_DIR = args.lora_dir
    IMAGE_ROOT_DIR = args.image_root
    IMAGE_EXT = args.image_ext
    LORA_PLOTS_DIR = args.plots_dir
    os.makedirs(LORA_PLOTS_DIR, exist_ok=True)
    LORA_TEXT_DIR = os.path.join(LORA_PLOTS_DIR, "reasoning_outputs")
    os.makedirs(LORA_TEXT_DIR, exist_ok=True)

    evaluate_lora_linear(
        do_generate_reasoning=(not args.no_reasoning),
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        reasoning_temperature=args.reasoning_temperature,
        reasoning_top_p=args.reasoning_top_p,
        save_reasoning_jsonl=(not args.no_save_jsonl),
    )
