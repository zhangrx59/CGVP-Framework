# inference_medgamma_lora_treatment.py
# -*- coding: utf-8 -*-
"""
封装好的推理模块：
  - load_lora_model_and_processor
  - build_clinical_note
  - run_classification
  - run_treatment_generation

server.py 可以直接从这里 import。
"""

import os
import re
from typing import Tuple, Any

import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# ===================== 0. 基本配置 =====================

# 和你训练时用的一致
BASE_MODEL = "google/medgemma-4b-it"

# 这里改成你 LoRA 的输出目录（你现在是 lab4）
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lab4"

# 列名配置（和训练脚本一致）
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

# 现在有效的 4 类标签（你已经放弃 bkl）
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]


# ===================== 1. 病历文本构造（复制自你训练脚本） =====================

def yn_str(v, yes="yes", no="no", unk="unknown"):
    """
    把各种 True/False/空/NaN 归一化成 yes/no/unk 或你指定的描述
    """
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in ["TRUE", "T", "YES", "Y", "1", "是"]:
            return yes
        if vs in ["FALSE", "F", "NO", "N", "0", "否"]:
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
    构造英文病历描述（和训练/评估脚本保持一致）
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
        sex_en = "unknown sex"

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
    elevated = yn_str(row.get(COL_ELEVATED), "raised", "flat")

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


# ===================== 2. 构造对话（Stage-1 分类） =====================

def build_classification_messages(clinical_note: str, image: Image.Image) -> list:
    label_set_str = "{ " + ", ".join(ALLOWED_DX) + " }"
    label_codes_str = ", ".join(ALLOWED_DX)

    user_text = (
        f"Clinical note:\n{clinical_note}\n\n"
        "You are given the above clinical note together with a dermoscopic image of the lesion.\n"
        "Based on both the clinical information and the image, decide which label code best describes the lesion.\n\n"
        f"Valid label codes are: {label_codes_str}.\n\n"
        f"You MUST respond with ONLY ONE label code from {label_set_str}.\n"
        "Do NOT output a list of labels.\n"
        "Do NOT repeat the full disease names.\n"
        "Do NOT add any explanation or extra words.\n\n"
        "Final answer (ONLY ONE code):"
    )

    system_text = (
        "You are an expert dermatology assistant specialized in dermoscopic images.\n"
        "Your task is to classify a skin lesion based on a clinical note and a dermoscopic image.\n\n"
        "The label codes are:\n"
        " - akiec\n"
        " - bcc\n"
        " - nev\n"
        " - mel\n\n"
        "IMPORTANT OUTPUT RULES:\n"
        "1. For each case, you MUST output exactly ONE label code from this set: {akiec, bcc, nev, mel}.\n"
        "2. Do NOT output the full disease names.\n"
        "3. Do NOT output a list of all labels.\n"
        "4. Do NOT add explanations, probabilities, or any other text.\n"
        "5. The final answer must consist of a single code token only, e.g. 'bcc'.\n"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
    ]
    return messages


# ===================== 3. 构造对话（Stage-2 治疗方案） =====================

def build_treatment_messages(
    clinical_note: str,
    pred_dx: str,
    image: Image.Image,
) -> list:
    """
    使用我们上次优化过的英文 prompt，要求模型用中文输出
    个体化的“治疗依据+方案”。
    """
    system_text = (
        "You are a dermatologist assistant working in a research setting.\n"
        "You are given a clinical note, a dermoscopic image, and a diagnosis code\n"
        "from a previous automatic classifier (akiec, bcc, nev, mel).\n\n"
        "Your job is to produce a SHORT, high-quality, individualized summary in Chinese\n"
        "with a treatment rationale and a possible management plan.\n\n"
        "IMPORTANT RULES:\n"
        "1. The output is for EDUCATIONAL AND RESEARCH PURPOSES ONLY and is NOT real medical advice.\n"
        "2. You must strongly emphasize this in the Chinese text once.\n"
        "3. You must personalize the rationale and plan using patient-specific details\n"
        "   from the clinical note, such as:\n"
        "   - age,\n"
        "   - sex,\n"
        "   - lesion location,\n"
        "   - lesion size,\n"
        "   - history of skin cancer or other cancers,\n"
        "   - lifestyle and risk factors,\n"
        "   - symptoms and dynamic changes (itching, pain, growth, bleeding, morphology change),\n"
        "   and explain HOW these factors influence risk and management.\n"
        "4. Avoid generic, copy-and-paste plans. Different patients with the same code\n"
        "   should still have different details in the rationale and plan if their\n"
        "   clinical features differ.\n"
        "5. Do NOT give exact drug names, dosages, or schedules.\n"
        "6. The final answer MUST be written entirely in Chinese.\n"
    )

    user_text = (
        "You are given the following information about a single patient:\n\n"
        f"- Automatic diagnosis code: {pred_dx}\n"
        f"- Clinical note (English summary generated from the medical record):\n"
        f"  {clinical_note}\n\n"
        "Please carefully read the clinical note and use the diagnosis code as the working diagnosis.\n"
        "Then, in CHINESE, generate an individualized explanation with the following format:\n\n"
        "免责声明：...\n"
        "诊断：...\n"
        "治疗依据：...\n"
        "治疗方案：...\n\n"
        "Detailed requirements:\n"
        "1. In '免责声明：', clearly state that the content is an automatically generated draft\n"
        "   for research and education only.\n"
        "2. In '诊断：', briefly describe the condition in Chinese (you may mention the code).\n"
        "3. In '治疗依据：', explicitly mention at least 2–4 patient-specific factors and explain\n"
        "   how they influence risk and treatment choice.\n"
        "4. In '治疗方案：', propose a guideline-style plan adapted to this patient's risk profile.\n"
        "5. Do NOT provide specific drug regimens or dosages.\n"
        "6. Do NOT say you are an AI model.\n"
        "7. The whole answer must be in fluent, natural Chinese.\n"
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
    ]
    return messages


# ===================== 4. 加载 LoRA 模型 =====================

def load_lora_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    if device.type == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if supports_bf16:
            dtype = torch.bfloat16
            print("🔧 GPU supports bfloat16, use torch.bfloat16")
        else:
            dtype = torch.float16
            print("🔧 GPU does not support bfloat16, use torch.float16")
    else:
        dtype = torch.float32
        print("🔧 Using CPU, dtype=torch.float32")

    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(LORA_DIR)
    processor.tokenizer.padding_side = "right"

    return model, processor, device


# ===================== 5. 推理函数：分类 + 方案生成 =====================

def run_classification(
    model,
    processor: AutoProcessor,
    device: torch.device,
    clinical_note: str,
    image: Image.Image,
) -> str:
    messages = build_classification_messages(clinical_note, image)
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_text = processor.tokenizer.decode(
        gen_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    gen_text_lower = gen_text.lower()
    m = re.search(r"\b(akiec|bcc|nev|mel)\b", gen_text_lower)
    if m:
        pred = m.group(1)
    else:
        pred = "unknown"
        for cls in ALLOWED_DX:
            if cls in gen_text_lower:
                pred = cls
                break

    return pred


def run_treatment_generation(
    model,
    processor: AutoProcessor,
    device: torch.device,
    clinical_note: str,
    pred_dx: str,
    image: Image.Image,
) -> str:
    messages = build_treatment_messages(clinical_note, pred_dx, image)
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_text = processor.tokenizer.decode(
        gen_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return gen_text
