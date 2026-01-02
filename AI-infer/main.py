# main.py
# -*- coding: utf-8 -*-
import io
import json
import os
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


# ===== 与训练/评估保持一致的 4 类顺序 =====
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]  # :contentReference[oaicite:3]{index=3}

# ===== 你 evaluate 里写死的默认超参 =====
DEFAULT_LOGIT_BIAS = [1.75, 1.0, 0.0, -1.0]  # :contentReference[oaicite:4]{index=4}
DEFAULT_MEL_THRESH = 0.55                   # :contentReference[oaicite:5]{index=5}

# ===== 你要求严格包含的 20 个字段（键名就是中文）=====
REQUIRED_FIELDS = [
    "年龄", "性别", "父籍贯", "母籍贯", "是否吸烟", "是否饮酒", "农药",
    "皮肤癌病史", "癌症病史", "生活环境是否有自来水", "生活环境是否有下水道",
    "皮肤光型", "区域", "直径1", "直径2", "瘙痒", "是否长大", "疼痛",
    "形态变化", "出血", "是否隆起"
]
# 注意：你给的是 20 行，但这里数下来是 21 个键名（包含“农药”也算一个字段）。
# 你上面列表里确实有 21 项（COL_PESTICIDE 也算）。
# 我这里按你贴的“严格包括下列字段”全部收齐了。


# ===== 模型路径：你自己改成你本机 LoRA 输出目录 =====
BASE_MODEL = "google/medgemma-4b-it"
LORA_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lab8"  # 改为你实际 LoRA 目录


app = FastAPI(title="MedGEMMA-LoRA Inference (multipart)", version="1.0.0")

MODEL = None
PROCESSOR = None
DEVICE = None
LABEL_TOKEN_IDS = None
MEL_IDX = ALLOWED_DX.index("mel")


def yn_str(v, yes: str, no: str, unk: str = "unknown") -> str:
    """
    与 evaluate_medgemma.py 一致：把 True/False/空/NaN 归一化成 yes/no/unk :contentReference[oaicite:6]{index=6}
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
    try:
        if v != v:  # NaN
            return unk
    except Exception:
        pass
    return str(v)


def build_clinical_note(row: Dict[str, Any]) -> str:
    """
    复刻 evaluate_medgemma.py 的英文病历拼接逻辑（保证对齐）:contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
    """
    age = row.get("年龄", "")
    sex_raw = str(row.get("性别", "") or "").strip().lower()
    region = str(row.get("区域", "") or "").strip()
    father_ori = str(row.get("父籍贯", "") or "").strip()
    mother_ori = str(row.get("母籍贯", "") or "").strip()

    # 性别英文化 :contentReference[oaicite:9]{index=9}
    if sex_raw in ["男", "male", "m"]:
        sex_en = "male"
    elif sex_raw in ["女", "female", "f"]:
        sex_en = "female"
    else:
        sex_en = "unknown"

    skin_ca = yn_str(row.get("皮肤癌病史"), "yes", "no")
    other_ca = yn_str(row.get("癌症病史"), "yes", "no")
    smoke = yn_str(row.get("是否吸烟"), "yes", "no")
    drink = yn_str(row.get("是否饮酒"), "yes", "no")
    pesticide = yn_str(row.get("农药"), "yes", "no")

    tap = yn_str(row.get("生活环境是否有自来水"), "yes", "no")
    sewer = yn_str(row.get("生活环境是否有下水道"), "yes", "no")

    phototype = row.get("皮肤光型", "")
    d1 = row.get("直径1", "")
    d2 = row.get("直径2", "")

    pruritus = yn_str(row.get("瘙痒"), "present", "absent")
    growth = yn_str(row.get("是否长大"), "present", "absent")
    pain = yn_str(row.get("疼痛"), "present", "absent")
    morph_change = yn_str(row.get("形态变化"), "present", "absent")
    bleeding = yn_str(row.get("出血"), "present", "absent")
    elevated = yn_str(row.get("是否隆起"), "present", "absent")

    region_en = region if region else "unknown region"

    size_str = ""
    if d1 and d2:
        size_str = f"Lesion size is about {d1} by {d2} mm."  # :contentReference[oaicite:10]{index=10}

    phototype_str = f"Fitzpatrick skin phototype: {phototype}." if phototype != "" else ""  # :contentReference[oaicite:11]{index=11}

    origin_str = ""
    if father_ori or mother_ori:
        origin_str = (
            f"The patient's father is from {father_ori or 'unknown'}, "
            f"and mother is from {mother_ori or 'unknown'}."
        )  # :contentReference[oaicite:12]{index=12}

    parts = []
    parts.append(f"{age}-year-old {sex_en} with a skin lesion on the {region_en}.")  # :contentReference[oaicite:13]{index=13}
    if size_str:
        parts.append(size_str)
    if origin_str:
        parts.append(origin_str)

    parts.append(f"Past history of skin cancer: {skin_ca}; other malignancies: {other_ca}.")  # :contentReference[oaicite:14]{index=14}
    parts.append(f"Lifestyle: smoking {smoke}, alcohol {drink}, pesticide exposure {pesticide}.")  # :contentReference[oaicite:15]{index=15}
    parts.append(f"Living environment: tap water {tap}, sewer system {sewer}.")  # :contentReference[oaicite:16]{index=16}
    if phototype_str:
        parts.append(phototype_str)

    parts.append(
        "Current symptoms and signs: "
        f"pruritus {pruritus}, growth {growth}, pain {pain}, "
        f"morphologic change {morph_change}, bleeding {bleeding}, "
        f"elevation {elevated}."
    )  # :contentReference[oaicite:17]{index=17}

    return " ".join(parts)  # :contentReference[oaicite:18]{index=18}


def _pick_device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # :contentReference[oaicite:19]{index=19}
    if device.type == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


@app.on_event("startup")
def _startup_load_model():
    global MODEL, PROCESSOR, DEVICE, LABEL_TOKEN_IDS

    DEVICE, dtype = _pick_device_dtype()

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    )
    MODEL = PeftModel.from_pretrained(base_model, LORA_DIR).to(DEVICE)
    MODEL.eval()

    PROCESSOR = AutoProcessor.from_pretrained(LORA_DIR)
    PROCESSOR.tokenizer.padding_side = "right"  # :contentReference[oaicite:20]{index=20}

    # 计算 4 类标签首 token id（与 evaluate 保持一致）:contentReference[oaicite:21]{index=21}
    ids = []
    for cls in ALLOWED_DX:
        tok = PROCESSOR.tokenizer(cls, add_special_tokens=False)["input_ids"]
        if not tok:
            raise RuntimeError(f"label '{cls}' token ids empty")
        ids.append(tok[0])
    LABEL_TOKEN_IDS = torch.tensor(ids, device=DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "modelLoaded": MODEL is not None}


def _build_prompt(clinical_note: str, image: Image.Image) -> str:
    """
    复刻 evaluate 的 chat_template，末尾必须含 Final answer: :contentReference[oaicite:22]{index=22}
    """
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

    return PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _linear_logits_predict(image: Image.Image, clinical_note: str,
                           logit_bias: List[float], mel_thresh: float) -> (str, Dict[str, float]):
    """
    严格复刻 evaluate 的线性 logits 推理：last_logits -> logits_4 -> +bias -> softmax -> mel gate :contentReference[oaicite:23]{index=23}
    """
    prompt_text = _build_prompt(clinical_note, image)

    inputs = PROCESSOR(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)  # 不 generate :contentReference[oaicite:24]{index=24}

    last_logits = outputs.logits[0, -1, :]             # :contentReference[oaicite:25]{index=25}
    logits_4 = last_logits[LABEL_TOKEN_IDS]            # :contentReference[oaicite:26]{index=26}
    logits_4 = logits_4 + torch.tensor(logit_bias, device=DEVICE)

    probs_4 = torch.softmax(logits_4, dim=-1)
    pred_idx = int(torch.argmax(probs_4).item())

    if pred_idx == MEL_IDX and float(probs_4[MEL_IDX]) < mel_thresh:  # :contentReference[oaicite:27]{index=27}
        tmp_logits = logits_4.clone()
        tmp_logits[MEL_IDX] -= 1.0
        probs_4 = torch.softmax(tmp_logits, dim=-1)
        pred_idx = int(torch.argmax(probs_4).item())

    pred_label = ALLOWED_DX[pred_idx]
    probs = {c: float(p) for c, p in zip(ALLOWED_DX, probs_4.detach().float().cpu().tolist())}
    return pred_label, probs


def _build_report_json(meta: Dict[str, Any], pred_label: str, probs: Dict[str, float], clinical_note: str) -> Dict[str, Any]:
    """
    构建中文结构化报告
    """
    # 类别中文映射
    label_map = {
        "akiec": "光化性角化病",
        "bcc": "基底细胞癌",
        "nev": "痣",
        "mel": "黑色素瘤"
    }
    
    # 构建报告
    report = {
        "诊断结果": label_map.get(pred_label, pred_label),
        "诊断类别": pred_label,
        "各类别概率": {
            label_map.get("akiec", "akiec"): round(probs.get("akiec", 0.0) * 100, 2),
            label_map.get("bcc", "bcc"): round(probs.get("bcc", 0.0) * 100, 2),
            label_map.get("nev", "nev"): round(probs.get("nev", 0.0) * 100, 2),
            label_map.get("mel", "mel"): round(probs.get("mel", 0.0) * 100, 2),
        },
        "患者信息": {
            "年龄": meta.get("年龄", ""),
            "性别": meta.get("性别", ""),
            "区域": meta.get("区域", ""),
        },
        "临床特征": {
            "直径": f"{meta.get('直径1', '')} x {meta.get('直径2', '')} mm" if meta.get("直径1") and meta.get("直径2") else "",
            "瘙痒": meta.get("瘙痒", ""),
            "疼痛": meta.get("疼痛", ""),
            "是否长大": meta.get("是否长大", ""),
            "形态变化": meta.get("形态变化", ""),
            "出血": meta.get("出血", ""),
            "是否隆起": meta.get("是否隆起", ""),
        },
        "病史信息": {
            "皮肤癌病史": meta.get("皮肤癌病史", ""),
            "癌症病史": meta.get("癌症病史", ""),
            "是否吸烟": meta.get("是否吸烟", ""),
            "是否饮酒": meta.get("是否饮酒", ""),
        },
        "临床记录": clinical_note
    }
    
    return report


def _validate_required_fields(meta: Dict[str, Any]):
    missing = [k for k in REQUIRED_FIELDS if k not in meta]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"meta json 缺少字段: {missing} (必须严格包含全部字段)"
        )


@app.post("/infer_multipart")
async def infer_multipart(
    meta_json: UploadFile = File(..., description="病人信息 JSON 文件（必须含固定字段）"),
    image: UploadFile = File(..., description="皮肤病灶图片"),
    logit_bias: Optional[str] = Query(None, description="可选：形如 [1.75,1.0,0.0,-1.0] 的 JSON 字符串"),
    mel_thresh: Optional[float] = Query(None, description="可选：mel 阈值，默认 0.55"),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    # 1) 读 meta json 文件
    raw = await meta_json.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"meta_json 不是合法 UTF-8 JSON: {e}")

    _validate_required_fields(meta)

    # 2) 读图片
    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image 读取失败: {e}")

    # 3) 解析可选超参
    if logit_bias is None:
        bias = DEFAULT_LOGIT_BIAS
    else:
        try:
            bias = json.loads(logit_bias)
            if not (isinstance(bias, list) and len(bias) == 4):
                raise ValueError("logit_bias must be list of len=4")
            bias = [float(x) for x in bias]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"logit_bias 格式错误: {e}")

    thr = DEFAULT_MEL_THRESH if mel_thresh is None else float(mel_thresh)

    # 4) 构造 clinical_note（严格对齐）
    clinical_note = build_clinical_note(meta)

    # 5) 推理（严格对齐 evaluate）
    pred, probs = _linear_logits_predict(pil, clinical_note, bias, thr)

    # 6) 构建中文结构化报告
    report_json = _build_report_json(meta, pred, probs, clinical_note)

    return {
        "predLabel": pred,
        "probs": probs,
        "clinicalNote": clinical_note,  # 方便你联调时核对对齐（上线可删）
        "reportJson": report_json,
        "modelVersion": "medgemma-4b-it+lora@202512"
    }
