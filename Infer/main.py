# main.py
# -*- coding: utf-8 -*-
import io
import json
import os
from typing import Dict, Any, List, Optional

import re
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, PlainTextResponse  # ✅ CHANGED: 补充 PlainTextResponse
from pydantic import BaseModel

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# ===== 与训练/评估保持一致的 4 类顺序 =====
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]  # :contentReference[oaicite:3]{index=3}

# ⭐ NEW: 英文标签到中文疾病名称的映射，用于展示 & 报告
DX_ZH_MAP = {
    "akiec": "日光性角化病",
    "bcc": "基底细胞癌",
    "nev": "良性痣",
    "mel": "黑色素瘤",
}

# ===== 你 evaluate 里写死的默认超参 =====
DEFAULT_LOGIT_BIAS = [1.75, 1.0, 0.0, -1.0]  # :contentReference[oaicite:4]{index=4}
DEFAULT_MEL_THRESH = 0.55  # :contentReference[oaicite:5]{index=5}

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
        return unk
    return unk


def safe_str(x) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    return s if s else "unknown"


def safe_float_str(x) -> str:
    try:
        return f"{float(x):.1f}"
    except Exception:
        return ""


def _validate_required_fields(meta: Dict[str, Any]):
    """
    保证 meta 至少有 REQUIRED_FIELDS 这些键，没有就补 "unknown"
    """
    for k in REQUIRED_FIELDS:
        if k not in meta:
            meta[k] = "unknown"


def build_clinical_note(row: Dict[str, Any]) -> str:
    """
    把 20+ 个中文字段拼成英文的临床 note，用于喂进 MedGEMMA :contentReference[oaicite:7]{index=7}
    """
    age = safe_str(row.get("年龄"))
    sex = safe_str(row.get("性别"))  # 期待 "男"/"女" 或其他
    father_ori = safe_str(row.get("父籍贯"))
    mother_ori = safe_str(row.get("母籍贯"))

    smoke = yn_str(row.get("是否吸烟"), "yes", "no")
    drink = yn_str(row.get("是否饮酒"), "yes", "no")
    pesticide = yn_str(row.get("农药"), "exposed", "not_exposed")

    skin_cancer_hx = yn_str(row.get("皮肤癌病史"), "yes", "no")
    cancer_hx = yn_str(row.get("癌症病史"), "yes", "no")

    tap_water = yn_str(row.get("生活环境是否有自来水"), "yes", "no")
    sewer = yn_str(row.get("生活环境是否有下水道"), "yes", "no")

    phototype = safe_str(row.get("皮肤光型"))
    region = safe_str(row.get("区域"))

    d1 = safe_float_str(row.get("直径1"))
    d2 = safe_float_str(row.get("直径2"))

    itch = yn_str(row.get("瘙痒"), "present", "absent")
    grow = yn_str(row.get("是否长大"), "present", "absent")
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
            f"and the mother is from {mother_ori or 'unknown'}."
        )

    social_env_str = (
        f"Tap water: {tap_water}; sewer system: {sewer}. "
        f"Pesticide exposure: {pesticide}."
    )

    hx_str = (
        f"History of skin cancer: {skin_cancer_hx}. "
        f"History of other cancers: {cancer_hx}."
    )

    symptom_str = (
        f"Itching: {itch}, pain: {pain}, morphological change: {morph_change}, "
        f"bleeding: {bleeding}, elevated: {elevated}, growth: {grow}."
    )

    cn = (
            f"Patient is {age} years old, sex: {sex}. "
            f"Lesion location: {region_en}. "
            + size_str + " "
            + phototype_str + " "
            + origin_str + " "
            + social_env_str + " "
            + hx_str + " "
            + symptom_str
    )

    cn = re.sub(r"\s+", " ", cn).strip()
    return cn


class JsonInferReq(BaseModel):
    clinical_note: str
    logit_bias: Optional[List[float]] = None
    mel_thresh: Optional[float] = None


class JsonInferResp(BaseModel):
    pred: str
    probs: Dict[str, float]
    raw_text: str
    report: Dict[str, Any]


def _build_json_repair_prompt(pred_label: str, probs: Dict[str, float]) -> str:
    """
    当模型输出一段“乱七八糟带 JSON”的文本时，用这个 prompt 让它自己整理成**纯 JSON**。
    """
    probs_str = ", ".join([f"{k}:{float(v):.4f}" for k, v in probs.items()])
    return f"""
You are an assistant that only outputs one valid JSON object, nothing else.

The JSON must have exactly three top-level keys, all in Chinese:
- "预测标签": string, one of "akiec", "bcc", "nev", "mel"
- "概率": an object with keys "akiec","bcc","nev","mel" and float values
- "诊断报告": an object with 2 keys:
    - "诊断依据": string (Chinese)
    - "就诊建议": string (Chinese)

You will be given the model's raw output (possibly mixed with prompts, texts, incomplete JSON, etc.)
and the final classification result:

pred_label={pred_label}
probabilities={probs_str}

Now output STRICT JSON ONLY (Chinese values), starting with '{{' and ending with '}}':
""".strip()

# ⭐ CHANGED: 强化诊断依据/就诊建议提示，显式使用中文疾病名与病例要点
def _build_report_prompt_txt(clinical_note: str, pred_label: str, probs: dict) -> str:
    # ⭐ NEW: 将概率以「中文名称(英文代码):概率」形式提供给模型
    zh_probs_parts = []
    for k in ALLOWED_DX:
        v = float(probs.get(k, 0.0))
        zh_name = DX_ZH_MAP.get(k, k)
        zh_probs_parts.append(f"{zh_name}({k}):{v:.4f}")
    probs_str = ", ".join(zh_probs_parts)

    # ⭐ NEW: 预测类别的中文名
    zh_pred = DX_ZH_MAP.get(pred_label, pred_label)

    return f"""
        你是面向皮肤科医生的临床辅助决策助手，需要根据【病例描述】和【图像+分类结果】给出结构化的三行中文报告。

        你将获得以下信息（禁止编造，不得使用未提供的信息）：
        【病例描述】
        {clinical_note}

        【模型预测】
        预测类别：{zh_pred}（内部代码：{pred_label}）
        预测概率：{probs_str}

        =====================
        【合格示例】（必须学习其写法）
        预测标签：{{日光性角化病:0.0010 基底细胞癌:0.9900 良性痣:0.0050 黑色素瘤:0.0040}}
        诊断依据：模型预测基底细胞癌概率最高；病变位于暴露部位面部，大小约 8×6 mm，为慢性进展性局限性斑块/结节，边界清楚但可见表面光亮与毛细血管扩张，无明显瘙痒、疼痛或出血，这些特征与基底细胞癌常见的临床和皮肤镜表现相符，同时缺乏黑色素瘤所见的明显结构紊乱和多色性。
        就诊建议：建议尽快至皮肤科进一步完善皮肤镜及必要时切取活检以明确病理；如病理提示基底细胞癌，可根据病灶大小与部位选择局部手术切除或显微外科（莫氏手术），如无法手术可考虑局部药物或物理治疗（如外用免疫调节剂或光/放射/光动力治疗），具体方案由专科医生结合患者全身情况决定。

        【不合格示例】（严禁这样写）
        诊断依据：模型预测为基底细胞癌，概率为99%。

        =====================

        【你的任务】
        请严格模仿【合格示例】的风格，输出 **仅两行中文纯文本**，格式如下（前缀必须一致）：

        诊断依据：必须**综合说明**【模型预测结果】与【至少 2–3 个关键病例要点】，包括但不限于：病变部位、大小和形态、颜色/结构特点、是否进展或发生形态变化、有无瘙痒/疼痛/出血等阴性症状，以及这些信息如何支持或反对当前诊断和主要鉴别诊断。
        就诊建议：用 1–3 句话，面向皮肤科医生，需同时包含：
          - 进一步检查方案（如皮肤镜、活检、随访复查等）；
          - 至少三种可行的治疗路径（如随访观察、局部药物治疗、冷冻/激光/光动力、手术切除等），可举例药物的具体类别和治疗方式；
          - 如怀疑恶性病变（特别是黑色素瘤），需明确提出尽快手术切除或转诊专科中心的建议。
        表达需专业，不要使用口语化表达。

        注意：
        - 如果病例信息为“否/unknown”，请如实写“无/无法判断”或说明信息有限；
        - 禁止只写模型概率而不提病例细节；
        - 禁止输出任何与以上两行无关的额外内容（如“我是 AI 模型”“请咨询医生”等提示语）。
        """.strip()


def _extract_two_lines(raw: str) -> tuple[str, str]:
    """
    从模型 raw 输出中抽取：
      - 诊断依据：...
      - 就诊建议：...
    只返回内容部分（不含前缀）。
    若抽不到则兜底用最后两行/前两行。
    """
    # ✅ CHANGED: 更强的清洗，避免把 prompt/user/model 混进来
    if not raw:
        return "", ""

    text = raw.strip()

    # 如果有 markdown 代码块，先去掉 ```...``` 包裹
    if "```" in text:
        parts = text.split("```")
        # 通常模型会把内容放在第 2 段
        if len(parts) >= 2:
            text = parts[1].strip()

    # 去掉明显的 prompt 残留，例如 "User:", "Assistant:", "系统提示"
    # 这里只做一个简单规则：从第一个出现 "诊断依据" 或 "就诊建议" 的地方开始截断
    idx = len(text)
    for key in ["诊断依据", "就诊建议"]:
        k = text.find(key)
        if k != -1:
            idx = min(idx, k)
    if idx != len(text):
        text = text[idx:]

    m_idx = text.lower().rfind("\nmodel")
    if m_idx != -1:
        tail = text[m_idx:]
        if ("诊断依据" in tail) or ("就诊建议" in tail):
            text = tail

    basis = ""
    advice = ""

    m1 = re.search(r"诊断依据\s*[:：]\s*(.*)", text)
    if m1:
        basis = m1.group(1).strip()

    m2 = re.search(r"就诊建议\s*[:：]\s*(.*)", text)
    if m2:
        advice = m2.group(1).strip()

    if basis and advice:
        return basis, advice

    lines_ = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines_) >= 2:
        return lines_[0], lines_[1]
    if len(lines_) == 1:
        return lines_[0], ""
    return "", ""


def _extract_proba_json(raw_text: str) -> Optional[str]:
    """
    从模型输出中，尽量抽取出 {...} 这一段 JSON，并返回字符串；失败则 None
    """
    if not raw_text:
        return None

    if raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
        return raw_text.strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]

    for line in raw_text.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            return s

    text = raw_text
    idx = 10**9
    for w in ["{\"", "{\n\"", "{\r\n\""]:
        k = text.find(w)
        if k != -1:
            idx = min(idx, k)
    if idx != 10**9:
        text = text[idx:]

    m_idx = text.lower().rfind("\nmodel")
    if m_idx != -1:
        tail = text[m_idx:]
        if "{" in tail and "}" in tail:
            s = tail[tail.find("{"): tail.rfind("}") + 1]
            return s.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def _repair_json_dict(raw_text: str, pred_label: str, probs: Dict[str, float]) -> Dict[str, Any]:
    """
    若模型输出的 “JSON” 不合法，用第二次调用把它修成合法 JSON
    """
    s = _extract_proba_json(raw_text)
    if s is not None:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    prompt = _build_json_repair_prompt(pred_label, probs)
    fixed = _generate_report_text(prompt, None, max_new_tokens=260)
    try:
        return json.loads(fixed)
    except Exception:
        return {
            "预测标签": pred_label,
            "概率": probs,
            "诊断报告": {
                "诊断依据": "模型输出无法解析，仅保留分类结果，请结合临床判断。",
                "就诊建议": "建议至皮肤科就诊，结合病史、体检及必要的辅助检查综合判断。",
            },
        }


# ⭐ CHANGED: 让多模态 prompt 里真正包含图片占位符，避免 0 image tokens 错误
def _generate_report_text(prompt: str, image: Optional[Image.Image], max_new_tokens: int = 256) -> str:
    """
    调用 MedGEMMA / LoRA 生成一段文本（可能是 JSON，也可能是自然语言）
    这里根据是否有 image，构造带 image token 的 chat 模板。
    """
    global MODEL, PROCESSOR, DEVICE

    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("MODEL/PROCESSOR not loaded.")

    # ⭐ NEW: Gemma3 多模态推荐的 messages 结构
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # 这里会把 "image" 那一段转换成一个专门的 <image> token
    text = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # ⭐ CHANGED: 只有在 prompt 里有 image token 时才传 images
    if image is not None:
        inputs = PROCESSOR(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(DEVICE)
    else:
        inputs = PROCESSOR(
            text=[text],
            return_tensors="pt",
        ).to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen_text = PROCESSOR.decode(out[0], skip_special_tokens=True)
    return gen_text


# ⭐ CHANGED: 使用多模态 chat 模板，保证 prompt 中包含 image token
def _linear_logits_predict(
        image: Image.Image,
        clinical_note: str,
        logit_bias: Optional[List[float]] = None,
        mel_thresh: Optional[float] = None,
):
    """
    使用“线性 logits + 可调 bias + mel 阈值”来做 4 类分类
    保持与你原来的 evaluate 逻辑一致，只是改成 Gemma3 推荐的多模态 messages 写法。
    """
    global MODEL, PROCESSOR, DEVICE, LABEL_TOKEN_IDS, MEL_IDX

    if MODEL is None or PROCESSOR is None or LABEL_TOKEN_IDS is None:
        raise RuntimeError("MODEL/PROCESSOR/LABEL_TOKEN_IDS not loaded.")

    # ⭐ NEW: 多模态 messages，包含 image + text 两个片段
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a dermatology assistant.\n"
                        "Given the clinical note and the skin lesion image, "
                        "your task is to classify the lesion into one of the following classes: "
                        "akiec, bcc, nev, mel.\n"
                        "Always answer with exactly one lowercase class name in English.\n\n"
                        f"Clinical note:\n{clinical_note}\n"
                    ),
                },
            ],
        }
    ]

    # 让模板自动插入一个 <image> token
    prompt_text = PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # ⭐ CHANGED: prompt 里有 1 个 image token，因此这里传 1 张 image，数量匹配
    inputs = PROCESSOR(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # 取最后一步的 logits
    last_scores = out.scores[0][0]  # [vocab_size]

    # 只抽取我们关心的 4 个标签的 logits
    logits_4 = torch.stack([last_scores[idx] for idx in LABEL_TOKEN_IDS])

    # ⭐ CHANGED: 兼容可选 logit_bias / mel_thresh
    if logit_bias is None:
        bias = torch.tensor(DEFAULT_LOGIT_BIAS, device=logits_4.device, dtype=logits_4.dtype)
    else:
        if len(logit_bias) != 4:
            raise ValueError("logit_bias must have length 4")
        bias = torch.tensor(logit_bias, device=logits_4.device, dtype=logits_4.dtype)

    logits_biased = logits_4 + bias
    probs_4 = torch.softmax(logits_biased, dim=-1)  # [4]

    mel_idx = MEL_IDX
    mel_prob = float(probs_4[mel_idx].item())
    thresh = DEFAULT_MEL_THRESH if mel_thresh is None else float(mel_thresh)

    pred_idx = int(torch.argmax(probs_4).item())
    # ⭐ 保留你原来“mel 自信度不够就降权重选其他类”的逻辑
    if pred_idx == mel_idx and mel_prob < thresh:
        tmp = probs_4.clone()
        tmp[mel_idx] = -1e9
        pred_idx = int(torch.argmax(tmp).item())

    pred_label = ALLOWED_DX[pred_idx]
    probs = {c: float(p) for c, p in zip(ALLOWED_DX, probs_4.detach().float().cpu().tolist())}
    return pred_label, probs


@app.on_event("startup")
def _load_model():
    global MODEL, PROCESSOR, DEVICE, LABEL_TOKEN_IDS

    DEVICE, dtype = _pick_device_dtype()

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=None,
    )

    MODEL = PeftModel.from_pretrained(
        base_model,
        LORA_DIR,
        torch_dtype=dtype,
    ).to(DEVICE)

    PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL)

    label_token_ids = {}
    for cls in ALLOWED_DX:
        tokens = PROCESSOR.tokenizer.encode(cls, add_special_tokens=False)
        if not tokens:
            raise RuntimeError(f"Cannot encode label {cls} into token ID.")
        label_token_ids[cls] = tokens[0]

    LABEL_TOKEN_IDS = [label_token_ids[c] for c in ALLOWED_DX]
    print(f"[startup] MODEL loaded on {DEVICE}, label token ids = {LABEL_TOKEN_IDS}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "labels": ALLOWED_DX}


@app.post("/infer_json", response_model=JsonInferResp)
def infer_json(req: JsonInferReq):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    pil = Image.new("RGB", (224, 224), color=(128, 128, 128))

    pred, probs = _linear_logits_predict(
        pil,
        req.clinical_note,
        logit_bias=req.logit_bias,
        mel_thresh=req.mel_thresh,
    )

    prompt = _build_report_prompt_txt(req.clinical_note, pred, probs)
    raw_text = _generate_report_text(prompt, pil, max_new_tokens=300)

    report_dict = _repair_json_dict(raw_text, pred, probs)

    return JsonInferResp(
        pred=pred,
        probs=probs,
        raw_text=raw_text,
        report=report_dict,
    )


@app.post("/infer_report_multipart_txt", response_class=PlainTextResponse)  # ✅ CHANGED: 明确返回纯文本
async def infer_report_multipart_txt(
        meta_json: UploadFile = File(...),
        image: UploadFile = File(...),
        logit_bias: Optional[str] = Query(None),
        mel_thresh: Optional[float] = Query(None),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await meta_json.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"meta_json 不是合法 UTF-8 JSON: {e}")

    _validate_required_fields(meta)

    if "临床描述" in meta and isinstance(meta["临床描述"], str):
        clinical_note_extra = meta["临床描述"]
    else:
        clinical_note_extra = ""

    row_for_cn = dict(meta)
    cn_base = build_clinical_note(row_for_cn)
    if clinical_note_extra:
        cn = cn_base + " Additional note: " + clinical_note_extra
    else:
        cn = cn_base

    clinical_note = cn

    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image 读取失败: {e}")

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

    clinical_note = build_clinical_note(meta)
    pred, probs = _linear_logits_predict(pil, clinical_note, bias, thr)

    # ⭐ CHANGED: 使用中文疾病名构造概率行
    probs_line = " ".join(
        f"{DX_ZH_MAP.get(k, k)}:{float(probs.get(k, 0.0)):.6f}"
        for k in ALLOWED_DX
    )

    report_prompt = _build_report_prompt_txt(clinical_note, pred, probs)
    raw_text = _generate_report_text(report_prompt, pil)

    basis, advice = _extract_two_lines(raw_text)

    if not basis:
        basis = "依据图像特征与模型分类结果综合判断，建议结合皮肤镜/病理检查进一步确认。"
    if not advice:
        advice = "建议至皮肤科就诊，完善皮肤镜检查，必要时进行活检以明确诊断。"

    txt = (
        f"预测标签：{{{probs_line}}}\n"
        f"诊断依据：{basis}\n"
        f"就诊建议：{advice}\n"
    )
    return PlainTextResponse(txt)  # ✅ CHANGED


def _pick_device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # :contentReference[oaicite:19]{index=19}
    if device.type == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _repair_to_json_only(raw_text: str) -> str:
    return f"""
You will receive a model output that may contain extra text or invalid/incomplete JSON.

Task:
- Output ONE valid JSON object only.
- NO markdown, NO code fences, NO explanations, NO extra text.
- All explanatory text inside JSON values MUST be Chinese.
- Keep it concise.

The JSON structure you MUST output:

{{
  "预测标签": "<string: akiec|bcc|nev|mel>",
  "概率": {{
    "akiec": <float>,
    "bcc": <float>,
    "nev": <float>,
    "mel": <float>
  }},
  "诊断报告": {{
    "诊断依据": "<Chinese string>",
    "就诊建议": "<Chinese string>"
  }}
}}

Now, based on the following raw model output, output ONE valid JSON object ONLY:
{raw_text}
""".strip()


@app.post("/infer_report_multipart")
async def infer_report_multipart(
        meta_json: UploadFile = File(...),
        image: UploadFile = File(...),
        logit_bias: Optional[str] = Query(None),
        mel_thresh: Optional[float] = Query(None),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await meta_json.read()
    try:
        meta = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"meta_json 不是合法 UTF-8 JSON: {e}")

    _validate_required_fields(meta)

    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image 读取失败: {e}")

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

    clinical_note = build_clinical_note(meta)
    pred, probs = _linear_logits_predict(pil, clinical_note, bias, thr)

    # ⭐ CHANGED: 使用中文疾病名构造概率行
    probs_line = " ".join(
        f"{DX_ZH_MAP.get(k, k)}:{float(probs.get(k, 0.0)):.6f}"
        for k in ALLOWED_DX
    )

    report_prompt = _build_report_prompt_txt(clinical_note, pred, probs)
    raw_text = _generate_report_text(report_prompt, pil, max_new_tokens=220)  # 直接生成纯文本

    out = f"预测标签：{{{probs_line}}}\n{raw_text.strip()}\n"

    return Response(content=out, media_type="text/plain; charset=utf-8")
