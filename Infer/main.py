# main.py
# -*- coding: utf-8 -*-
import io
import json
import os
from typing import Dict, Any, List, Optional
from fastapi.responses import Response
import re
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import json
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

def _extract_json_from_text(text: str):
    # 优先抓 ```json ... ```（如果模型仍然输出了 code fence）
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)

    # 其次：找第一个 { 到最后一个 }（适用于带前后废话）
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return None

def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

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



def _build_report_prompt(clinical_note: str, pred_label: str, probs: dict) -> str:
    probs_str = ", ".join([f"{k}:{float(v):.4f}" for k, v in probs.items()])

    return f"""
You are a clinical assistant writing a concise decision-support report for clinicians.

You will be given:
1) A clinical note (structured text)
2) A model classification result (predicted label + probabilities)

STRICT REQUIREMENTS:
- Output MUST be strict JSON only (no markdown, no code fences, no extra text, no schema, no repeating the prompt).
- The content of the JSON values MUST be in CHINESE.
  (You may keep labels as akiec/bcc/nev/mel, but all explanations must be Chinese.)
- Do NOT invent patient facts not present in the note. If uncertain, say "无法判断/需进一步检查".
- Keep it concise: 3–6 bullets per list.

JSON schema (field names must match exactly):
{{
  "pred_label": "akiec|bcc|nev|mel",
  "probabilities": {{"akiec":0-1,"bcc":0-1,"nev":0-1,"mel":0-1}},
  "诊断倾向": "一句话总结(中文)",
  "主要依据": ["..."],
  "图像观察要点": ["..."],
  "鉴别诊断": [{{"疾病":"...","理由":"..."}}],
  "不确定性与局限": ["..."],
  "建议下一步": ["..."],
  "免责声明": "..."
}}

Clinical note:
{clinical_note}

Model result:
pred_label={pred_label}
probabilities={probs_str}

Now output STRICT JSON ONLY (Chinese values), starting with '{{' and ending with '}}':
""".strip()


def _build_report_prompt_txt(clinical_note: str, pred_label: str, probs: dict) -> str:
    probs_str = ", ".join([f"{k}:{float(v):.4f}" for k, v in probs.items()])
    return f"""
        你是临床辅助决策助手。

        你将获得以下信息（禁止编造，不得使用未提供的信息）：
        【病例描述】
        {clinical_note}

        【模型预测】
        预测类别：{pred_label}
        预测概率：{probs_str}

        =====================
        【合格示例】（必须学习其写法）
        预测标签：akiec:0.0010 bcc:0.9900 nev:0.0050 mel:0.0040
        诊断依据：模型预测基底细胞癌概率较高；病例为面部皮损，尺寸约8×6 mm，存在生长及形态改变但无出血和疼痛，符合基底细胞癌常见临床表现。
        就诊建议：建议皮肤科就诊，行皮肤镜检查；如存在可疑特征，建议进一步活检明确诊断。

        【不合格示例】（严禁这样写）
        诊断依据：模型预测为基底细胞癌，概率为99%。

        =====================

        【你的任务】
        请严格模仿【合格示例】的风格，输出 **仅三行中文纯文本**，格式如下（前缀必须一致）：

        预测标签：akiec:... bcc:... nev:... mel:...
        诊断依据：必须同时包含【模型结果】+【至少两个病例要点（如部位/大小/症状/变化/否定症状）】
        就诊建议：1-2句话；不要给出具体药物剂量；可建议皮肤科就诊、皮肤镜检查、必要时活检

        注意：
        - 如果病例信息为“否/unknown”，请如实写“无/无法判断”
        - 禁止只写模型概率而不提病例
        - 不要输出任何多余内容
        """.strip()

def _extract_two_lines(text: str) -> (str, str):
    """
    从模型输出里尽量稳地抽出“诊断依据/就诊建议”两行，允许模型有少量废话。
    """
    # 先统一换行
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 尝试直接按标签提取
    m1 = re.search(r"诊断依据[:：]\s*(.*)", t)
    m2 = re.search(r"就诊建议[:：]\s*(.*)", t)

    basis = m1.group(1).strip() if m1 else ""
    advice = m2.group(1).strip() if m2 else ""

    # 兜底：如果匹配不到，就取前两行
    if not basis or not advice:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        if not basis and len(lines) >= 1:
            basis = lines[0]
            if not basis.startswith("诊断依据：") and not basis.startswith("诊断依据:"):
                basis = "诊断依据：" + basis
            else:
                basis = re.sub(r"^诊断依据[:：]\s*", "", basis)

        if not advice and len(lines) >= 2:
            advice = lines[1]
            if not advice.startswith("就诊建议：") and not advice.startswith("就诊建议:"):
                advice = "就诊建议：" + advice
            else:
                advice = re.sub(r"^就诊建议[:：]\s*", "", advice)

    # 最终保证非空
    if not basis:
        basis = "需进一步检查/无法判断（模型未按格式输出）"
    if not advice:
        advice = "建议皮肤科就诊，结合皮肤镜检查，必要时活检明确诊断。"

    return basis, advice


def _generate_report_text(prompt: str, image, max_new_tokens: int = 220) -> str:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}
    ]
    prompt_text = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = PROCESSOR(text=[prompt_text], images=[image], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        gen_ids = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    out = PROCESSOR.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return out

@app.post("/infer_report_multipart_txt")
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

    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image 读取失败: {e}")

    # 参数解析（保持你原逻辑不动）
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

    # A) clinical note + 分类
    clinical_note = build_clinical_note(meta)
    pred, probs = _linear_logits_predict(pil, clinical_note, bias, thr)

    # B) 生成两行报告（txt）
    report_prompt = _build_report_prompt_txt(clinical_note, pred, probs)
    advice = _generate_report_text(report_prompt, pil)

    # C) 拼成你想要的 txt 格式
    probs_line = " ".join([f"{k}:{probs.get(k, 0.0):.6f}" for k in ALLOWED_DX])
    txt = (
        f"预测标签：{{{probs_line}}}\n"
        f"就诊建议：{advice}\n"
    )

    return Response(content=txt, media_type="text/plain; charset=utf-8")

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

Here is the original output:
{raw_text}

Now output the repaired JSON only, starting with '{{' and ending with '}}':
""".strip()


def _generate_report_json(report_prompt: str, image, max_new_tokens: int = 700) -> dict:
    # 第一次生成（带图像）
    messages = [
        {"role": "user", "content": [{"type": "text", "text": report_prompt}, {"type": "image", "image": image}]}
    ]
    prompt_text = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = PROCESSOR(text=[prompt_text], images=[image], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        gen_ids = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = PROCESSOR.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    cand = _extract_json_from_text(text)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass

    # 失败：第二次让模型“只修复 JSON”（不带图像也行，速度更快）
    repair_prompt = _repair_to_json_only(text)
    repair_messages = [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}]
    repair_text = PROCESSOR.apply_chat_template(repair_messages, add_generation_prompt=True, tokenize=False)
    repair_inputs = PROCESSOR(text=[repair_text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        repair_ids = MODEL.generate(
            **repair_inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0,   # 修复阶段更“死板”更稳
        )

    repair_out = PROCESSOR.tokenizer.decode(repair_ids[0], skip_special_tokens=True)
    cand2 = _extract_json_from_text(repair_out)
    if cand2:
        try:
            return json.loads(cand2)
        except Exception:
            pass

    return {"raw_text": text}

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


def _validate_required_fields(meta: Dict[str, Any]):
    missing = [k for k in REQUIRED_FIELDS if k not in meta]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"meta json 缺少字段: {missing} (必须严格包含全部字段)"
        )


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

    # 解析参数
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

    # A) 生成 clinical note + 分类 probs
    clinical_note = build_clinical_note(meta)
    pred, probs = _linear_logits_predict(pil, clinical_note, bias, thr)

    # B) 生成报告
    report_prompt = _build_report_prompt_txt(clinical_note, pred, probs)
    txt = _generate_report_text(report_prompt, pil, max_new_tokens=220)  # 直接生成纯文本

    probs_line = " ".join([f"{k}:{float(probs.get(k, 0.0)):.6f}" for k in ALLOWED_DX])
    out = f"预测标签：{{{probs_line}}}\n{txt.strip()}\n"

    return Response(content=out, media_type="text/plain; charset=utf-8")

