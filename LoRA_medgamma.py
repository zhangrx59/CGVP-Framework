# LoRA_medgamma_focal.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset  # 避免与 HF Dataset 冲突
from PIL import Image
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

# ===================== 0. 配置区域 =====================

BASE_MODEL = "google/medgemma-4b-it"

METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata.csv"
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"

TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"




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

COL_TARGET      = "dx"
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]

# LoRA + Focal Loss 微调后的输出目录
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lora_focal"


# ===================== 1. 病历文本 & 标签处理 =====================

def yn_str(v, yes="yes", no="no", unk="unknown"):
    """
    把各种 True/False/空/NaN 归一化成 yes/no/unk 或你指定的描述
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


def build_clinical_note(row) -> str:
    """
    构造英文病历描述（和 evaluate_medgamma.py 中逻辑保持一致）
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


def normalize_dx(label: str) -> str:
    """
    把 nv 统一成 nev，其余小写
    """
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


# ===================== 2. train/val/test 划分（不重采样） =====================

def prepare_splits(
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[str, str, str]:
    """
    不做重采样，只做分层划分，结果写入 *_train_5cls.csv / *_val_5cls.csv / *_test_5cls.csv
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV):
        print("📁 发现已有划分文件，直接复用：")
        print(f"  train: {TRAIN_CSV}")
        print(f"  val  : {VAL_CSV}")
        print(f"  test : {TEST_CSV}")
        return TRAIN_CSV, VAL_CSV, TEST_CSV

    print(f"📄 读取原始 CSV: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV, encoding="utf-8")

    if COL_TARGET not in df.columns:
        raise ValueError(f"CSV 中找不到标签列 {COL_TARGET!r}")

    df["dx"] = df[COL_TARGET].apply(normalize_dx)
    df = df[df["dx"].isin(ALLOWED_DX)].copy()

    print(f"✅ 过滤后只保留 {ALLOWED_DX}，剩余样本数: {len(df)}")

    set_seed(seed)

    df_train, df_tmp = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["dx"],
        random_state=seed,
    )

    tmp_ratio = test_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_tmp,
        test_size=tmp_ratio,
        stratify=df_tmp["dx"],
        random_state=seed,
    )

    print("📊 按类别分层划分完成：")
    print("  train:", df_train["dx"].value_counts().to_dict())
    print("  val  :", df_val["dx"].value_counts().to_dict())
    print("  test :", df_test["dx"].value_counts().to_dict())

    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print("💾 已保存划分文件：")
    print(f"  train → {TRAIN_CSV}")
    print(f"  val   → {VAL_CSV}")
    print(f"  test  → {TEST_CSV}")

    return TRAIN_CSV, VAL_CSV, TEST_CSV


# ===================== 3. 自定义 Dataset =====================

class DermMetadataDataset(TorchDataset):
    """
    使用 HF Dataset 或 pandas.DataFrame 作为底层存储，
    返回 (messages, image[PIL], target_text) 供 collator 使用。
    target_text 就是 dx code: 'akiec' / 'bcc' / 'bkl' / 'nev' / 'mel'
    """
    def __init__(self, hf_or_df):
        if isinstance(hf_or_df, pd.DataFrame):
            self.df = hf_or_df.reset_index(drop=True)
        else:
            self.df = hf_or_df.to_pandas().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_id = row[COL_IMAGE_ID]
        image_id = str(raw_id).strip("[]'\" ").replace(",", "").strip()
        image_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        image = Image.open(image_path).convert("RGB")

        clinical_note = build_clinical_note(row)
        label = normalize_dx(str(row[COL_TARGET]))
        if label not in ALLOWED_DX:
            # 理论上 prepare_splits 已过滤，这里防御一下
            label = "nev"

        target_text = label  # 直接用 dx code 作为监督信号

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert dermatology assistant specialized in dermoscopic images.\n"
                            "Your task is to classify a skin lesion based on a clinical note and a dermoscopic image.\n\n"
                            "The dataset uses the following label codes:\n"
                            " - akiec = actinic keratoses / intraepithelial carcinoma of the skin (Bowen's disease),\n"
                            " - bcc   = basal cell carcinoma,\n"
                            " - bkl   = benign keratosis-like lesions (including solar lentigines / seborrheic keratoses / lichen-planus like keratoses),\n"
                            " - nev   = melanocytic nevi,\n"
                            " - mel   = melanoma.\n\n"
                            "When you see these codes, you should understand them as the corresponding disease entities above.\n\n"
                            "IMPORTANT OUTPUT RULES:\n"
                            "1. For each case, you MUST output exactly ONE label code from this set: {akiec, bcc, bkl, nev, mel}.\n"
                            "2. Do NOT output the full disease names.\n"
                            "3. Do NOT output a list of all labels.\n"
                            "4. Do NOT add explanations, probabilities, or any other text.\n"
                            "5. The final answer must consist of a single code token only, e.g. 'bcc'.\n"
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
                            "You are given the above clinical note together with a dermoscopic image of the lesion.\n"
                            "Based on both the clinical information and the image, decide which label code best describes the lesion.\n\n"
                            "Valid label codes are: akiec, bcc, bkl, nev, mel (as defined in the system instructions).\n\n"
                            "You MUST respond with ONLY ONE label code from {akiec, bcc, bkl, nev, mel}.\n"
                            "Do NOT output a list of labels.\n"
                            "Do NOT repeat the full disease names.\n"
                            "Do NOT add any explanation or extra words.\n\n"
                            "Final answer (ONLY ONE code):"
                        ),
                    },
                    {"type": "image", "image": image},
                ],
            },
        ]

        return {
            "messages": messages,
            "image": image,
            "target_text": target_text,
        }


# ===================== 4. collator：构造输入 & 只在答案 token 上算 loss =====================

@dataclass
class MedGemmaCollator:
    processor: AutoProcessor

    def __call__(self, batch) -> Dict[str, Any]:
        images = [eg["image"] for eg in batch]
        messages_list = [eg["messages"] for eg in batch]
        targets = [eg["target_text"] for eg in batch]

        texts = []
        prompt_texts = []
        for msgs, tgt in zip(messages_list, targets):
            chat_text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompt_texts.append(chat_text)
            # 把监督信号拼在 prompt 后面，作为 gold completion
            full_text = chat_text + " " + tgt
            texts.append(full_text)

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        tokenizer = self.processor.tokenizer

        # 只让 "答案" 部分参与 loss，prompt 部分 label = -100
        for i, prompt_text in enumerate(prompt_texts):
            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            prompt_len = len(prompt_tokens)
            # 注意：transformers 通常会加 BOS，所以答案开始位置 = 1 + prompt_len
            ans_start = 1 + prompt_len
            labels[i, :ans_start] = -100

        # pad 部分也不算 loss
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


# ===================== 5. 模型 + LoRA =====================

def load_model_and_processor():
    print("🔧 加载 MedGEMMA 基础模型（bf16 + LoRA，单卡加载）...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 不再使用 device_map / 自动分布，避免 meta device 的梯度错误
    if device.type == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if supports_bf16:
            dtype = torch.bfloat16
            print("🔧 GPU 支持 bfloat16，使用 torch.bfloat16")
        else:
            dtype = torch.float16
            print("🔧 GPU 不支持 bfloat16，使用 torch.float16")
    else:
        dtype = torch.float32
        print("🔧 使用 CPU，dtype=torch.float32")

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    # 先关掉 cache，再开 gradient checkpointing
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 让输入参与反传，配合 LoRA
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"📊 总参数: {total/1e6:.1f}M, 可训练(LoRA): {trainable/1e6:.1f}M")
    print(f"🔧 模型当前设备: {next(model.parameters()).device}")

    return model, processor


# ===================== 6. FocalTrainer：在 CE 基础上套 Focal Loss =====================

class FocalTrainer(Trainer):
    """
    自定义 Trainer：
    - 不用模型内置 loss
    - 自己从 logits + labels 计算 token-level CE
    - 再做 sample-level Focal Loss:  ((1 - p_t)^gamma) * loss
    """
    def __init__(self, focal_gamma: float = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.ce_loss_fct = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 这里一定要接受 **kwargs，兼容 Trainer 内部额外参数（比如 num_items_in_batch）
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        vocab_size = logits.size(-1)

        # causal LM: 右移一位
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_flat = self.ce_loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )  # (B*T,)

        loss = loss_flat.view(shift_labels.size())  # (B, T)

        # 忽略 label=-100 的位置
        active_mask = shift_labels.ne(-100)
        loss = loss * active_mask

        # 每个样本的平均 token loss
        token_num = active_mask.sum(dim=-1).clamp(min=1)
        loss_per_sample = loss.sum(dim=-1) / token_num  # (B,)

        if self.focal_gamma is not None and self.focal_gamma > 0:
            pt = torch.exp(-loss_per_sample)  # 视为“预测正确”的概率
            focal_factor = (1 - pt) ** self.focal_gamma
            loss_per_sample = focal_factor * loss_per_sample

        loss_mean = loss_per_sample.mean()

        if return_outputs:
            return loss_mean, outputs
        return loss_mean


# ===================== 7. 主训练入口 =====================

def main():
    set_seed(42)

    train_csv, val_csv, test_csv = prepare_splits()

    # 不做重采样：直接用原 train/val 划分
    raw = load_dataset("csv", data_files={"train": train_csv, "val": val_csv})
    train_hf = raw["train"]
    val_hf = raw["val"]

    train_ds = DermMetadataDataset(train_hf)
    val_ds = DermMetadataDataset(val_hf)

    model, processor = load_model_and_processor()
    collator = MedGemmaCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,           # 如 GPU 不支持，可以改为 bf16=False, fp16=True
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
        max_grad_norm=1.0,
    )

    trainer = FocalTrainer(
        focal_gamma=2.0,       # 你可以改成 1.0 / 1.5 做对比
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA+Focal 模型已保存到: {OUTPUT_DIR}")
    print(f"✅ 评估用的 test CSV 在: {TEST_CSV}")


if __name__ == "__main__":
    main()
