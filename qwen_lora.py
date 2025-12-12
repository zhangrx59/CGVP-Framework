# lora_qwen25vl_derm.py
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset as TorchDataset

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model

# ===================== 0. 基本配置 =====================

# Qwen2.5-VL 基座模型
BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

# 你的总 metadata CSV（未划分）
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata.csv"

# 划分后的 CSV
TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train_balanced.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 图像根目录 / 后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"  # 如果是 .jpg 就改成 ".jpg"

# ResNet 先验（可选）
USE_RESNET_PRIOR = True
RESNET_CKPT = r"C:\Users\zhangrx59\PycharmProjects\LoRA\best_resnet50_custom_cbam_focal.pth"

# LoRA 输出目录
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen25vl_derm_lora"

# 标签相关
COL_IMAGE_ID = "image_id"
COL_TARGET   = "dx"

# 其他临床信息列名（按你之前的定义）
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

# 训练/评估的 4 类
ALLOWED_DX = ["akiec", "bcc", "nev", "mel"]


# ===================== 0.1 如果你要用 ResNet 先验：CBAM 模块 =====================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat((avg_out, max_out), dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


def build_resnet_with_cbam(num_classes: int) -> nn.Module:
    from torchvision.models import resnet50
    model_ft = resnet50(weights=None)
    model_ft.layer4 = nn.Sequential(
        model_ft.layer4,
        CBAMBlock(in_planes=2048, reduction=16, kernel_size=7),
    )
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def load_resnet_prior_model(device: torch.device):
    """
    需要你在训练 ResNet 时，把 dx_categories 也存进 ckpt：
    torch.save({"model_state_dict": model.state_dict(), "dx_categories": class_names}, path)
    """
    import torchvision.transforms as T
    assert os.path.exists(RESNET_CKPT), f"ResNet ckpt not found: {RESNET_CKPT}"
    ckpt = torch.load(RESNET_CKPT, map_location="cpu")
    dx_categories = ckpt["dx_categories"]
    num_classes = len(dx_categories)

    model_res = build_resnet_with_cbam(num_classes)
    model_res.load_state_dict(ckpt["model_state_dict"])
    model_res.to(device)
    model_res.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model_res, transform, dx_categories


# ===================== 1. 一些工具函数 =====================

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


def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def build_clinical_note(row: pd.Series) -> str:
    """
    跟你之前 evaluate_medgamma / Lora_ResNet 中的逻辑一致
    """
    age = row.get(COL_AGE, "")
    sex_raw = str(row.get(COL_SEX, "") or "").strip().lower()
    region = str(row.get(COL_REGION, "") or "").strip()
    father_ori = str(row.get(COL_FATHER_ORI, "") or "").strip()
    mother_ori = str(row.get(COL_MOTHER_ORI, "") or "").strip()

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

    phototype_str = f"Fitzpatrick skin phototype: {phototype}." if phototype else ""

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
        f"Past history of skin cancer: {skin_ca}; other malignancies: {other_ca}."
    )
    parts.append(
        f"Lifestyle: smoking {smoke}, alcohol {drink}, pesticide exposure {pesticide}."
    )
    parts.append(f"Living environment: tap water {tap}, sewer system {sewer}.")
    if phototype_str:
        parts.append(phototype_str)

    parts.append(
        "Current symptoms and signs: "
        f"pruritus {pruritus}, growth {growth}, pain {pain}, "
        f"morphologic change {morph_change}, bleeding {bleeding}, "
        f"elevation {elevated}."
    )

    return " ".join(parts)


# ===================== 2. 划分 train/val/test（不重采样） =====================

def prepare_splits(
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[str, str, str]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV):
        print("📁 已检测到现有划分 CSV，直接使用：")
        print("  train:", TRAIN_CSV)
        print("  val  :", VAL_CSV)
        print("  test :", TEST_CSV)
        return TRAIN_CSV, VAL_CSV, TEST_CSV

    print(f"📄 读取原始 CSV: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV, encoding="utf-8")

    if COL_TARGET not in df.columns:
        raise ValueError(f"找不到标签列 {COL_TARGET!r}")

    df["dx"] = df[COL_TARGET].apply(normalize_dx)
    df = df[df["dx"].isin(ALLOWED_DX)].copy()
    print(f"✅ 过滤后只保留 {ALLOWED_DX}，剩余样本: {len(df)}")

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

    print("📊 train 分布:", df_train["dx"].value_counts().to_dict())
    print("📊 val   分布:", df_val["dx"].value_counts().to_dict())
    print("📊 test  分布:", df_test["dx"].value_counts().to_dict())

    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print("💾 已保存划分 CSV：")
    print("  train →", TRAIN_CSV)
    print("  val   →", VAL_CSV)
    print("  test  →", TEST_CSV)

    return TRAIN_CSV, VAL_CSV, TEST_CSV


# ===================== 3. Dataset：图 + 文 +（可选）ResNet 先验 =====================

class DermMetadataDataset(TorchDataset):
    def __init__(
        self,
        hf_or_df,
        resnet_model: nn.Module = None,
        resnet_transform=None,
        resnet_dx_categories: List[str] = None,
        resnet_device: torch.device = None,
    ):
        if isinstance(hf_or_df, pd.DataFrame):
            self.df = hf_or_df.reset_index(drop=True)
        else:
            self.df = hf_or_df.to_pandas().reset_index(drop=True)

        self.resnet_model = resnet_model
        self.resnet_transform = resnet_transform
        self.resnet_dx_categories = resnet_dx_categories
        self.resnet_device = resnet_device

        self._prior_cache: Dict[str, str] = {}

    def __len__(self):
        return len(self.df)

    def _get_resnet_prior(self, image_id: str, pil_img: Image.Image) -> str:
        if not USE_RESNET_PRIOR or self.resnet_model is None:
            return ""

        if image_id in self._prior_cache:
            return self._prior_cache[image_id]

        import torch.nn.functional as F_local

        x = self.resnet_transform(pil_img).unsqueeze(0).to(self.resnet_device)
        with torch.no_grad():
            logits = self.resnet_model(x)
            probs = F_local.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_label = self.resnet_dx_categories[top_idx]
        top_prob = float(probs[top_idx])

        parts = [f"{cls}={probs[i]:.3f}" for i, cls in enumerate(self.resnet_dx_categories)]
        dist_str = ", ".join(parts)

        prior = (
            "An independent ResNet-based classifier provides a weak visual prior: "
            f"top prediction is {top_label} (approx. prob {top_prob:.3f}). "
            f"Full probability distribution: {dist_str}. "
            "Use this information as a weak auxiliary hint only."
        )

        self._prior_cache[image_id] = prior
        return prior

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_id = row[COL_IMAGE_ID]
        image_id = str(raw_id).strip("[]'\" ").replace(",", "").strip()
        img_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        image = Image.open(img_path).convert("RGB")

        clinical_note = build_clinical_note(row)
        dx = normalize_dx(str(row[COL_TARGET]))
        if dx not in ALLOWED_DX:
            dx = "nev"
        cls_idx = ALLOWED_DX.index(dx)

        prior_text = self._get_resnet_prior(image_id, image)
        if prior_text:
            clinical_prompt = (
                f"Clinical note:\n{clinical_note}\n\n"
                "Auxiliary visual prior from an independent ResNet classifier:\n"
                f"{prior_text}\n\n"
            )
        else:
            clinical_prompt = f"Clinical note:\n{clinical_note}\n\n"

        return {
            "image": image,
            "clinical_text": clinical_prompt,
            "dx_code": dx,
            "cls_label": cls_idx,
        }


# ===================== 4. Collator：Qwen chat + image + labels =====================

@dataclass
class QwenCollator:
    processor: AutoProcessor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [x["image"] for x in batch]
        clinical_texts = [x["clinical_text"] for x in batch]
        dx_codes = [x["dx_code"] for x in batch]
        cls_labels = torch.tensor([x["cls_label"] for x in batch], dtype=torch.long)

        texts_full: List[str] = []
        texts_prompt_only: List[str] = []

        tokenizer = self.processor.tokenizer

        for clinical, dx in zip(clinical_texts, dx_codes):
            user_text = (
                f"{clinical}"
                "Based on the clinical information and the dermoscopic image, "
                "predict the most likely diagnosis and answer with only one label code "
                "from {akiec, bcc, nev, mel}."
                "Final answer:"
            )

            # 关键：Qwen2.5-VL 需要“结构化多模态消息”来注入真正的 image placeholder token
            messages_full = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert dermatologist. "
                        "Your task is to classify dermoscopic images based on clinical notes. "
                        "You MUST answer with exactly one label code from {akiec, bcc, nev, mel}. "
                        "Do not output anything else."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": dx},
            ]

            chat_full = self.processor.apply_chat_template(
                messages_full,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts_full.append(chat_full)

            # prompt-only（不含答案），用于构造 label mask
            messages_prompt = messages_full[:-1]
            chat_prompt = self.processor.apply_chat_template(
                messages_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            texts_prompt_only.append(chat_prompt)

        # 编码（图像 + 文本）；此时 text 内已包含“图像占位 token”，与 images 一一对应
        model_inputs = self.processor(
            text=texts_full,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # 只在答案 token 上计算 loss：prompt 部分设为 -100
        # 用 prompt 的 token 长度切分（比“取最后一位”更稳，避免特殊 token 干扰）
        for i, prompt_text in enumerate(texts_prompt_only):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            ans_start = min(len(prompt_ids), labels.size(1) - 1)
            labels[i, :ans_start] = -100

        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        model_inputs["cls_labels"] = cls_labels
        return model_inputs



# ===================== 5. 加载 Qwen2.5-VL + LoRA =====================

def load_qwen_model_and_processor():
    if torch.cuda.is_available():
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        dtype = torch.float32

    print(f"🔧 Loading base model: {BASE_MODEL}, dtype={dtype}")
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    # LoRA 配置（可以和你 medgemma 的类似）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params: {total_params/1e6:.1f}M, Trainable (LoRA): {trainable_params/1e6:.1f}M")
    print(f"🔧 Model device: {next(model.parameters()).device}")

    return model, processor


# ===================== 6. 自定义 Trainer：Softmax4 + class_weights =====================

class Softmax4Trainer(Trainer):
    def __init__(self, label_token_ids: List[int], class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_token_ids = torch.LongTensor(label_token_ids)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # kwargs 用来兼容 transformers 新版本传入的 num_items_in_batch 等参数
        labels = inputs.pop("labels")          # (B, T)
        cls_labels = inputs.pop("cls_labels")  # (B,)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        device = logits.device
        label_token_ids = self.label_token_ids.to(device)

        # 找第一个 labels != -100 的位置作为答案 token
        mask = labels.ne(-100)
        first_pos = mask.float().argmax(dim=1)  # (B,)

        batch_idx = torch.arange(logits.size(0), device=device)
        logits_first = logits[batch_idx, first_pos, :]  # (B, V)

        logits4 = logits_first[:, label_token_ids]  # (B, 4)

        weight = self.class_weights.to(device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits4, cls_labels.to(device), weight=weight)

        if return_outputs:
            return loss, outputs
        return loss

# ===================== 7. main：组装一切并训练 =====================

def main():
    set_seed(42)

    train_csv, val_csv, test_csv = prepare_splits()

    raw = load_dataset("csv", data_files={"train": train_csv, "val": val_csv})
    train_hf = raw["train"]
    val_hf = raw["val"]

    # 统计类别频次 → class_weights（缓解不平衡）
    df_train = train_hf.to_pandas()
    df_train["dx"] = df_train[COL_TARGET].apply(normalize_dx)
    counts = df_train["dx"].value_counts().to_dict()
    class_weights_list = []
    for cls in ALLOWED_DX:
        c = counts.get(cls, 1)
        class_weights_list.append(1.0 / math.sqrt(c))
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32)
    class_weights = class_weights * (len(class_weights_list) / class_weights.sum())
    print("🔧 class weights:", dict(zip(ALLOWED_DX, class_weights.tolist())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 Qwen2.5-VL + LoRA
    model, processor = load_qwen_model_and_processor()
    collator = QwenCollator(processor=processor)

    # （可选）ResNet 先验
    if USE_RESNET_PRIOR:
        resnet_model, resnet_transform, resnet_dx_categories = load_resnet_prior_model(device)
    else:
        resnet_model = resnet_transform = resnet_dx_categories = None

    train_ds = DermMetadataDataset(
        train_hf,
        resnet_model=resnet_model,
        resnet_transform=resnet_transform,
        resnet_dx_categories=resnet_dx_categories,
        resnet_device=device,
    )
    val_ds = DermMetadataDataset(
        val_hf,
        resnet_model=resnet_model,
        resnet_transform=resnet_transform,
        resnet_dx_categories=resnet_dx_categories,
        resnet_device=device,
    )

    # 构造 4 个 label 的 token id（建议加前导空格，保证和答案位置对齐）
    tokenizer = processor.tokenizer
    label_token_ids: List[int] = []
    for lab in ALLOWED_DX:
        ids = tokenizer(" " + lab, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Label {lab!r} has empty tokenization!")
        label_token_ids.append(ids[0])
    print("🔧 label_token_ids:", dict(zip(ALLOWED_DX, label_token_ids)))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        max_grad_norm=0.5,
    )

    trainer = Softmax4Trainer(
        label_token_ids=label_token_ids,
        class_weights=class_weights,
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
    print("✅ Qwen2.5-VL-7B dermoscopy LoRA 已保存到:", OUTPUT_DIR)
    print("✅ 测试集 CSV:", test_csv)


if __name__ == "__main__":
    main()
