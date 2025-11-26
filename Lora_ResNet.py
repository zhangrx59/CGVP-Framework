# LoRA_medgamma_focal_with_resnet_prior_softmax5.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
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

from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F

# ===================== 0. 配置区域 =====================

BASE_MODEL = "google/medgemma-4b-it"

METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata.csv"
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"

TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 🔴 ResNet 视觉模块 pth 路径（你训练好的那个）
RESNET_CKPT = r"C:\Users\zhangrx59\PycharmProjects\LoRA\best_resnet50_custom_cbam_focal.pth"

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
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]  # 固定顺序，后面要用 index

# LoRA 微调输出目录
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\lora_focal_with_resnet_prior_softmax5"


# ===================== 0.1 ResNet+CBAM 定义（用于推理） =====================

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
    """
    与你训练 ResNet 时保持一致：ResNet50 + layer4 -> CBAMBlock -> fc(num_classes)
    """
    model_ft = resnet50(weights=None)
    model_ft.layer4 = nn.Sequential(
        model_ft.layer4,
        CBAMBlock(in_planes=2048, reduction=16, kernel_size=7)
    )
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def load_resnet_prior_model(device: torch.device):
    """
    加载你训练好的 ResNet 视觉模块，并返回：
      - model_resnet (eval 模式)
      - transform_resnet
      - dx_categories（按训练时顺序）
    """
    assert os.path.exists(RESNET_CKPT), f"ResNet checkpoint not found: {RESNET_CKPT}"
    ckpt = torch.load(RESNET_CKPT, map_location="cpu")

    if "dx_categories" not in ckpt:
        raise RuntimeError(
            "ResNet checkpoint 中没有 'dx_categories' 字段，请确认训练脚本保存了它。"
        )
    dx_categories = ckpt["dx_categories"]
    num_classes = len(dx_categories)

    model_resnet = build_resnet_with_cbam(num_classes)
    model_resnet.load_state_dict(ckpt["model_state_dict"])
    model_resnet.to(device)
    model_resnet.eval()

    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    transform_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    return model_resnet, transform_resnet, dx_categories


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
    返回:
      - messages: system+user 提示词（含 ResNet 先验）
      - image: PIL 图像
      - target_text: label code 文本 ('akiec' / 'bcc' / 'bkl' / 'nev' / 'mel')
      - cls_label_idx: 0~4（在 ALLOWED_DX 里的索引，用于 5 类 softmax）
    """
    def __init__(
        self,
        hf_or_df,
        resnet_model: nn.Module,
        resnet_transform: transforms.Compose,
        resnet_dx_categories: List[str],
        resnet_device: torch.device,
    ):
        if isinstance(hf_or_df, pd.DataFrame):
            self.df = hf_or_df.reset_index(drop=True)
        else:
            self.df = hf_or_df.to_pandas().reset_index(drop=True)

        self.resnet_model = resnet_model
        self.resnet_transform = resnet_transform
        self.resnet_dx_categories = resnet_dx_categories
        self.resnet_device = resnet_device

        # 缓存每个 image_id 的先验结果，确保“一张图只过一次 ResNet”
        self._prior_cache: Dict[str, str] = {}

    def __len__(self):
        return len(self.df)

    def _get_resnet_prior_text(self, image_id: str, pil_image: Image.Image) -> str:
        """对单张图像做一次 ResNet 推理，输出英文先验描述，并做缓存。"""
        if image_id in self._prior_cache:
            return self._prior_cache[image_id]

        if self.resnet_model is None:
            prior = "ResNet prior not available."
            self._prior_cache[image_id] = prior
            return prior

        x = self.resnet_transform(pil_image).unsqueeze(0).to(self.resnet_device)

        with torch.no_grad():
            logits = self.resnet_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_label = self.resnet_dx_categories[top_idx]
        top_prob = float(probs[top_idx])

        parts = []
        for cls, p in zip(self.resnet_dx_categories, probs):
            parts.append(f"{cls}={p:.3f}")
        dist_str = ", ".join(parts)

        prior_text = (
            f"ResNet visual classifier top prediction: {top_label} "
            f"(probability {top_prob:.3f}). "
            f"Full probability distribution over labels: {dist_str}."
        )

        self._prior_cache[image_id] = prior_text
        return prior_text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_id = row[COL_IMAGE_ID]
        image_id = str(raw_id).strip("[]'\" ").replace(",", "").strip()
        image_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        image = Image.open(image_path).convert("RGB")

        # 原有病历文本
        clinical_note = build_clinical_note(row)
        label = normalize_dx(str(row[COL_TARGET]))
        if label not in ALLOWED_DX:
            # 理论上 prepare_splits 已过滤，这里防御一下
            label = "nev"

        target_text = label  # 直接用 dx code 作为监督信号
        cls_label_idx = ALLOWED_DX.index(label)  # 0~4

        # 🔴 ResNet 视觉先验
        resnet_prior = self._get_resnet_prior_text(image_id, image)

        # ====== 保留你原有的提示词逻辑，只在中间插入 ResNet 先验 ======
        user_text = (
            f"Clinical note:\n{clinical_note}\n\n"
            "You are given the above clinical note together with a dermoscopic image of the lesion.\n"
            "Based on both the clinical information and the image, decide which label code best describes the lesion.\n\n"
            "Additionally, an independent ResNet-based dermoscopic image classifier "
            "provides the following visual prior based only on the image:\n"
            f"{resnet_prior}\n\n"
            "You should use this visual prior only as auxiliary information and still make your own final decision.\n\n"
            "Valid label codes are: akiec, bcc, bkl, nev, mel (as defined in the system instructions).\n\n"
            "You MUST respond with ONLY ONE label code from {akiec, bcc, bkl, nev, mel}.\n"
            "Do NOT output a list of labels.\n"
            "Do NOT repeat the full disease names.\n"
            "Do NOT add any explanation or extra words.\n\n"
            "Final answer (ONLY ONE code):"
        )

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
                        "text": user_text,
                    },
                    {"type": "image", "image": image},
                ],
            },
        ]

        return {
            "messages": messages,
            "image": image,
            "target_text": target_text,
            "cls_label_idx": cls_label_idx,
        }


# ===================== 4. collator：构造输入 & 只在答案 token 上算损失 =====================

@dataclass
class MedGemmaCollator:
    processor: AutoProcessor

    def __call__(self, batch) -> Dict[str, Any]:
        images = [eg["image"] for eg in batch]
        messages_list = [eg["messages"] for eg in batch]
        targets = [eg["target_text"] for eg in batch]
        cls_labels = [eg["cls_label_idx"] for eg in batch]

        texts = []
        prompt_texts = []
        for msgs, tgt in zip(messages_list, targets):
            # chat_text 不包含监督标签
            chat_text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompt_texts.append(chat_text)
            # 把监督信号（label code）拼在最后，作为 gold completion
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
            # 注意：transformers 通常会在最前面加 BOS（例如 <bos>），
            # 所以答案开始位置 = 1 + prompt_len
            ans_start = 1 + prompt_len
            labels[i, :ans_start] = -100

        # pad 部分也不算 loss
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        # 新增：5 类分类标签（0~4），供 Trainer 使用
        model_inputs["cls_labels"] = torch.tensor(cls_labels, dtype=torch.long)

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
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
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


# ===================== 6. Softmax5Trainer：强制 5 类 softmax 头 =====================

class Softmax5Trainer(Trainer):
    """
    自定义 Trainer：
    - 不用模型内置的 token-level loss
    - 从 logits + labels 中找到“第一个答案 token 的位置”
    - 在该位置，只取 5 个标签对应的 token logit，做 5 类 softmax 交叉熵
    """
    def __init__(self, label_token_ids: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 长度为 5 的 token id 列表，对应 ALLOWED_DX 顺序
        self.label_token_ids = torch.LongTensor(label_token_ids)
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 兼容 Trainer 额外参数
        labels = inputs.pop("labels")        # (B, T)
        cls_labels = inputs.pop("cls_labels")  # (B,)
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        device = logits.device
        label_token_ids = self.label_token_ids.to(device)

        # 找每个样本中，第一个 label != -100 的位置（即答案首 token 位置）
        # mask: True 表示是答案 token
        mask = labels.ne(-100)  # (B, T)
        # 如果某行全是 False，会 argmax 到 0，这种情况理论上不会发生（至少有一个答案 token）
        first_pos = mask.float().argmax(dim=1)  # (B,)

        batch_idx = torch.arange(logits.size(0), device=device)
        # 取出每个样本在答案首 token 位置的 logits: (B, V)
        logits_first = logits[batch_idx, first_pos, :]  # (B, V)

        # 只保留 5 个 label token 的 logits，得到 (B, 5)
        logits_5 = logits_first[:, label_token_ids]

        loss = self.ce_loss(logits_5, cls_labels.to(device))

        if return_outputs:
            return loss, outputs
        return loss


# ===================== 7. 主训练入口 =====================

def main():
    set_seed(42)

    train_csv, val_csv, test_csv = prepare_splits()

    # 不做重采样：直接用原 train/val 划分
    raw = load_dataset("csv", data_files={"train": train_csv, "val": val_csv})
    train_hf = raw["train"]
    val_hf = raw["val"]

    # 先加载 MedGEMMA + LoRA
    model, processor = load_model_and_processor()
    collator = MedGemmaCollator(processor=processor)

    # 加载 ResNet 视觉模块（只用于先验推理）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model, resnet_transform, resnet_dx_categories = load_resnet_prior_model(device)

    # 构造 Dataset（带 ResNet 先验）
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

    # ===================== 关键：构造 5 个 label 的 token id =====================
    # 我们只看 label code 的“首 token”，它们的 token id 组成 5 类 softmax 头
    tokenizer = processor.tokenizer
    label_token_ids: List[int] = []
    for label in ALLOWED_DX:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Label {label!r} 被 tokenizer 分成空 token 序列，异常")
        label_token_ids.append(ids[0])
    print("🔧 5 类标签的首 token id:", label_token_ids)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
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
        max_grad_norm=0.5,
    )

    trainer = Softmax5Trainer(
        label_token_ids=label_token_ids,
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
    print(f"✅ LoRA+ResNetPrior+Softmax5 模型已保存到: {OUTPUT_DIR}")
    print(f"✅ 评估用的 test CSV 在: {TEST_CSV}")


if __name__ == "__main__":
    main()
