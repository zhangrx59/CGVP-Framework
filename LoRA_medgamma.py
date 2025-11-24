# LoRA_medgamma_finetune.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
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

# 基础模型
BASE_MODEL = "google/medgemma-4b-it"

# 原始 metadata CSV（包含所有样本）
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 图片所在根目录 + 后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"   # 如果是 .jpg 就改成 ".jpg"

# 输出的划分后 CSV
TRAIN_CSV = METADATA_CSV.replace(".csv", "_train_5cls.csv")
VAL_CSV   = METADATA_CSV.replace(".csv", "_val_5cls.csv")
TEST_CSV  = METADATA_CSV.replace(".csv", "_test_5cls.csv")   # 用于后续 baseline / LoRA 评估

# 列名（与你的数据一致）
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

# 皮肤病分类标签列（dx / 诊断标签 等）
COL_TARGET      = "dx"

# 只保留这 5 类
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]

# LoRA adapter 输出目录（调好的大模型就保存在这里）
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\medgemma_lora_derm_from_metadata"


# ===================== 1. 工具函数：病历摘要拼接（英文） =====================

def yn_str(v, yes="yes", no="no", unk="unknown"):
    """
    统一把各种 True/False/UNK/NaN 等映射到英文 yes/no/unknown.
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
    根据多列字段自动拼接成一段英文病历摘要文本。
    尽量符合英文医学记录的风格。
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

    # 病史 / 生活方式 / 环境
    skin_ca = yn_str(row.get(COL_SKIN_CANCER))
    other_ca = yn_str(row.get(COL_OTHER_CA))
    smoke = yn_str(row.get(COL_SMOKE), yes="smoker", no="non-smoker", unk="unknown smoking status")
    drink = yn_str(row.get(COL_DRINK), yes="drinker", no="non-drinker", unk="unknown drinking status")
    pesticide = yn_str(
        row.get(COL_PESTICIDE),
        yes="pesticide exposure",
        no="no pesticide exposure",
        unk="unknown pesticide exposure"
    )

    tap = yn_str(row.get(COL_TAP_WATER), yes="has tap water", no="no tap water", unk="unknown tap water supply")
    sewer = yn_str(row.get(COL_SEWER), yes="has sewerage", no="no sewerage", unk="unknown sewerage")

    phototype = row.get(COL_PHOTOTYPE, "")
    d1 = row.get(COL_D1, "")
    d2 = row.get(COL_D2, "")

    # 症状类，统一成 present/absent/unknown（elevation 单独写成 raised/flat/unknown）
    pruritus = yn_str(row.get(COL_PRURITUS), yes="present", no="absent", unk="unknown")
    growth = yn_str(row.get(COL_GROWTH), yes="present", no="absent", unk="unknown")
    pain = yn_str(row.get(COL_PAIN), yes="present", no="absent", unk="unknown")
    morph_change = yn_str(row.get(COL_MORPH_CHANGE), yes="present", no="absent", unk="unknown")
    bleeding = yn_str(row.get(COL_BLEEDING), yes="present", no="absent", unk="unknown")
    elevated = yn_str(row.get(COL_ELEVATED), yes="raised", no="flat", unk="unknown")

    # 部位
    region_en = region if region else "unknown location"

    # 皮损大小
    size_str = ""
    if d1 and d2:
        size_str = f"Lesion size approximately {d1}×{d2} mm."
    elif d1:
        size_str = f"Lesion largest diameter approximately {d1} mm."

    # 光型
    photo_str = f"Skin phototype: {phototype}." if phototype != "" else ""

    # 出生地
    origin_str = ""
    if father_ori or mother_ori:
        origin_str = (
            f"Father's birthplace: {father_ori or 'unknown'}, "
            f"mother's birthplace: {mother_ori or 'unknown'}."
        )

    parts = []

    # 基本信息
    parts.append(f"{age}-year-old {sex_en} with a skin lesion located on {region_en}.")
    if size_str:
        parts.append(size_str)
    if origin_str:
        parts.append(origin_str)

    # 病史
    parts.append(f"History of skin cancer: {skin_ca}; other cancer history: {other_ca}.")

    # 生活方式 + 环境
    parts.append(f"Lifestyle: {smoke}, {drink}, {pesticide}.")
    parts.append(f"Living condition: {tap}, {sewer}.")
    if photo_str:
        parts.append(photo_str)

    # 症状体征
    parts.append(
        f"Symptoms: itching {pruritus}, growth {growth}, pain {pain}, "
        f"shape change {morph_change}, bleeding {bleeding}, elevation {elevated}."
    )

    # 拼成一段英文
    note = " ".join(parts)
    return note


def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


# ===================== 2. 按类别均匀划分 train/val/test =====================

def prepare_splits(
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[str, str, str]:
    """
    从 METADATA_CSV 中：
    - 仅保留 dx ∈ ALLOWED_DX 的样本
    - 按类别 stratify 划分 train/val/test（目前按图片级别分层）
    - 保存到 *_train_5cls.csv / *_val_5cls.csv / *_test_5cls.csv
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

    # 归一化 dx
    df["dx"] = df[COL_TARGET].apply(normalize_dx)
    df = df[df["dx"].isin(ALLOWED_DX)].copy()

    print(f"✅ 过滤后只保留 {ALLOWED_DX}，剩余样本数: {len(df)}")

    # 固定随机种子，保证可复现
    set_seed(seed)

    # 按 dx 分层划分 train / (val+test)
    df_train, df_tmp = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["dx"],
        random_state=seed,
    )

    # 再把 tmp 分成 val/test
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

    # 保存
    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print("💾 已保存划分文件：")
    print(f"  train → {TRAIN_CSV}")
    print(f"  val   → {VAL_CSV}")
    print(f"  test  → {TEST_CSV}")

    return TRAIN_CSV, VAL_CSV, TEST_CSV


# ===================== 3. Dataset：病例 + 图像 → 分类标签 =====================

class DermMetadataDataset(Dataset):
    """
    基于划分后的 CSV 的 Dataset：
    - image: 由 图片ID + IMAGE_ROOT_DIR + IMAGE_EXT 拼路径
    - clinical_note: 由多列字段自动拼接成英文摘要
    - target_text: 皮肤病分类标签（akiec/bcc/bkl/nev/mel），作为生成目标
    """
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        image_id = str(row[COL_IMAGE_ID])
        image_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        image = Image.open(image_path).convert("RGB")

        clinical_note = build_clinical_note(row)
        # 标签统一成小写字符串
        target_text = normalize_dx(str(row[COL_TARGET]))

        # 多模态对话 Prompt（英文，结构尽量稳定）
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
                            "1. For each case, you MUST output exactly ONE label code from this set: {bcc, akiec, bkl, nev, mel}.\n"
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


# ===================== 4. collator：AutoProcessor 打包多模态 =====================

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
            # 1) 生成对话文本（不包含模型答案），不自动加 generation prompt
            chat_text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompt_texts.append(chat_text)

            # 2) 训练时：在文本末尾拼接一个空格和标签，例如 "... Final answer: mel"
            full_text = chat_text + " " + tgt
            texts.append(full_text)

        # 3) 用 processor 正常构造模型输入（文本 + 图像）
        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # 4) 只让模型在“答案那几 token”上计算 loss：
        #    即 mask 掉 BOS + prompt 部分，只保留标签 token 的损失
        tokenizer = self.processor.tokenizer

        for i, prompt_text in enumerate(prompt_texts):
            # 使用与上面相同的 tokenizer 设置，拿到 prompt 自身的 token 长度
            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            prompt_len = len(prompt_tokens)

            # 对应到 input_ids 的位置：
            # input_ids = [BOS] + tokens(chat_text + " " + tgt) + [EOS]
            # 其中 prompt_tokens 是 tokens(chat_text)，因此答案起始大致在：
            ans_start = 1 + prompt_len  # 跳过 BOS，再跳过 prompt token

            # 将答案之前的所有 token mask 掉（不参与 loss）
            labels[i, :ans_start] = -100

        # 5) 再把 padding 的位置也设成 -100（保持原逻辑）
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


# ===================== 5. 加载模型 + LoRA（更“温和”的设置） =====================

def load_model_and_processor():
    print("🔧 加载 MedGEMMA 基础模型（bf16 + LoRA，全精权重，不用 bitsandbytes）...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,   # 如 GPU 不支持 bfloat16，则改为 torch.float16 并在 TrainingArguments 里 fp16=True
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    # 开启梯度检查点（节省显存）
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA 配置：比之前更“温和”，只打在注意力/MLP 线性层上
    lora_config = LoraConfig(
        r=8,                     # 从 16 降到 8，减弱扰动强度
        lora_alpha=16,
        lora_dropout=0.1,        # dropout 从 0.05 提高到 0.1，缓解过拟合和塌缩
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
        ],                       # 不再用 "all-linear"，避免把所有线性层都改坏
    )

    # 让输入需要 grad，方便 LoRA 训练
    model.enable_input_require_grads()

    # 挂载 LoRA 适配器
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"📊 总参数: {total/1e6:.1f}M, 可训练(LoRA): {trainable/1e6:.1f}M")

    return model, processor


# ===================== 6. 主训练入口 =====================

def main():
    # 固定随机种子，方便复现
    set_seed(42)

    # 1) 先按类别分层划分 train/val/test
    train_csv, val_csv, test_csv = prepare_splits()

    # 2) 用 HF Dataset 只加载 train/val（测试集只留给评估）
    raw = load_dataset(
        "csv",
        data_files={"train": train_csv, "val": val_csv},
    )
    train_hf = raw["train"]
    val_hf = raw["val"]

    train_ds = DermMetadataDataset(train_hf)
    val_ds = DermMetadataDataset(val_hf)

    model, processor = load_model_and_processor()
    collator = MedGemmaCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # 从 10 降到 3，防止严重过拟合+遗忘
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,              # 从 1e-4 降到 5e-5，更温和
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,                       # 如不支持则：bf16=False, fp16=True
        fp16=False,
        report_to="none",
        remove_unused_columns=False,     # 保留 image/messages 等自定义字段
        seed=42,
    )

    trainer = Trainer(
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
    print(f"✅ LoRA adapter 已保存到: {OUTPUT_DIR}")
    print(f"✅ 评估用的 test CSV 在: {TEST_CSV}")


if __name__ == "__main__":
    main()
