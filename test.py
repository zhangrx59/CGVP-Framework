# ===================== Qwen2.5-VL-7B QLoRA (16GB 稳跑版) =====================

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ===================== 路径与配置 =====================

BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train_balanced.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
IMAGE_ROOT = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"

OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\qwen25vl_qlora"

LABELS = ["akiec", "bcc", "nev", "mel"]

# ===================== Dataset =====================

class DermDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(
            os.path.join(IMAGE_ROOT, row["image_id"] + ".png")
        ).convert("RGB")
        dx = row["dx"].lower()
        return {
            "image": image,
            "dx": dx,
            "cls_label": LABELS.index(dx),
        }

# ===================== Collator（Qwen 多模态规范） =====================

@dataclass
class QwenCollator:
    processor: AutoProcessor

    def __call__(self, batch):
        images = [x["image"] for x in batch]
        cls_labels = torch.tensor([x["cls_label"] for x in batch], dtype=torch.long)

        texts = []
        for x in batch:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert dermatologist. "
                        "You MUST answer with exactly one label code "
                        "from {akiec, bcc, nev, mel}."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Based on the dermoscopic image, "
                                "predict the diagnosis. "
                                "Final answer:"
                            ),
                        },
                    ],
                },
                {"role": "assistant", "content": x["dx"]},
            ]
            txt = self.processor.apply_chat_template(
                messages, tokenize=False
            )
            texts.append(txt)

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # 不用语言模型 loss，自己算 softmax4
        inputs["labels"] = torch.full_like(inputs["input_ids"], -100)
        inputs["cls_labels"] = cls_labels
        return inputs

# ===================== Softmax4 Trainer =====================

class Softmax4Trainer(Trainer):
    def __init__(self, label_token_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_token_ids = torch.tensor(label_token_ids)

    def compute_loss(self, model, inputs, **kwargs):
        cls_labels = inputs.pop("cls_labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        last_logits = logits[:, -1, :]
        logits4 = last_logits[:, self.label_token_ids.to(logits.device)]

        loss = F.cross_entropy(
            logits4,
            cls_labels.to(logits.device),
        )
        return loss

# ===================== main =====================

def main():
    set_seed(42)

    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)

    train_ds = DermDataset(df_train)
    val_ds   = DermDataset(df_val)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.image_processor.size = {"shortest_edge": 336}

    # ===================== QLoRA 核心配置 =====================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # QLoRA 必须
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 冻结 vision encoder（非常省显存）
    if hasattr(model, "vision_model"):
        for p in model.vision_model.parameters():
            p.requires_grad = False

    # ===================== LoRA =====================
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenizer = processor.tokenizer
    label_token_ids = [
        tokenizer(" " + x, add_special_tokens=False)["input_ids"][0]
        for x in LABELS
    ]

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        learning_rate=2e-4,   # QLoRA 通常 LR 要大一点
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=200,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Softmax4Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=QwenCollator(processor),
        label_token_ids=label_token_ids,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ QLoRA 训练完成")

if __name__ == "__main__":
    main()
