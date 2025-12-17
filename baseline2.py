# evaluate_qwen_baseline_linear_full.py
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL-7B-Instruct Baseline 4分类评估（方案A：token-level / linear logits）
- 不加载 LoRA
- 不 generate（只 forward 一次，取最后 token logits）
- 仅对 4 个类别 token 做 softmax，得到概率
- 输出：classification_report、confusion matrix、ROC/PR（可选用真实概率或伪分数）
- 结构/格式尽量仿照 evaluate_medgamma / evaluate_qwen_lora 的“完整版”
"""

import os
import json
import time
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


# =========================
# 配置区（按你的工程路径修改）
# =========================

BASE_MODEL = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"

# 输出目录
OUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\eval_baseline_linear_full"
os.makedirs(OUT_DIR, exist_ok=True)

# CSV 列名（按你的数据实际列名）
COL_IMAGE_ID = "image_id"
COL_TARGET = "dx"  # 你的 label 列（此前你用 dx）
# 如果你的 baseline CSV label 列叫 label，请改成 "label"

# 4分类（无 bkl）
CLASSES = ["akiec", "bcc", "nev", "mel"]

# 设备与精度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# 视觉 token 控制（强烈建议：和 LoRA 评估保持一致）
# 你之前 LoRA/基线速度差很多，常常就是这里不一致导致的
USE_MIN_MAX_PIXELS = True
MIN_PIXELS = 224 * 224
MAX_PIXELS = 1280 * 1280  # 如仍慢/显存紧张可降到 384*384 或 336*336

# 临床文本（如果你的 CSV 有临床字段，可拼接；没有也没关系）
USE_CLINICAL_NOTE = True
CLINICAL_NOTE_MAX_CHARS = 600

# 日志与保存
PRINT_EACH_SAMPLE = True
SAVE_PREDS_CSV = True
SAVE_PREDS_JSONL = True

# ROC/PR 方式：
# - use_probs=True：用真实概率画 ROC/PR（更正确）
# - use_probs=False：用 one-hot(pred) 伪分数画 ROC/PR（更接近你之前 medgamma 里那种“简化”）
ROC_PR_USE_PROBS = True

# 随机种子（保证可复现）
SEED = 42


# =========================
# 工具与辅助
# =========================

def normalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def truncate_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


def build_clinical_note(row: pd.Series) -> str:
    """
    如果你 CSV 里没有这些字段，也不会报错（get 会返回空）。
    你可以按你自己的 evaluate_qwen_lora 的拼接方式来改；
    关键是 baseline 与 lora 评估保持一致。
    """
    # 示例字段（按需替换）
    age = row.get("年龄", row.get("age", ""))
    sex = row.get("性别", row.get("sex", ""))
    region = row.get("区域", row.get("region", ""))

    d1 = row.get("直径1", row.get("diameter1", ""))
    d2 = row.get("直径2", row.get("diameter2", ""))

    note = f"{age}岁{sex}，部位：{region}。"
    if d1 or d2:
        note += f"皮损大小：{d1}×{d2}。"

    # 控制长度（不要 truncation 多模态 input！）
    note = truncate_text(note, CLINICAL_NOTE_MAX_CHARS)
    return note


def build_messages_qwen(clinical_note: str) -> List[dict]:
    """
    Qwen2.5-VL：必须有 image 占位符，否则会出现 tokens=0 features>0 的报错。
    """
    system_text = (
        "You are a dermatology assistant. "
        "Given the clinical note and the skin lesion image, "
        "classify the lesion into one of: akiec, bcc, nev, mel. "
        "Return exactly one lowercase label from this set with no extra words."
    )

    user_text = (
        f"Clinical note:\n{clinical_note}\n\n"
        "Please classify the lesion into one of the following labels:\n"
        "akiec, bcc, nev, mel.\n"
        "Final answer:"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    return messages


@dataclass
class EvalStats:
    total_rows: int = 0
    used: int = 0
    correct: int = 0
    skipped_bad_label: int = 0
    missing_image: int = 0
    bad_image: int = 0
    seconds_total: float = 0.0


# =========================
# 模型与 Processor
# =========================

def load_processor() -> AutoProcessor:
    if USE_MIN_MAX_PIXELS:
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    else:
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 一致性：padding side
    try:
        processor.tokenizer.padding_side = "right"
    except Exception:
        pass

    return processor


def load_model() -> AutoModelForImageTextToText:
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=DTYPE,
        trust_remote_code=True,
    )
    # 评估下可关 cache，节省显存波动（也让速度更稳定）
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    if DEVICE == "cuda":
        model.to("cuda")
    model.eval()
    return model


def compute_label_token_ids(processor: AutoProcessor) -> Dict[str, int]:
    """
    与你之前 qwen_lora 评估一致的做法：
    - 对每个类别取 tokenizer(label) 的首 token id
    """
    out = {}
    for c in CLASSES:
        ids = processor.tokenizer(c, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"Tokenizer 对标签 {c} 输出为空，请检查 tokenizer。")
        out[c] = int(ids[0])
    return out


# =========================
# 推理核心（方案A：linear logits）
# =========================

@torch.inference_mode()
def infer_linear_one(
    model,
    processor,
    label_token_ids: Dict[str, int],
    image: Image.Image,
    clinical_note: str,
) -> Tuple[str, np.ndarray]:
    """
    返回:
      pred_label: str
      probs: np.ndarray shape=(4,)
    """
    messages = build_messages_qwen(clinical_note)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 关键：不要 truncation=max_length（会导致 image token mismatch）
    inputs = processor(
        text=[prompt_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

    # last token logits: (1, vocab)
    last_logits = outputs.logits[:, -1, :]

    # bf16 -> fp32 再做 softmax / numpy
    last_logits = last_logits.float()

    # 取 4 个 label token 对应的 logit
    ids = [label_token_ids[c] for c in CLASSES]
    logits4 = last_logits[:, ids]  # (1, 4)

    probs = torch.softmax(logits4, dim=-1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    return pred_label, probs


# =========================
# 评估主流程
# =========================

def evaluate_baseline_linear_full():
    warnings.filterwarnings("once")
    set_seed(SEED)

    t0 = time.time()

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"未找到 TEST_CSV: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV, encoding="utf-8")
    stats = EvalStats(total_rows=len(df))

    if COL_IMAGE_ID not in df.columns or COL_TARGET not in df.columns:
        raise ValueError(f"TEST_CSV 缺少列：{COL_IMAGE_ID} 或 {COL_TARGET}")

    # 只保留 4类
    df[COL_TARGET] = df[COL_TARGET].astype(str).apply(normalize_label)

    print(f"📄 Test CSV: {TEST_CSV}")
    print(f"📄 总行数: {len(df)}")
    print(f"🔧 设备: {DEVICE}, dtype: {DTYPE}")
    if USE_MIN_MAX_PIXELS:
        print(f"🔧 min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}")

    processor = load_processor()
    model = load_model()
    label_token_ids = compute_label_token_ids(processor)
    print("🔧 label_token_ids:", label_token_ids)

    y_true: List[str] = []
    y_pred: List[str] = []
    y_probs: List[np.ndarray] = []

    # 逐样本保存
    preds_rows = []
    jsonl_path = os.path.join(OUT_DIR, "preds.jsonl")

    # 清空旧 jsonl
    if SAVE_PREDS_JSONL and os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        image_id = str(getattr(row, COL_IMAGE_ID))
        true_label = normalize_label(str(getattr(row, COL_TARGET)))

        if true_label not in CLASSES:
            stats.skipped_bad_label += 1
            continue

        img_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        if not os.path.exists(img_path):
            stats.missing_image += 1
            continue

        img = safe_open_image(img_path)
        if img is None:
            stats.bad_image += 1
            continue

        clinical_note = build_clinical_note(pd.Series(row._asdict())) if USE_CLINICAL_NOTE else ""

        t1 = time.time()
        pred_label, probs = infer_linear_one(
            model=model,
            processor=processor,
            label_token_ids=label_token_ids,
            image=img,
            clinical_note=clinical_note,
        )
        dt = time.time() - t1

        stats.used += 1
        correct = int(pred_label == true_label)
        stats.correct += correct

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_probs.append(probs)

        rec = {
            "idx": stats.used,
            "image_id": image_id,
            "true": true_label,
            "pred": pred_label,
            "correct": bool(correct),
            "probs": [float(x) for x in probs.tolist()],
            "latency_sec": float(dt),
            "image_path": img_path,
        }
        preds_rows.append(rec)

        if SAVE_PREDS_JSONL:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if PRINT_EACH_SAMPLE:
            print(
                f"🩺 [{stats.used}] image_id={image_id} | pred={pred_label} | true={true_label} | "
                f"{'✅' if correct else '❌'} | probs={np.round(probs, 4)} | {dt*1000:.1f} ms"
            )

        # 可选：定期清缓存，让显存更稳定
        if DEVICE == "cuda" and stats.used % 50 == 0:
            torch.cuda.empty_cache()

    stats.seconds_total = time.time() - t0

    # =========================
    # 输出报告
    # =========================
    if stats.used == 0:
        print("⚠ 没有有效样本用于评估。请检查 CSV label 与图片路径。")
        return

    acc = stats.correct / stats.used
    print("\n====== 📊 Baseline Linear Evaluation (Qwen2.5-VL, 4 classes) ======")
    print(f"总行数: {stats.total_rows}")
    print(f"有效样本: {stats.used}")
    print(f"缺失图片: {stats.missing_image}")
    print(f"坏图/打不开: {stats.bad_image}")
    print(f"跳过(非4类): {stats.skipped_bad_label}")
    print(f"Accuracy: {acc:.4%}")
    print(f"总耗时: {stats.seconds_total:.2f} sec, 平均每张: {stats.seconds_total/max(stats.used,1):.3f} sec")

    report = classification_report(y_true, y_pred, labels=CLASSES, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    print("\n====== classification_report ======")
    print(report)
    print("\n====== Confusion matrix (rows=true, cols=pred) ======")
    print(CLASSES)
    print(cm)

    # 保存 report.txt
    report_path = os.path.join(OUT_DIR, "baseline_linear_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Baseline Linear Evaluation (Qwen2.5-VL, 4 classes)\n")
        f.write(f"BASE_MODEL: {BASE_MODEL}\n")
        f.write(f"TEST_CSV: {TEST_CSV}\n")
        f.write(f"IMAGE_ROOT_DIR: {IMAGE_ROOT_DIR}\n")
        f.write(f"min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}\n" if USE_MIN_MAX_PIXELS else "min/max_pixels: default\n")
        f.write(f"used={stats.used}, missing_image={stats.missing_image}, bad_image={stats.bad_image}, skipped_bad_label={stats.skipped_bad_label}\n")
        f.write(f"accuracy={acc:.6f}\n")
        f.write(f"total_sec={stats.seconds_total:.3f}, avg_sec={stats.seconds_total/max(stats.used,1):.6f}\n\n")
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(CLASSES) + "\n")
        f.write(np.array2string(cm) + "\n")
    print(f"\n📁 报告已保存: {report_path}")

    # 保存 preds.csv
    if SAVE_PREDS_CSV:
        preds_df = pd.DataFrame(preds_rows)
        preds_csv_path = os.path.join(OUT_DIR, "preds.csv")
        preds_df.to_csv(preds_csv_path, index=False, encoding="utf-8-sig")
        print(f"📁 逐样本预测已保存: {preds_csv_path}")

    if SAVE_PREDS_JSONL:
        print(f"📁 逐样本预测 JSONL 已保存: {jsonl_path}")

    # =========================
    # 画混淆矩阵图
    # =========================
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig_cm.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Baseline, Linear)")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(
                c, r, str(cm[r, c]),
                ha="center", va="center",
                color="white" if cm[r, c] > thresh else "black",
                fontsize=10
            )
    fig_cm.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"📁 混淆矩阵图已保存: {cm_path}")

    # =========================
    # ROC / PR
    # =========================
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)  # (N,4)

    y_true_bin = label_binarize(y_true_arr, classes=CLASSES)

    if ROC_PR_USE_PROBS:
        scores = y_probs_arr
        score_tag = "probs"
    else:
        # 伪分数（one-hot）
        scores = np.zeros_like(y_true_bin, dtype=float)
        for i_, p in enumerate(y_pred):
            scores[i_, CLASSES.index(p)] = 1.0
        score_tag = "onehot"

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(CLASSES):
        # 某类在测试集中如果全是 0 或全是 1，会报错，做保护
        if y_true_bin[:, idx].sum() == 0 or y_true_bin[:, idx].sum() == len(y_true_bin):
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], scores[:, idx])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="chance")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC (Baseline Linear, {score_tag})")
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    roc_path = os.path.join(OUT_DIR, f"roc_curve_{score_tag}.png")
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)
    print(f"📁 ROC 图已保存: {roc_path}")

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for idx, cls in enumerate(CLASSES):
        if y_true_bin[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], scores[:, idx])
        ap = average_precision_score(y_true_bin[:, idx], scores[:, idx])
        ax_pr.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR (Baseline Linear, {score_tag})")
    ax_pr.legend(fontsize=8)
    fig_pr.tight_layout()
    pr_path = os.path.join(OUT_DIR, f"pr_curve_{score_tag}.png")
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"📁 PR 图已保存: {pr_path}")

    # 保存配置快照
    cfg = {
        "BASE_MODEL": BASE_MODEL,
        "TEST_CSV": TEST_CSV,
        "IMAGE_ROOT_DIR": IMAGE_ROOT_DIR,
        "IMAGE_EXT": IMAGE_EXT,
        "CLASSES": CLASSES,
        "DEVICE": DEVICE,
        "DTYPE": str(DTYPE),
        "USE_MIN_MAX_PIXELS": USE_MIN_MAX_PIXELS,
        "MIN_PIXELS": MIN_PIXELS,
        "MAX_PIXELS": MAX_PIXELS,
        "USE_CLINICAL_NOTE": USE_CLINICAL_NOTE,
        "CLINICAL_NOTE_MAX_CHARS": CLINICAL_NOTE_MAX_CHARS,
        "ROC_PR_USE_PROBS": ROC_PR_USE_PROBS,
        "SEED": SEED,
        "stats": stats.__dict__,
        "accuracy": float(acc),
    }
    cfg_path = os.path.join(OUT_DIR, "config_snapshot.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"📁 配置快照已保存: {cfg_path}")


if __name__ == "__main__":
    evaluate_baseline_linear_full()
