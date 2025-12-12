# make_splits_balanced.py
# -*- coding: utf-8 -*-

import os
import random
from collections import defaultdict

import pandas as pd

# ====== 路径配置：改成你自己的即可 ======
INPUT_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata.csv"

TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train_balanced.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 只保留这 5 类
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]


def normalize_dx(label: str) -> str:
    """和训练脚本保持一致：小写 + nv -> nev"""
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """按 dx 做分层划分 -> train / val / test"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    df = df.copy()
    df["dx_norm"] = df["dx"].apply(normalize_dx)
    df = df[df["dx_norm"].isin(ALLOWED_DX)].reset_index(drop=True)

    groups = defaultdict(list)
    for idx, row in df.iterrows():
        groups[row["dx_norm"]].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for cls, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)

        # 基于比例的数量，至少保证每个 split 有 1 个
        n_val = max(1, round(val_ratio * n))
        n_test = max(1, round(test_ratio * n))
        # 防止极端情况：val+test >= n
        if n_val + n_test >= n:
            n_val = max(1, min(n - 2, n_val))
            n_test = max(1, min(n - 1 - n_val, n_test))

        n_train = n - n_val - n_test

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def make_balanced_train(
    train_df: pd.DataFrame,
    target_per_class: int = 300,
    max_upsample_factor: int = 5,
    seed: int = 42,
):
    """
    只对训练集做类平衡：
      - 对多数类：下采样到 target_per_class（比如 bcc）
      - 对少数类：最多放大到 max_upsample_factor 倍（避免过度复制）
    """
    rng = random.Random(seed)
    df = train_df.copy()
    df["dx_norm"] = df["dx"].apply(normalize_dx)

    balanced_parts = []
    stats = {}

    for cls in ALLOWED_DX:
        cls_df = df[df["dx_norm"] == cls]
        n = len(cls_df)
        if n == 0:
            continue

        # 最多放大到 n * max_upsample_factor
        max_target = n * max_upsample_factor
        t = min(target_per_class, max_target)

        if t <= n:
            # 足够多：随机下采样
            sampled = cls_df.sample(n=t, random_state=seed)
        else:
            # 不够：重复 + 余数采样
            reps = t // n
            rem = t % n
            sampled_list = [cls_df] * reps
            if rem > 0:
                sampled_list.append(cls_df.sample(n=rem, random_state=seed))
            sampled = pd.concat(sampled_list, ignore_index=True)

        balanced_parts.append(sampled)
        stats[cls] = (n, len(sampled))

    balanced_df = pd.concat(balanced_parts, ignore_index=True)

    # 打乱顺序
    indices = list(balanced_df.index)
    rng.shuffle(indices)
    balanced_df = balanced_df.loc[indices].reset_index(drop=True)

    return balanced_df, stats


def main():
    print(f"📄 读取原始 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    # 1. 按原始分布做分层划分
    train_df, val_df, test_df = stratified_split(df)

    print("📊 原始分布下的划分：")
    print("  train 总数:", len(train_df), "\n", train_df["dx"].apply(normalize_dx).value_counts(), "\n")
    print("  val   总数:", len(val_df), "\n", val_df["dx"].apply(normalize_dx).value_counts(), "\n")
    print("  test  总数:", len(test_df), "\n", test_df["dx"].apply(normalize_dx).value_counts(), "\n")

    # 2. 对训练集做类平衡版
    balanced_train_df, stats = make_balanced_train(
        train_df,
        target_per_class=300,      # 你可以改大/改小
        max_upsample_factor=5,     # 最多放大 5 倍，防止过拟合
    )

    print("📊 训练集类平衡前后对比（原始数量 -> 平衡后数量）：")
    for cls in ALLOWED_DX:
        old_n, new_n = stats.get(cls, (0, 0))
        print(f"  {cls:5s}: {old_n:4d}  ->  {new_n:4d}")
    print("\n📊 平衡后训练集分布：")
    print(balanced_train_df["dx"].apply(normalize_dx).value_counts())

    # 3. 保存到 CSV
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)

    balanced_train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print(f"\n💾 已保存：")
    print(f"  训练集（平衡版） → {TRAIN_CSV}")
    print(f"  验证集           → {VAL_CSV}")
    print(f"  测试集           → {TEST_CSV}")


if __name__ == "__main__":
    main()
