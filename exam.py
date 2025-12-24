# -*- coding: utf-8 -*-
"""
ISIC 数据完整性体检脚本
- 遍历图片文件夹，作为“真实图片集合”
- 读取 train / val / test CSV（第一列是图片名）
- 对比三者，找出：
  1) CSV 中有但文件夹中不存在的图片
  2) 文件夹中存在但 CSV 从未使用的图片
  3) train / val / test 之间的交集（数据泄露）
  4) 各集合的唯一图片数量 vs CSV 行数
"""

import os
from pathlib import Path
import pandas as pd

# ===================== 你只需要改这里 =====================
IMG_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"

TRAIN_CSV = r"C:\Users\zhangrx59\Desktop\dataset\metadata_train.csv"
VAL_CSV   = r"C:\Users\zhangrx59\Desktop\dataset\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\Desktop\dataset\metadata_test.csv"

# 支持的图片后缀（按你的实际情况）
IMG_EXTS = {".jpg", ".jpeg", ".png"}
# =========================================================


def load_image_set(img_dir: str):
    """从文件夹中读取所有图片名（不含后缀）"""
    img_set = set()
    for fn in os.listdir(img_dir):
        p = Path(fn)
        if p.suffix.lower() in IMG_EXTS:
            img_set.add(p.stem)  # 只保留文件名，不含后缀
    return img_set


def load_csv_set(csv_path: str):
    """从 CSV 第一列读取图片名（去后缀，去空格）"""
    df = pd.read_csv(csv_path)
    col = df.columns[0]

    names = set()
    for x in df[col].astype(str):
        x = x.strip()
        if not x:
            continue
        names.add(Path(x).stem)  # 统一成不含后缀
    return names, len(df)


def main():
    print("===== 📂 扫描图片文件夹 =====")
    img_set = load_image_set(IMG_DIR)
    print(f"真实图片文件数（文件夹）：{len(img_set)}")

    print("\n===== 📄 读取 CSV =====")
    train_set, train_rows = load_csv_set(TRAIN_CSV)
    val_set,   val_rows   = load_csv_set(VAL_CSV)
    test_set,  test_rows  = load_csv_set(TEST_CSV)

    print(f"TRAIN: 行数={train_rows}, 唯一图片数={len(train_set)}")
    print(f"VAL  : 行数={val_rows},   唯一图片数={len(val_set)}")
    print(f"TEST : 行数={test_rows},  唯一图片数={len(test_set)}")

    print("\n===== 🔍 CSV 内部异常 =====")
    if train_rows > len(train_set):
        print(f"[WARN] TRAIN 中存在重复图片行：{train_rows - len(train_set)} 条")
    if val_rows > len(val_set):
        print(f"[WARN] VAL 中存在重复图片行：{val_rows - len(val_set)} 条")
    if test_rows > len(test_set):
        print(f"[WARN] TEST 中存在重复图片行：{test_rows - len(test_set)} 条")

    print("\n===== ❌ CSV 引用但文件夹中不存在的图片 =====")
    train_missing = train_set - img_set
    val_missing   = val_set - img_set
    test_missing  = test_set - img_set

    print(f"TRAIN 缺失：{len(train_missing)}")
    print(f"VAL   缺失：{len(val_missing)}")
    print(f"TEST  缺失：{len(test_missing)}")

    # 打印前若干个样例
    def preview(s, k=10):
        return list(s)[:k]

    if train_missing:
        print("  示例（TRAIN）：", train_missing)
    if val_missing:
        print("  示例（VAL）：", preview(val_missing))
    if test_missing:
        print("  示例（TEST）：", preview(test_missing))

    print("\n===== ❌ 文件夹中存在但 CSV 从未使用的图片 =====")
    used_all = train_set | val_set | test_set
    unused = img_set - used_all
    print(f"未被任何 CSV 使用的图片数：{len(unused)}")
    if unused:
        print("  示例：", preview(unused))

    print("\n===== 🚨 集合之间的交集（数据泄露检查） =====")
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    print(f"TRAIN ∩ VAL  : {len(tv)}")
    print(f"TRAIN ∩ TEST : {len(tt)}")
    print(f"VAL   ∩ TEST : {len(vt)}")

    if tv:
        print("  示例（TRAIN∩VAL）：", preview(tv))
    if tt:
        print("  示例（TRAIN∩TEST）：", preview(tt))
    if vt:
        print("  示例（VAL∩TEST）：", preview(vt))

    print("\n===== ✅ 总结 =====")
    print(f"CSV 总唯一图片数：{len(used_all)}")
    print(f"文件夹真实图片数：{len(img_set)}")

    if len(used_all) != len(img_set):
        print("[WARN] CSV 使用的图片集合 ≠ 文件夹中的真实图片集合")
    else:
        print("[OK] CSV 与文件夹图片集合一致")

if __name__ == "__main__":
    main()
