# -*- coding: utf-8 -*-
"""
删除 metadata CSV 中 dx == 'bkl' 的所有行
- 输入：train / val / test CSV
- 输出：*_no_bkl.csv（不覆盖原文件）
"""

import pandas as pd
from pathlib import Path

# ===================== 路径配置 =====================
TRAIN_CSV = r"C:\Users\zhangrx59\Desktop\dataset\metadata_train.csv"
VAL_CSV   = r"C:\Users\zhangrx59\Desktop\dataset\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\Desktop\dataset\metadata_test.csv"
# ===================================================

def remove_bkl(csv_path: str):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "dx" not in df.columns:
        raise ValueError(f"{csv_path.name} 中找不到 dx 列，当前列：{list(df.columns)}")

    before = len(df)

    # 统一大小写 + 去空格，再过滤
    df["dx"] = df["dx"].astype(str).str.strip().str.lower()
    df_clean = df[df["dx"] != "bkl"].copy()

    after = len(df_clean)

    out_path = csv_path.with_name(csv_path.stem + "_no_bkl.csv")
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {csv_path.name}")
    print(f"  原始行数: {before}")
    print(f"  删除 bkl : {before - after}")
    print(f"  剩余行数: {after}")
    print(f"  输出文件: {out_path}\n")

def main():
    remove_bkl(TRAIN_CSV)
    remove_bkl(VAL_CSV)
    remove_bkl(TEST_CSV)

if __name__ == "__main__":
    main()
