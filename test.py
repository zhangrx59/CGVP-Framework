import os
import pandas as pd
import shutil

# ======= 请修改成你实际的路径 =======
CSV_PATH = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train_balanced.csv"
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"

# 可选：删除的文件备份到这里（如果不想备份，设为 None）
BACKUP_DIR = None
# BACKUP_DIR = None  # 如果你想直接删掉，不备份，请用这一行替换上面一行

# =============== 1. 读取 CSV ===============
df = pd.read_csv(CSV_PATH, encoding="utf-8")
print(f"原始数据集总行数：{len(df)}")

# =============== 2. 找到所有 bkl 行 ===============
bkl_df = df[df["dx"].str.lower() == "bkl"]
print(f"检测到 bkl 行数：{len(bkl_df)}")

# =============== 3. 删除 bkl 行 ===============
df_clean = df[df["dx"].str.lower() != "bkl"].copy()
df_clean.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"已写回清洗后的 CSV：{CSV_PATH}")
print(f"清洗后数据集总行数：{len(df_clean)}")

# =============== 4. 删除图片文件 ===============
if BACKUP_DIR is not None:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    print(f"bkl 图片将被备份到：{BACKUP_DIR}")

count_deleted = 0
for _, row in bkl_df.iterrows():
    image_id = str(row["image_id"]).strip("[]'\" ").replace(",", "").strip()
    img_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)

    if os.path.exists(img_path):
        if BACKUP_DIR is not None:
            shutil.move(img_path, os.path.join(BACKUP_DIR, image_id + IMAGE_EXT))
        else:
            os.remove(img_path)
        count_deleted += 1

print(f"已删除 bkl 图片数量：{count_deleted}")
print("✅ 数据集清洗完成（bkl 已完全移除）")
