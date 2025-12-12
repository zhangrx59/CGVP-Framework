# -*- coding: utf-8 -*-
"""
Test ResNet50 on custom dataset with saved weights
可视化并保存：
1. Confusion Matrix 热力图
2. 每类疾病的 ROC 曲线（多条曲线一张图）
3. 每类疾病的 Precision-Recall 曲线（多条曲线一张图）
4. 文本版分类报告（打印 + 保存为 txt）
"""

import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


# ================== 0. 路径配置（按实际情况修改） ==================

TEST_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 图片根目录（和训练脚本保持一致）
IMAGE_ROOT_DIR = r"/ISIC_dataset"

# 会依次尝试的图片后缀（和训练脚本保持一致）
IMAGE_EXT_CANDIDATES = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

# 训练脚本中保存的 best 模型路径
BEST_MODEL_PATH = r"../best_resnet50_custom_cbam_focal.pth"

# 输出图片/报告目录
PICS_DIR = r".\resnet50_eval_outputs"


# ================== 1. 工具函数和模块 ==================

def build_image_path(image_id: str) -> str:
    """
    根据 image_id 在 IMAGE_ROOT_DIR 中尝试不同后缀找到图片路径
    """
    candidates = [
        os.path.join(IMAGE_ROOT_DIR, f"{image_id}{ext}")
        for ext in IMAGE_EXT_CANDIDATES
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 都没找到就返回第一个，用于调试
    return candidates[0]


class TestDermDataset(Dataset):
    """
    简单测试集 Dataset
    需要 df 至少包含：
      - path: 图片路径
      - label_idx: 整数标签
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.loc[index, "path"]
        X = Image.open(img_path).convert("RGB")
        y = torch.tensor(int(self.df.loc[index, "label_idx"]))

        if self.transform:
            X = self.transform(X)

        return X, y


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


def build_model(num_classes: int) -> nn.Module:
    """
    搭建和训练脚本完全一致的结构：
      ResNet50 + 在 layer4 后加 CBAMBlock + 最后一层 fc 输出 num_classes
    """
    model_ft = resnet50(weights=None)  # 这里不需要预训练权重，后面会 load_state_dict 覆盖
    model_ft.layer4 = nn.Sequential(
        model_ft.layer4,
        CBAMBlock(in_planes=2048, reduction=16, kernel_size=7)
    )
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    打印并绘制混淆矩阵。
    """
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ================== 2. 主流程 ==================

def main():
    # -------- 2.1 准备输出目录 --------
    os.makedirs(PICS_DIR, exist_ok=True)

    # -------- 2.2 读取测试集 CSV --------
    assert os.path.exists(TEST_CSV), f"TEST_CSV not found: {TEST_CSV}"
    df_test = pd.read_csv(TEST_CSV)

    # 只保留 image_id / dx 不为空的行
    df_test = df_test.dropna(subset=["image_id", "dx"])
    df_test["image_id"] = df_test["image_id"].astype(str)

    # -------- 2.3 加载模型和 dx_categories --------
    assert os.path.exists(BEST_MODEL_PATH), f"Model file not found: {BEST_MODEL_PATH}"
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")

    if "dx_categories" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain 'dx_categories'. "
                           "请确认训练脚本保存了 dx_categories。")

    dx_categories = checkpoint["dx_categories"]
    num_classes = len(dx_categories)
    print("Loaded dx_categories from checkpoint:", dx_categories)

    dx_to_idx = {dx: i for i, dx in enumerate(dx_categories)}

    # 过滤掉 dx 不在 dx_categories 的样本（理论上不会发生）
    df_test = df_test[df_test["dx"].isin(dx_categories)].reset_index(drop=True)

    # 映射标签为索引
    df_test["label_idx"] = df_test["dx"].map(dx_to_idx)

    # 构造图片路径并过滤不存在的文件
    df_test["path"] = df_test["image_id"].apply(build_image_path)
    before = len(df_test)
    df_test = df_test[df_test["path"].apply(os.path.exists)].reset_index(drop=True)
    dropped = before - len(df_test)
    if dropped > 0:
        print(f"[test] dropped {dropped} rows because image file not found.")

    print(f"Test size (after filtering): {len(df_test)}")
    print("Test label counts:\n", df_test["label_idx"].value_counts())

    # -------- 2.4 构建 Dataset / DataLoader --------
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_set = TestDermDataset(df_test, transform=test_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=pin_memory)

    # -------- 2.5 搭建模型并加载权重 --------
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # -------- 2.6 推理，收集预测结果 --------
    y_true, y_pred = [], []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)                  # [N, C]
            probs = F.softmax(outputs, dim=1)        # [N, C]
            preds = outputs.max(1, keepdim=False)[1] # [N]

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_scores = np.concatenate(all_probs, axis=0)  # [N, C]

    # ================== 3. 指标 & 可视化 ==================

    # -------- 3.1 混淆矩阵 --------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes=dx_categories,
                          normalize=False,
                          title="Test Confusion Matrix")
    cm_path = os.path.join(PICS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix -> {cm_path}")

    # -------- 3.2 文本分类报告 --------
    report = classification_report(
        y_true,
        y_pred,
        target_names=dx_categories,
        digits=4
    )
    print("\n====== Classification report (Test set) ======\n")
    print(report)

    report_path = os.path.join(PICS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("====== Classification report (Test set) ======\n\n")
        f.write(report)
    print(f"[SAVED] Text classification report -> {report_path}")

    # -------- 3.3 多类别 ROC 曲线 --------
    # 将真实标签 one-hot 化
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    # 若类别数为 2，label_binarize 输出 [N, 1]，我们也按二维处理
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(
            fpr_dict[i],
            tpr_dict[i],
            lw=1.5,
            label=f"{dx_categories[i]} (AUC = {roc_auc_dict[i]:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)  # 随机猜测基线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC curves (Test set)")
    plt.legend(loc="lower right", fontsize=8)
    roc_path = os.path.join(PICS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] ROC curves -> {roc_path}")

    # -------- 3.4 多类别 Precision-Recall 曲线 --------
    precision_dict = {}
    recall_dict = {}
    ap_dict = {}

    for i in range(num_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_scores[:, i]
        )
        ap_dict[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(
            recall_dict[i],
            precision_dict[i],
            lw=1.5,
            label=f"{dx_categories[i]} (AP = {ap_dict[i]:.3f})"
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class Precision-Recall curves (Test set)")
    plt.legend(loc="lower left", fontsize=8)
    pr_path = os.path.join(PICS_DIR, "pr_curves.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Precision-Recall curves -> {pr_path}")

    # -------- 3.5 最后的文字汇总 --------
    print("\n=================== Finished ===================")
    print("✅ Text classification report has been printed above.")
    print(f"   And saved to: {report_path}")
    print("✅ All images (Confusion Matrix, ROC Curves, PR Curves) have been saved to:")
    print(f"   📁 {PICS_DIR}")
    print("You can open them for visualization or include them in reports/papers.")
    print("=================================================\n")


if __name__ == "__main__":
    main()
