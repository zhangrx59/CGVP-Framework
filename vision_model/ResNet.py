# -*- coding: utf-8 -*-
"""
ResNet50 + CBAM + FocalLoss 训练脚本（带类平衡 alpha + 进度条）
适配 metadata_train.csv / metadata_val.csv / metadata_test.csv：
  - 必须包含列: image_id, dx
  - 其它临床列会被忽略
"""

import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm   # <<< 用于进度条


# ================== 0. 路径配置（请按实际修改） ==================

TRAIN_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_train.csv"
VAL_CSV   = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_val.csv"
TEST_CSV  = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_test.csv"

# 图片根目录（所有图片都在这个文件夹里，文件名形如 PAT_xxx_yyy_zzz.png）
IMAGE_ROOT_DIR = r"/ISIC_dataset"

# 会依次尝试的图片后缀
IMAGE_EXT_CANDIDATES = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


# =========== 工具函数和 Dataset ===========

def set_seed(seed: int = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """用于统计 loss / acc 的平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def build_image_path(image_id: str) -> str:
    """
    根据 image_id 在 IMAGE_ROOT_DIR 中尝试不同后缀找到图片路径
    找不到时返回第一个候选路径（即使不存在，方便你调试）
    """
    candidates = [
        os.path.join(IMAGE_ROOT_DIR, f"{image_id}{ext}")
        for ext in IMAGE_EXT_CANDIDATES
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


class GenericDermDataset(Dataset):
    """
    通用 Dataset，要求 df 至少包含：
      - path: 图片完整路径
      - cell_type_idx: 整数标签
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.loc[index, "path"]
        X = Image.open(img_path).convert("RGB")
        y = torch.tensor(int(self.df.loc[index, "cell_type_idx"]))

        if self.transform:
            X = self.transform(X)

        return X, y


def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# =========== CBAM 注意力模块 ===========

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


# =========== Focal Loss ===========

class FocalLoss(nn.Module):
    """
    多分类 Focal Loss
    inputs: [N, C], targets: [N]
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits [N, C]
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))
        targets = targets.view(-1)

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)  # pt = prob of correct class
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def initialize_model(num_classes: int,
                     feature_extract: bool,
                     use_pretrained: bool = True):
    if use_pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model_ft = resnet50(weights=weights)

    # 在 layer4 后面加 CBAMBlock（通道数 2048）
    model_ft.layer4 = nn.Sequential(
        model_ft.layer4,
        CBAMBlock(in_planes=2048, reduction=16, kernel_size=7)
    )

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


# =========== 训练 & 验证函数（带 tqdm） ===========

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device,
                    total_loss_train, total_acc_train):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch} [Train]",
        ncols=100
    )

    for i, (images, labels) in progress_bar:
        N = images.size(0)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prediction = outputs.max(1, keepdim=True)[1]
        acc = prediction.eq(labels.view_as(prediction)).sum().item() / N

        train_acc.update(acc, N)
        train_loss.update(loss.item(), N)

        # 更新进度条后缀
        progress_bar.set_postfix(loss=f"{train_loss.avg:.4f}",
                                 acc=f"{train_acc.avg:.4f}")

        # 为了后面画曲线，隔一段记录一次
        if (i + 1) % 20 == 0:
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)

    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    progress_bar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Epoch {epoch} [Val]",
        ncols=100
    )

    with torch.no_grad():
        for i, (images, labels) in progress_bar:
            N = images.size(0)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            acc = prediction.eq(labels.view_as(prediction)).sum().item() / N
            val_acc.update(acc, N)

            loss = criterion(outputs, labels).item()
            val_loss.update(loss, N)

            progress_bar.set_postfix(loss=f"{val_loss.avg:.4f}",
                                     acc=f"{val_acc.avg:.4f}")

    print('------------------------------------------------------------')
    print(f"[epoch {epoch}], [val loss {val_loss.avg:.5f}], [val acc {val_acc.avg:.5f}]")
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    打印并绘制混淆矩阵。
    Normalization 可以通过 normalize=True 开启。
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


# =========== 主流程 ===========

def main():
    set_seed(10)

    # ====== 1. 读取三个划分好的 CSV ======
    assert os.path.exists(TRAIN_CSV), f"TRAIN_CSV not found: {TRAIN_CSV}"
    assert os.path.exists(VAL_CSV),   f"VAL_CSV not found: {VAL_CSV}"
    assert os.path.exists(TEST_CSV),  f"TEST_CSV not found: {TEST_CSV}"

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # 只保留 image_id 和 dx 不为空的样本
    train_df = train_df.dropna(subset=["image_id", "dx"])
    val_df   = val_df.dropna(subset=["image_id", "dx"])
    test_df  = test_df.dropna(subset=["image_id", "dx"])

    train_df["image_id"] = train_df["image_id"].astype(str)
    val_df["image_id"]   = val_df["image_id"].astype(str)
    test_df["image_id"]  = test_df["image_id"].astype(str)

    # ====== 2. 统一类别顺序（从三份数据 union，再排序） ======
    all_df = pd.concat([train_df[["dx"]], val_df[["dx"]], test_df[["dx"]]], axis=0)
    dx_categories = sorted(all_df["dx"].unique().tolist())
    dx_to_idx = {dx: i for i, dx in enumerate(dx_categories)}
    print("Detected dx categories:", dx_categories)

    train_df["cell_type_idx"] = train_df["dx"].map(dx_to_idx)
    val_df["cell_type_idx"]   = val_df["dx"].map(dx_to_idx)
    test_df["cell_type_idx"]  = test_df["dx"].map(dx_to_idx)

    # ====== 3. 构造图片路径，并过滤不存在的图片 ======
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df["path"] = df["image_id"].apply(build_image_path)
        before = len(df)
        df_valid = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
        dropped = before - len(df_valid)
        if dropped > 0:
            print(f"[{name}] dropped {dropped} rows because image not found.")

        if name == "train":
            train_df = df_valid
        elif name == "val":
            val_df = df_valid
        else:
            test_df = df_valid

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print("Train label counts:\n", train_df["cell_type_idx"].value_counts())
    print("Val label counts:\n",   val_df["cell_type_idx"].value_counts())
    print("Test label counts:\n",  test_df["cell_type_idx"].value_counts())

    # ====== 4. 归一化参数（直接用 ImageNet 标准值） ======
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    # ====== 5. 初始化模型 ======
    num_classes = len(dx_categories)
    feature_extract = False

    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True

    model = model_ft.to(device)

    # ====== 6. transforms ======
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # ====== 7. Dataset / DataLoader ======
    pin_memory = (device.type == "cuda")

    train_set = GenericDermDataset(train_df, transform=train_transform)
    val_set   = GenericDermDataset(val_df, transform=val_transform)
    test_set  = GenericDermDataset(test_df, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)

    # ====== 8. 类平衡权重 + FocalLoss(alpha=class_weights) ======
    class_counts = train_df["cell_type_idx"].value_counts().sort_index().values.astype(float)
    print("Class counts:", class_counts)

    # 用“平均样本数 / 当前类样本数”做权重，平均权重 ≈ 1
    mean_count = class_counts.mean()
    raw_weights = mean_count / (class_counts + 1e-6)

    # 可以再做一次 sqrt，让差距不要太离谱：
    class_weights = np.sqrt(raw_weights)

    print("Class weights for FocalLoss alpha (smoothed):", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(
        alpha=class_weights_tensor,
        gamma=1.5,  # gamma 也稍微降低一点，更稳定
        reduction="mean"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ====== 9. 训练循环 ======
    epoch_num = 20
    best_val_acc = 0.0
    total_loss_train, total_acc_train = [], []
    total_loss_val, total_acc_val = [], []

    # 模型和类别名一起保存，供评估脚本读取
    best_model_path = "best_resnet50_custom_cbam_focal.pth"

    for epoch in range(1, epoch_num + 1):
        loss_train, acc_train = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, device,
            total_loss_train, total_acc_train
        )
        loss_val, acc_val = validate(val_loader, model, criterion, epoch, device)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dx_categories": dx_categories,
                },
                best_model_path,
            )
            print("*****************************************************")
            print(f"best record: [epoch {epoch}], [val loss {loss_val:.5f}], [val acc {acc_val:.5f}]")
            print(f"Model + label names saved at {best_model_path}")
            print("*****************************************************")

    # ====== 10. 画验证集损失/准确率曲线（简单版） ======
    plt.figure(figsize=(8, 6))
    plt.plot(total_acc_val, label="Validation accuracy")
    plt.plot(total_loss_val, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ====== 11. 在验证集上做混淆矩阵和报告 ======
    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            y_label.extend(labels.numpy())
            y_predict.extend(prediction.cpu().numpy().ravel())

    cm_val = confusion_matrix(y_label, y_predict)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm_val, dx_categories,
                          normalize=False,
                          title="Validation Confusion Matrix")
    plt.show()
    print("Validation classification report:")
    print(classification_report(y_label, y_predict, target_names=dx_categories))

    # ====== 12. 在测试集上评估（简单版） ======
    test_y_label = []
    test_y_predict = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            test_y_label.extend(labels.numpy())
            test_y_predict.extend(prediction.cpu().numpy().ravel())

    cm_test = confusion_matrix(test_y_label, test_y_predict)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm_test, dx_categories,
                          normalize=False,
                          title="Test Confusion Matrix")
    plt.show()
    print("Test classification report:")
    print(classification_report(test_y_label, test_y_predict, target_names=dx_categories))


if __name__ == "__main__":
    main()
