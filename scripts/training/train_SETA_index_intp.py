import os
import time
import torch
import warnings
import numpy as np
from torch import nn
import pandas as pd
from torch import optim
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
from torch.utils.data import DataLoader, random_split
from dataset.Unet_dataset_add import MyDataset
from losses.hausdorff import HausdorffDTLoss

warnings.filterwarnings("ignore")
torch.manual_seed(17)  # 设置随机种子以确保实验的可重复性

# 设置数据集路径
DATA_DIR = r'E:\A_workbench\A-lab\Unet\Unet_complit\data'
x_dir = os.path.join(DATA_DIR, 'ConvertedImages')  # 图像路径
y_dir = os.path.join(DATA_DIR, 'ResizedLabels')  # 标签路径
batch_size = 4  # 每个批次的样本数量
num_classes = 3  # 分类数量
learning_rate = 0.001  # 学习率
epochs_num = 10  # 训练的轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备：GPU 或 CPU

# 创建保存文件的目录
save_dir = "savefile"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果目录不存在，创建一个新的目录

checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)  # 如果目录不存在，创建一个新的目录

# 加载数据集
dataset = MyDataset(x_dir, y_dir, num_classes=3, resize_shape=(224, 224), augment=False, expansion_factor=5)  # 使用自定义的数据集类加载图像和标签

# 计算训练集和验证集大小
train_size = int(0.7 * len(dataset))  # 70% 用于训练
val_size = len(dataset) - train_size  # 30% 用于验证

# 定义一个函数统计数据集中每个标签的数量
def count_label_classes(data_loader, num_classes):
    label_counts = torch.zeros(num_classes)  # 初始化计数器
    for _, labels, *_ in data_loader:  # 遍历数据加载器
        for i in range(num_classes):
            label_counts[i] += torch.sum(labels == i).item()  # 统计每个类别的像素数量
    return label_counts

# 将数据集打乱并重新分为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # 加载训练集
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # 加载验证集

# 统计并打印训练集和验证集的标签类别分布
train_label_counts = count_label_classes(train_loader, num_classes)
val_label_counts = count_label_classes(val_loader, num_classes)

print(f"Train Dataset Label Counts: {train_label_counts.numpy()}")  # 打印训练集的标签分布
print(f"Validation Dataset Label Counts: {val_label_counts.numpy()}")  # 打印验证集的标签分布

# 计算类别权重
class_weights = 1.0 / (train_label_counts + 1e-10)  # 计算每个类别的权重，避免除以零
class_weights = class_weights / class_weights.sum()  # 归一化权重
class_weights = class_weights.to(device)  # 将权重转移到 GPU

# 初始化模型并移动到设备
model = UNetWithAttention(num_classes=3).to(device)

# 检查是否存在检查点以继续训练
start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}...")

# 选用SGD优化器来训练
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
if os.path.exists(checkpoint_path):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)  # 学习率调度器

from losses.lovasz_loss import LovaszSoftmax
lovasz_loss = LovaszSoftmax(reduction='mean')

# 定义一个函数计算 IoU
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float() - intersection
        if union == 0:
            ious.append(float('nan'))  # 当类别在该 batch 中不存在时，忽略该类别
        else:
            ious.append(intersection / union)  # 计算 IoU

    return torch.tensor(ious)

# 定义函数计算 F1分数, ACC, MPA, Precision, Recall, Specificity, Dice 和 Hausdorff Distance
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(pred, target, num_classes):
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()

    f1 = f1_score(target, pred, average='weighted')
    acc = np.sum(pred == target) / len(target)
    mpa = np.mean([np.sum((pred == cls) & (target == cls)) / np.sum(target == cls) for cls in range(num_classes) if np.sum(target == cls) > 0])
    precision = precision_score(target, pred, average='weighted', zero_division=0)
    recall = recall_score(target, pred, average='weighted', zero_division=0)
    specificity = np.mean([np.sum((pred != cls) & (target != cls)) / np.sum(target != cls) for cls in range(num_classes) if np.sum(target != cls) > 0])
    dice = 2 * precision * recall / (precision + recall + 1e-10)
    hausdorff_distance = np.nan  # Placeholder, needs actual computation if implemented

    return f1, acc, mpa, precision, recall, specificity, dice, hausdorff_distance


# 修改损失函数，加入权重
lossf = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

# 合成的损失函数
def combined_loss(pred, target):
    cross_entropy_loss = lossf(pred, target)  # 计算加权的交叉熵损失
    lovasz_dt_loss = lovasz_loss(pred, target)  # 计算lovasz_loss损失
    return cross_entropy_loss * 0.5 + lovasz_dt_loss * 0.5  # 将两者相加

# 定义模型评估函数
def evaluate_model(model, data_loader, device, num_classes):
    model.eval()  # 将模型设置为评估模式
    iou_list = []
    total_loss = 0.0
    f1_list = []
    acc_list = []
    mpa_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    dice_list = []
    hausdorff_list = []

    with torch.no_grad():  # 关闭梯度计算，减少内存占用
        for features, labels, *_ in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            # ---------------------------------------------在这里添加合成的loss函数
            loss = combined_loss(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            # 计算每个 batch 的 IoU 并保存
            iou = compute_iou(predicted, labels, num_classes)
            iou_list.append(iou)

            # 计算 F1, ACC, MPA, Precision, Recall, Specificity, Dice 和 Hausdorff Distance
            f1, acc, mpa, precision, recall, specificity, dice, hausdorff_distance = compute_metrics(predicted, labels, num_classes)
            f1_list.append(f1)
            acc_list.append(acc)
            mpa_list.append(mpa)
            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)
            dice_list.append(dice)
            hausdorff_list.append(hausdorff_distance)

    avg_iou = torch.stack(iou_list).nanmean()  # 计算平均 IoU，忽略 NaN
    avg_loss = total_loss / len(data_loader.dataset)
    avg_f1 = np.mean(f1_list)
    avg_acc = np.mean(acc_list)
    avg_mpa = np.mean(mpa_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_specificity = np.mean(specificity_list)
    avg_dice = np.mean(dice_list)
    avg_hausdorff = np.mean(hausdorff_list)

    return avg_iou, avg_loss, avg_f1, avg_acc, avg_mpa, avg_precision, avg_recall, avg_specificity, avg_dice, avg_hausdorff


# 训练函数
def mytrain(net, train_iter, test_iter, loss, optimizer, num_epochs, scheduler, num_classes, start_epoch=0):
    loss_list = []
    train_iou_list = []
    test_iou_list = []
    f1_list = []
    acc_list = []
    mpa_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    dice_list = []
    hausdorff_list = []
    epochs_list = []
    time_list = []

    for epoch in range(start_epoch, num_epochs):
        net.train()  # 将模型设置为训练模式
        train_loss, correct, total = 0.0, 0, 0

        epoch_start_time = time.time()  # 开始计时

        for i, (features, labels, *_) in enumerate(train_iter):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(features)  # 前向传播
            # ---------------------------------------------在这里添加合成的loss函数
            loss_value = combined_loss(outputs, labels)  # 计算损失
            loss_value.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss_value.item() * labels.size(0)

        train_loss_avg = train_loss / len(train_iter.dataset)  # 计算平均损失

        # 计算训练集上的 mIoU
        train_iou, _, _, _, _, _, _, _, _, _ = evaluate_model(net, train_iter, device, num_classes)

        # 打印训练信息
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Train mIoU: {train_iou:.4f}')

        scheduler.step()  # 更新学习率

        # 计算验证集上的所有指标
        test_iou, test_loss_avg, avg_f1, avg_acc, avg_mpa, avg_precision, avg_recall, avg_specificity, avg_dice, avg_hausdorff = evaluate_model(net, test_iter, device, num_classes)

        epoch_time = time.time() - epoch_start_time  # 记录当前epoch消耗的时间

        print(
            f"Epoch {epoch + 1} --- Train Loss: {train_loss_avg:.3f} --- Train mIoU: {train_iou:.3f} --- Test mIoU: {test_iou:.3f} --- Test Loss: {test_loss_avg:.3f} --- F1 Score: {avg_f1:.3f} --- ACC: {avg_acc:.3f} --- MPA: {avg_mpa:.3f} --- Precision: {avg_precision:.3f} --- Recall: {avg_recall:.3f} --- Specificity: {avg_specificity:.3f} --- Dice: {avg_dice:.3f} --- Hausdorff: {avg_hausdorff:.3f} --- Time: {epoch_time:.2f} sec")

        # 保存训练数据
        loss_list.append(train_loss_avg)
        train_iou_list.append(train_iou.item())
        test_iou_list.append(test_iou.item())
        f1_list.append(avg_f1)
        acc_list.append(avg_acc)
        mpa_list.append(avg_mpa)
        precision_list.append(avg_precision)
        recall_list.append(avg_recall)
        specificity_list.append(avg_specificity)
        dice_list.append(avg_dice)
        hausdorff_list.append(avg_hausdorff)
        epochs_list.append(epoch + 1)
        time_list.append(epoch_time)

        # 将训练信息保存为 Excel 文件
        df = pd.DataFrame({
            'epoch': epochs_list,
            'train_loss': loss_list,
            'train_mIoU': train_iou_list,
            'test_mIoU': test_iou_list,
            'f1_score': f1_list,
            'accuracy': acc_list,
            'mpa': mpa_list,
            'precision': precision_list,
            'recall': recall_list,
            'specificity': specificity_list,
            'dice': dice_list,
            'hausdorff': hausdorff_list,
            'time': time_list
        })

        df.to_excel(os.path.join("savefile", "Unet_SETA_camvid1.xlsx"), index=False)

        # 保存模型检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # 保存模型每5个epoch
        if np.mod(epoch + 1, 5) == 0:
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'Unet_SETA_{epoch + 1}.pth'))

# 开始训练
mytrain(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler, num_classes, start_epoch)
