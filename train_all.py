import os
import torch
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, random_split
import warnings
from dataset.Unet_dataset_add import MyDataset  # 请根据实际情况调整导入路径
import numpy as np
from torch import nn
import time
import multiprocessing
import matplotlib.pyplot as plt

from model.UNet_ACC.ACC_UNet_w import ACC_UNet_W

warnings.filterwarnings("ignore")
torch.manual_seed(17)

# 设置数据集路径
DATA_DIR = r'E:\A_workbench\A-lab\Unet\Unet_complit\data'
x_dir = os.path.join(DATA_DIR, 'ConvertedImages')  # 输入图像路径
y_dir = os.path.join(DATA_DIR, 'Labels')  # 标签路径
batch_size = 4  # 每个批次的样本数量
num_classes = 3  # 分类数量
learning_rate = 0.001  # 学习率
epochs_num = 10  # 训练的轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备：GPU 或 CPU

# 创建保存文件的目录
save_dir = "savefile"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果目录不存在，创建一个新的目录

checkpoints_dir = "checkpoints"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# 加载数据集
dataset = MyDataset(x_dir, y_dir, num_classes=3, resize_shape=(224, 224), augment=False, expansion_factor=10)

# 计算训练集和验证集大小
train_size = int(0.7 * len(dataset))  # 70% 用于训练
val_size = len(dataset) - train_size  # 30% 用于验证

# 将数据集打乱并重新分为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 定义一个函数统计数据集中每个标签的数量
def count_label_classes(data_loader, num_classes):
    label_counts = torch.zeros(num_classes)  # 初始化计数器
    for _, labels, *_ in data_loader:  # 遍历数据加载器
        for i in range(num_classes):
            label_counts[i] += torch.sum(labels == i).item()  # 统计每个类别的像素数量
    return label_counts

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 统计并打印训练集和验证集的标签类别分布
train_label_counts = count_label_classes(train_loader, num_classes)
val_label_counts = count_label_classes(val_loader, num_classes)

print(f"训练数据集标签计数: {train_label_counts.numpy()}")  # 打印训练集的标签分布
print(f"验证数据集标签计数: {val_label_counts.numpy()}")  # 打印验证集的标签分布

# 计算类别权重
class_weights = 1.0 / (train_label_counts + 1e-10)  # 计算每个类别的权重，避免除以零
class_weights = class_weights / class_weights.sum()  # 归一化权重
class_weights = class_weights.to(device)  # 将权重转移到 GPU

# 修改损失函数，加入权重
lossf = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)  # 使用加权的交叉熵损失函数

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

# 定义模型评估函数
def evaluate_model(model, data_loader, device, num_classes):
    model.eval()  # 将模型设置为评估模式
    iou_list = []
    total_loss = 0.0

    with torch.no_grad():  # 关闭梯度计算，减少内存占用
        for features, labels, *_ in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = lossf(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            # 计算每个 batch 的 IoU 并保存
            iou = compute_iou(predicted, labels, num_classes)
            iou_list.append(iou)

    avg_iou = torch.stack(iou_list).nanmean()  # 计算平均 IoU，忽略 NaN
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_iou, avg_loss

# 定义训练函数
def mytrain(net, train_iter, test_iter, loss, optimizer, num_epochs, scheduler, num_classes, model_name, device):
    loss_list = []
    train_iou_list = []
    test_iou_list = []
    epochs_list = []
    time_list = []

    for epoch in range(num_epochs):
        net.train()  # 将模型设置为训练模式
        train_loss = 0.0

        epoch_start_time = time.time()  # 开始计时

        for i, (features, labels, *_) in enumerate(train_iter):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(features)  # 前向传播
            loss_value = loss(outputs, labels)  # 计算损失
            loss_value.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss_value.item() * labels.size(0)

        train_loss_avg = train_loss / len(train_iter.dataset)  # 计算平均损失

        # 计算训练集上的 mIoU
        train_iou, _ = evaluate_model(net, train_iter, device, num_classes)

        scheduler.step()  # 更新学习率

        # 计算验证集上的 mIoU 和损失
        test_iou, test_loss_avg = evaluate_model(net, test_iter, device, num_classes)

        epoch_time = time.time() - epoch_start_time  # 记录当前 epoch 消耗的时间

        print(f"Model: {model_name}, Epoch {epoch + 1}/{num_epochs} --- Train Loss: {train_loss_avg:.3f} --- Train mIoU: {train_iou:.3f} --- Test mIoU: {test_iou:.3f} --- Test Loss: {test_loss_avg:.3f} --- Time: {epoch_time:.2f} sec")

        # 保存训练数据
        loss_list.append(train_loss_avg)
        train_iou_list.append(train_iou.item())
        test_iou_list.append(test_iou.item())
        epochs_list.append(epoch + 1)
        time_list.append(epoch_time)

    # 将训练信息保存为 Excel 文件
    df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': loss_list,
        'train_mIoU': train_iou_list,
        'test_mIoU': test_iou_list,
        'time': time_list
    })

    df.to_excel(os.path.join("savefile", f"{model_name}_training_data.xlsx"), index=False)

    # 保存模型
    torch.save(net.state_dict(), f'checkpoints/{model_name}_final.pth')

# 定义模型参数统计函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义训练模型的函数（用于多进程）
def train_model(model_class, model_args, model_name):
    # 设置设备（可以根据需要调整设备编号）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)  # 设置随机数种子

    # 在每个进程中重新创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 初始化模型并移动到设备
    model = model_class(**model_args).to(device)

    # 定义优化器和学习率调度器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    # 开始训练
    mytrain(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler, num_classes, model_name, device)

# 导入您的模型类（请根据实际情况调整导入路径）
from model.UNet_Swin.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from model.UNet_KAN.archs import UKAN
from model.UNet_BRAU.bra_unet import BRAUnet
from model.UNet_plus.Unet_plus import UnetPlusPlus
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention as UNet_SETA
from model.UNet_ACC.ACC_UNet import ACC_UNet
from model.UNet_ACC.ACC_UNet_lite import ACC_UNet_Lite
from model.UNet_ACC.ACC_UNet_w import ACC_UNet_W
from model.UNet_EGE.egeunet import EGEUNet
if __name__ == '__main__':
    # 定义模型列表
    models_to_train = [

        {'model_class': UKAN, 'model_args': {'num_classes': 3, 'input_channels': 3}, 'model_name': 'UKAN'},
        {'model_class': BRAUnet, 'model_args': {'img_size': 224, 'in_chans': 3, 'num_classes': 3, 'n_win': 7}, 'model_name': 'BRAUnet'},
        {'model_class': UnetPlusPlus, 'model_args': {'num_classes': 3}, 'model_name': 'UnetPlusPlus'},
        {'model_class': UNet_SETA, 'model_args': {'num_classes': 3}, 'model_name': 'UNet_SETA'},
        {'model_class': EGEUNet, 'model_args': {'input_channels': 3,'num_classes':3}, 'model_name': 'EGEUNet'},
        {'model_class': ACC_UNet, 'model_args': {'n_channels': 3,'n_classes':3}, 'model_name': 'ACC_UNet'},
        {'model_class': ACC_UNet_W, 'model_args': {'n_channels': 3,'n_classes':3}, 'model_name': 'ACC_UNet_W'},
        {'model_class': ACC_UNet_Lite, 'model_args': {'n_channels': 3,'n_classes':3}, 'model_name': 'ACC_UNet_Lite'},
        {'model_class': SwinTransformerSys, 'model_args': {'num_classes': 3, 'input_channels': 3},
         'model_name': 'SwinTransformerSys'},
    ]

    # 统计模型参数数量
    model_params = {}
    for model_info in models_to_train:
        model = model_info['model_class'](**model_info['model_args'])
        num_params = count_parameters(model)
        model_params[model_info['model_name']] = num_params

    # 启动多个进程进行训练
    processes = []
    for model_info in models_to_train:
        p = multiprocessing.Process(target=train_model, args=(model_info['model_class'], model_info['model_args'], model_info['model_name']))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 绘制模型参数数量对比图
    model_names = list(model_params.keys())
    params_counts = list(model_params.values())

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, params_counts)
    plt.xlabel('Model Name')
    plt.ylabel('Number of Parameters')
    plt.title('Model Parameter Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('savefile/model_parameters_comparison.png')
    plt.show()

    # 绘制损失过程对比图和 mIoU 曲线
    plt.figure(figsize=(10, 6))

    for model_info in models_to_train:
        model_name = model_info['model_name']
        df = pd.read_excel(os.path.join("savefile", f"{model_name}_training_data.xlsx"))
        plt.plot(df['epoch'], df['train_loss'], label=f'{model_name} Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('savefile/training_loss_comparison.png')
    plt.show()

    plt.figure(figsize=(10, 6))

    for model_info in models_to_train:
        model_name = model_info['model_name']
        df = pd.read_excel(os.path.join("savefile", f"{model_name}_training_data.xlsx"))
        plt.plot(df['epoch'], df['test_mIoU'], label=f'{model_name} Test mIoU')

    plt.xlabel('Epoch')
    plt.ylabel('Test mIoU')
    plt.title('Test mIoU Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('savefile/test_mIoU_comparison.png')
    plt.show()

    # 绘制耗时对比图
    model_times = {}
    for model_info in models_to_train:
        model_name = model_info['model_name']
        df = pd.read_excel(os.path.join("savefile", f"{model_name}_training_data.xlsx"))
        total_time = df['time'].sum()
        model_times[model_name] = total_time

    model_names = list(model_times.keys())
    times = list(model_times.values())

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, times)
    plt.xlabel('Model Name')
    plt.ylabel('Total Training Time (sec)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('savefile/model_training_time_comparison.png')
    plt.show()
