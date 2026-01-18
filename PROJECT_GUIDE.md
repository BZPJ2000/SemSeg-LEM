# SemSeg-LEM 项目指南

**版本**: 2.0
**更新日期**: 2026-01-18

---

## 目录

1. [项目概述](#项目概述)
2. [目录结构](#目录结构)
3. [快速开始](#快速开始)
4. [可用模型](#可用模型)
5. [配置说明](#配置说明)
6. [常见问题](#常见问题)
7. [更新日志](#更新日志)

---

## 项目概述

SemSeg-LEM 是一个语义分割深度学习框架，集成了多种先进的 UNet 变体模型。

**主要特性**:
- 10+ 种模型架构（UNet、注意力机制、Transformer、KAN等）
- 24 种损失函数
- 统一的配置管理系统
- 命令行和GUI双界面
- 完整的训练、预测、评估流程

**技术栈**:
- PyTorch
- Python 3.8+
- PyQt5 (GUI)

---

## 目录结构

```
SemSeg-LEM/
├── config/              # 配置管理
├── models/              # 模型架构
│   ├── v1/             # 原有模型
│   └── v2/             # V2版本模型
├── losses/              # 损失函数
├── data/                # 数据相关
│   ├── datasets/       # 数据集类
│   └── preprocessing/  # 预处理工具
├── utils/               # 工具函数
├── scripts/             # 脚本文件
│   ├── training/       # 训练脚本
│   ├── prediction/     # 预测脚本
│   ├── image_processing/  # 图像处理
│   └── utils/          # 工具脚本
├── gui/                 # 图形界面
├── outputs/             # 输出目录
│   ├── results/        # 结果
│   ├── temp/           # 临时文件
│   ├── savefile/       # 模型保存
│   └── output_parts/   # 输出部分
├── tests/               # 测试文件
├── archive/             # 归档文件
└── main.py              # 统一入口
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置项目

```bash
# 复制配置文件
cp config.example.yaml config.yaml

# 编辑配置文件，设置数据路径等参数
```

### 3. 训练模型

```bash
# 使用统一入口（推荐）
python main.py --mode train --model unet --config config.yaml

# 或使用原有脚本
python scripts/training/train_all.py

# 查看所有可用模型
python main.py --list-models
```

### 4. 预测

```bash
# 使用统一入口
python main.py --mode predict --model unet --checkpoint outputs/savefile/model.pth

# 或使用原有脚本
python scripts/prediction/predict_all.py
```

### 5. GUI界面

```bash
python gui/main_window.py
```

---

## 可用模型

### 基础 UNet
- `unet` - 标准 UNet
- `unet_plus` - UNet++

### 注意力机制 UNet
- `acc_unet` - ACC-UNet (通道和空间注意力)
- `brau_unet` - BRAU-UNet (双路径注意力)
- `seta_unet` - SETA-UNet (自适应注意力)
- `ege_unet` - EGE-UNet

### Transformer UNet
- `swin_unet` - Swin Transformer UNet
- `da_trans_unet` - DA-Transformer UNet

### KAN UNet
- `kan_unet` - Kolmogorov-Arnold Networks UNet

---

## 配置说明

### 配置文件示例

```yaml
training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.001
  device: cuda

model:
  name: unet
  num_classes: 2
  in_channels: 3

data:
  image_size: [256, 256]
  augmentation: true

loss:
  name: dice
```

### 路径配置

项目使用自动路径检测，无需硬编码路径：

```python
from config.paths import paths

# 自动检测项目根目录
data_dir = paths.data_root
checkpoint = paths.get_checkpoint_path('unet', epoch=50)
```

---

## 常见问题

**Q: 如何切换不同的模型？**
```bash
python main.py --mode train --model acc_unet
```

**Q: 如何修改训练参数？**
- 方法1: 编辑 config.yaml 文件
- 方法2: 使用命令行参数 `--epochs 200 --batch-size 8 --lr 0.0001`

**Q: 数据应该放在哪里？**
默认数据目录为 `data_root/`，可通过配置文件修改

**Q: 旧的训练脚本还能用吗？**
可以，所有脚本已移至 `scripts/` 目录，仍可正常使用

---

## 更新日志

### Version 2.0 (2026-01-18)

**新增功能**:
- 统一的配置管理系统
- 模型注册系统
- 统一入口脚本 main.py
- 完整的损失函数导出

**目录优化**:
- 整合模型目录：model/ + model_V2/ → models/v1 + models/v2
- 整合数据目录：dataset/ + data_processing/ → data/datasets + data/preprocessing
- 脚本分类：训练、预测、图像处理脚本分类到 scripts/
- 输出统一：所有输出集中到 outputs/
- 标准化命名：所有目录和文件使用英文命名

**修复问题**:
- 修复 losses 模块导入问题
- 提供硬编码路径解决方案

---

**项目地址**: https://github.com/your-repo/SemSeg-LEM
**文档更新**: 2026-01-18
