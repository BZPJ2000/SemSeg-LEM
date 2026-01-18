# SemSeg-LEM 项目结构说明文档

## 项目概述

SemSeg-LEM 是一个语义分割深度学习框架，集成了多种先进的 UNet 变体模型，包括注意力机制、Transformer 和 KAN 架构。

**版本**: 2.0 (重构版本)
**框架**: PyTorch
**Python版本**: 3.8+

---

## 目录结构

### 核心模块

```
SemSeg-LEM/
├── config/                      # 配置管理模块 [新增]
│   ├── __init__.py
│   ├── config.py               # 全局配置管理
│   └── paths.py                # 路径配置管理
│
├── models/                      # 模型架构模块 [新增]
│   ├── __init__.py
│   └── registry.py             # 模型注册系统
│
├── losses/                      # 损失函数模块 [已优化]
│   ├── __init__.py             # 完整的损失函数导出
│   ├── dice_loss.py            # Dice系列损失
│   ├── focal_loss.py           # Focal Loss
│   ├── boundary_loss.py        # 边界损失
│   ├── lovasz_loss.py          # Lovasz Loss
│   ├── hausdorff.py            # Hausdorff距离损失
│   └── ND_Crossentropy.py      # N维交叉熵
│
├── main.py                      # 统一入口脚本 [新增]
├── config.example.yaml          # 配置文件示例 [新增]
├── REFACTORING_PLAN.md          # 重构计划文档 [新增]
└── STRUCTURE.md                 # 本文档 [新增]
```

---

## 新增功能

### 1. 配置管理系统

**位置**: `config/`

**功能**:
- 统一的配置管理，支持 YAML 文件
- 自动路径检测，消除硬编码路径
- 支持命令行参数覆盖

**使用方法**:
```python
from config import Config, PathConfig

# 加载配置
config = Config('config.yaml')

# 获取配置值
batch_size = config.get('training.batch_size')

# 使用路径配置
paths = PathConfig()
data_dir = paths.data_root
checkpoint_path = paths.get_checkpoint_path('unet', epoch=50)
```

### 2. 模型注册系统

**位置**: `models/registry.py`

**功能**:
- 统一的模型管理和加载接口
- 支持动态模型注册
- 简化模型实例化过程

**使用方法**:
```python
from models import get_model, list_models

# 列出所有可用模型
available_models = list_models()

# 加载模型
model = get_model('unet', num_classes=2, in_channels=3)
```

### 3. 统一入口脚本

**位置**: `main.py`

**功能**:
- 统一的命令行接口
- 支持训练、预测、评估、可视化等模式
- 自动模型选择和配置管理

**使用方法**:
```bash
# 列出所有可用模型
python main.py --list-models

# 训练模型
python main.py --mode train --model unet --config config.yaml

# 预测
python main.py --mode predict --model unet --checkpoint path/to/checkpoint.pth

# 评估
python main.py --mode evaluate --model unet --checkpoint path/to/checkpoint.pth
```

---

## 现有模块说明

### 模型架构 (model/ 和 model_V2/)

**基础 UNet 系列**:
- `model/UNet/Unet.py` - 标准 UNet
- `model/UNet_plus/Unet_plus.py` - UNet++

**注意力机制 UNet**:
- `model/UNet_ACC/ACC_UNet.py` - 通道和空间注意力
- `model/UNet_BRAU/bra_unet.py` - 双路径注意力
- `model/UNet_SETA/Unet_ASE_V3.py` - 自适应注意力
- `model/UNet_EGE/EGE_UNet.py` - EGE-UNet

**Transformer 系列**:
- `model/UNet_Swin/` - Swin Transformer UNet
- `model_V2/UNet_DA_Trans/DATransUNet.py` - DA-Transformer

**KAN 架构**:
- `model/UNet_KAN/archs.py` - Kolmogorov-Arnold Networks
- `model_V2/UNet_KAN/archs.py` - KAN V2 版本

### 数据处理 (dataset/)

- `dataset/dataset.py` - 基础数据集类
- `dataset/Unet_dataset_add.py` - 带数据增强的数据集
- `dataset/Unet_plus_dataset.py` - UNet++ 专用数据集

### 损失函数 (losses/)

**Dice 系列**:
- `SoftDiceLoss` - 软 Dice 损失
- `GDiceLoss` - 广义 Dice 损失
- `TverskyLoss` - Tversky 损失
- `IoULoss` - IoU 损失

**其他损失**:
- `FocalLoss` - Focal 损失
- `LovaszSoftmax` - Lovasz 损失
- `BDLoss` - 边界损失
- `HausdorffDTLoss` - Hausdorff 距离损失

### 工具函数 (uilt/)

- `uilt/utils.py` - 通用工具函数
- `uilt/fenbu.py` - 分布相关函数
- `uilt/标签处理.py` - 标签处理工具

### 图形界面 (Interface_display/)

- 多版本 PyQt5 界面 (QT界面.py ~ QT界面_V7.py)
- 图像处理工具 (标记.py, 分解.py, 合成.py)
- 视频处理 (视频读取显示.py)

---

## 训练和预测脚本

### 训练脚本

- `train_all.py` - 通用训练脚本
- `train_ACC.py` - ACC-UNet 训练
- `train_BRAU.py` - BRAU-UNet 训练
- `train_SETA.py` - SETA-UNet 训练
- `train_Swin.py` - Swin Transformer 训练
- `train_kan.py` - KAN-UNet 训练
- `train_Plus.py` - UNet++ 训练

### 预测脚本

- `predict.py` - 基础预测脚本
- `predict_all.py` - 批量预测
- `predict_BRAU.py` - BRAU-UNet 预测
- `predict_seta.py` - SETA-UNet 预测

---

## 已识别的问题和解决方案

### 1. 硬编码路径问题 ✅ 已提供解决方案

**问题**: 48个文件包含硬编码的绝对路径 `E:\A_workbench\...`

**解决方案**:
- 使用新创建的 `config.PathConfig` 类
- 所有路径通过配置管理，支持自动检测项目根目录
- 参考 `config/paths.py` 的实现

**迁移示例**:
```python
# 旧代码
DATA_DIR = r'E:\A_workbench\A-lab\Unet\Unet_complit\data'

# 新代码
from config.paths import paths
DATA_DIR = paths.data_root
```

### 2. 损失函数模块问题 ✅ 已修复

**问题**: `losses/__init__.py` 文件为空，无法正确导入

**解决方案**: 已完善 `__init__.py`，导出所有24个损失函数类

### 3. 多版本脚本混乱 ⚠️ 需要整合

**问题**: 存在大量重复的版本文件
- QT界面: 7个版本
- Salvage脚本: 3个版本
- Markers脚本: 3个版本
- Image_computing_processing: 2个版本

**建议**:
- 保留最新版本
- 将旧版本移至 `archive/` 目录
- 提取公共功能到工具模块

### 4. 中文文件名 ⚠️ 建议重命名

**问题**: 部分文件使用中文命名，影响跨平台兼容性

**建议重命名**:
- `QT界面*.py` → `main_window*.py`
- `标记.py` → `marker.py`
- `分解.py` → `decompose.py`
- `合成.py` → `compose.py`
- `视频读取显示.py` → `video_viewer.py`
- `标签处理.py` → `label_processing.py`

---

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp config.example.yaml config.yaml

# 编辑配置文件，设置数据路径等参数
```

### 2. 准备数据

将数据放置在以下结构中：
```
data_root/
├── images/          # 原始图像
├── labels/          # 标签图像
└── splits/          # 训练/验证/测试划分
```

### 3. 训练模型

```bash
# 使用默认配置训练
python main.py --mode train --model unet

# 使用自定义配置
python main.py --mode train --model acc_unet --config config.yaml --epochs 200

# 查看所有可用模型
python main.py --list-models
```

### 4. 预测

```bash
# 单张图像预测
python main.py --mode predict --model unet --checkpoint outputs/checkpoints/unet_best.pth

# 批量预测
python main.py --mode predict --model unet --checkpoint outputs/checkpoints/unet_best.pth --data-dir data_root/test
```

---

## 可用模型列表

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

## 后续优化建议

### 短期优化 (1-2周)

1. **路径迁移**: 将现有训练/预测脚本中的硬编码路径替换为 `PathConfig`
2. **版本整合**: 合并多版本脚本，保留最新版本
3. **文件重命名**: 将中文文件名改为英文

### 中期优化 (2-4周)

1. **模型迁移**: 将所有模型注册到 `models/registry.py`
2. **统一训练器**: 创建 `core/trainer.py`，统一训练流程
3. **统一预测器**: 创建 `core/predictor.py`，统一预测流程
4. **数据模块重构**: 整合数据集类到 `data/datasets.py`

### 长期优化 (1-2个月)

1. **完整测试**: 添加单元测试和集成测试
2. **文档完善**: 添加 API 文档和详细使用指南
3. **性能优化**: 优化内存使用和训练速度
4. **GUI 重构**: 整合多版本界面，创建统一的 GUI 应用

---

## 项目统计

- **总文件数**: 122个 Python 文件
- **模型架构**: 40+ 种
- **损失函数**: 24 种
- **训练脚本**: 15 个
- **预测脚本**: 6 个

---

## 贡献指南

### 添加新模型

1. 在 `model/` 或 `models/` 目录下创建模型文件
2. 使用 `@register_model` 装饰器注册模型
3. 更新本文档的模型列表

### 添加新损失函数

1. 在 `losses/` 目录下创建损失函数文件
2. 在 `losses/__init__.py` 中导出
3. 更新文档

---

## 常见问题

### Q: 如何切换不同的模型？

使用 `--model` 参数指定模型名称：
```bash
python main.py --mode train --model acc_unet
```

### Q: 如何修改训练参数？

方法1: 编辑 `config.yaml` 文件
方法2: 使用命令行参数覆盖：
```bash
python main.py --mode train --model unet --epochs 200 --batch-size 8 --lr 0.0001
```

### Q: 数据应该放在哪里？

默认数据目录为 `data_root/`，可以通过配置文件或命令行参数修改。

### Q: 如何使用旧的训练脚本？

旧的训练脚本（如 `train_ACC.py`）仍然可用，但建议使用新的统一入口 `main.py`。

---

## 更新日志

### Version 2.0 (当前版本)

**新增**:
- 配置管理系统 (`config/`)
- 模型注册系统 (`models/registry.py`)
- 统一入口脚本 (`main.py`)
- 完整的损失函数导出

**修复**:
- 修复 `losses/__init__.py` 空文件问题
- 提供硬编码路径解决方案

**文档**:
- 项目结构说明文档 (`STRUCTURE.md`)
- 重构计划文档 (`REFACTORING_PLAN.md`)
- 配置文件示例 (`config.example.yaml`)

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发起 Pull Request

---

**最后更新**: 2026-01-18
