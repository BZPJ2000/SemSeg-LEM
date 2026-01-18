# SemSeg-LEM 文件夹整理说明

## 整理日期
2026-01-18

---

## 整理概述

本次对项目进行了全面的文件夹整理，将原本杂乱的根目录文件按功能分类到不同的子目录中，使项目结构更加清晰、易于维护。

---

## 新的目录结构

```
SemSeg-LEM/
├── config/                      # 配置管理模块
│   ├── __init__.py
│   ├── config.py               # 全局配置
│   └── paths.py                # 路径配置
│
├── models/                      # 新模型注册系统
│   ├── __init__.py
│   └── registry.py
│
├── model/                       # 原有模型架构（V1）
│   ├── UNet/
│   ├── UNet_ACC/
│   ├── UNet_BRAU/
│   ├── UNet_SETA/
│   ├── UNet_Swin/
│   ├── UNet_KAN/
│   └── ...
│
├── model_V2/                    # 模型架构（V2）
│   ├── UNet_DA_Trans/
│   └── UNet_KAN/
│
├── losses/                      # 损失函数库
│   ├── __init__.py
│   ├── dice_loss.py
│   ├── focal_loss.py
│   └── ...
│
├── dataset/                     # 数据集类
│   ├── dataset.py
│   ├── Unet_dataset_add.py
│   └── ...
│
├── data_processing/             # 数据预处理工具
│   ├── Format_conversion.py
│   └── Image_cropping.py
│
├── utils/                       # 工具函数
│   ├── utils.py
│   ├── fenbu.py
│   └── label_processing.py     # 原：标签处理.py
│
├── scripts/                     # 脚本文件（新整理）
│   ├── training/               # 训练脚本
│   │   ├── train_all.py
│   │   ├── train_ACC.py
│   │   ├── train_BRAU.py
│   │   ├── train_SETA.py
│   │   ├── train_Swin.py
│   │   ├── train_kan.py
│   │   └── ...（共15个训练脚本）
│   │
│   ├── prediction/             # 预测脚本
│   │   ├── predict.py
│   │   ├── predict_all.py
│   │   ├── predict_BRAU.py
│   │   └── ...（共6个预测脚本）
│   │
│   ├── image_processing/       # 图像处理脚本
│   │   ├── Image_computing_processing.py
│   │   ├── Image_computing_processing_V2.py
│   │   ├── Markers_all.py
│   │   ├── Markers_all_V2.py
│   │   ├── Markers_all_V3.py
│   │   ├── Salvage_all.py
│   │   ├── Salvage_all_V2.py
│   │   ├── Salvage_all_V3.py
│   │   └── Compositing_*.py
│   │
│   └── utils/                  # 工具脚本
│       ├── Check_PyTorch_V.py
│       ├── loss_api.py
│       └── fix_hardcoded_paths.py
│
├── gui/                         # 图形界面（已重命名）
│   ├── main_window.py          # 原：QT界面_V7.py（最新版本）
│   ├── marker.py               # 原：标记.py
│   ├── decompose.py            # 原：分解.py
│   ├── compose.py              # 原：合成.py
│   └── video_viewer.py         # 原：视频读取显示.py
│
├── outputs/                     # 输出目录（新整理）
│   ├── results/                # 结果输出
│   │   ├── result/
│   │   ├── result_A/
│   │   ├── result_B/
│   │   └── result_C/
│   ├── temp/                   # 临时文件
│   │   ├── temp_input/
│   │   └── temp_output/
│   ├── savefile/               # 模型保存和训练日志
│   └── output_parts/           # 输出部分
│
├── archive/                     # 归档目录（新建）
│   └── gui_old_versions/       # 旧版本GUI文件
│       ├── QT界面.py
│       ├── QT界面_V2.py
│       ├── QT界面_V3.py
│       ├── QT界面_V4.py
│       ├── QT界面_V5.py
│       └── QT界面_V6.py
│
├── testes/                      # 测试文件
│
├── main.py                      # 统一入口脚本
├── config.example.yaml          # 配置示例
├── requirements.txt             # 依赖列表
├── README.md                    # 项目说明
├── STRUCTURE.md                 # 项目结构文档
├── REFACTORING_PLAN.md          # 重构计划
├── OPTIMIZATION_SUMMARY.md      # 优化总结
└── FOLDER_ORGANIZATION.md       # 本文档
```

---

## 整理详情

### 1. 脚本文件整理

**训练脚本** (15个文件)
- 从根目录移动到 `scripts/training/`
- 包括：train_all.py, train_ACC.py, train_BRAU.py, train_SETA.py 等

**预测脚本** (6个文件)
- 从根目录移动到 `scripts/prediction/`
- 包括：predict.py, predict_all.py, predict_BRAU.py 等

**图像处理脚本** (12个文件)
- 从根目录移动到 `scripts/image_processing/`
- 包括：Markers_*.py, Salvage_*.py, Compositing_*.py 等

**工具脚本** (3个文件)
- 从根目录移动到 `scripts/utils/`
- 包括：Check_PyTorch_V.py, loss_api.py, fix_hardcoded_paths.py

### 2. 输出目录整理

**结果目录**
- `result/`, `result_A/`, `result_B/`, `result_C/` → `outputs/results/`

**临时目录**
- `temp_input/`, `temp_output/` → `outputs/temp/`

**模型保存**
- `savefile/` → `outputs/savefile/`
- `output_parts/` → `outputs/output_parts/`

### 3. 目录重命名

**标准化命名**
- `uilt/` → `utils/` (修正拼写错误)
- `datacommn/` → `data_processing/` (更清晰的命名)
- `Interface_display/` → `gui/` (更简洁的命名)

### 4. 文件重命名

**GUI文件**
- `QT界面_V7.py` → `main_window.py` (最新版本)
- `标记.py` → `marker.py`
- `分解.py` → `decompose.py`
- `合成.py` → `compose.py`
- `视频读取显示.py` → `video_viewer.py`

**工具文件**
- `标签处理.py` → `label_processing.py`

### 5. 旧版本归档

**GUI旧版本** (6个文件)
- 移动到 `archive/gui_old_versions/`
- 包括：QT界面.py ~ QT界面_V6.py

---

## 整理效果对比

### 整理前
- 根目录文件：50+ 个
- 训练脚本散落在根目录
- 预测脚本散落在根目录
- 输出目录混乱
- 中文文件名影响兼容性

### 整理后
- 根目录文件：约 15 个（仅保留核心文件）
- 脚本按功能分类到 scripts/ 子目录
- 输出统一管理在 outputs/ 目录
- 所有文件使用英文命名
- 旧版本文件归档到 archive/

---

## 使用指南

### 训练模型

```bash
# 使用新的统一入口
python main.py --mode train --model unet

# 或使用原有的训练脚本
python scripts/training/train_all.py
```

### 预测

```bash
# 使用新的统一入口
python main.py --mode predict --model unet --checkpoint path/to/model.pth

# 或使用原有的预测脚本
python scripts/prediction/predict_all.py
```

### 图像处理

```bash
# 标记处理
python scripts/image_processing/Markers_all_V3.py

# 图像合成
python scripts/image_processing/Compositing_original_image_all_V2.py
```

### GUI界面

```bash
# 启动图形界面
python gui/main_window.py
```

---

## 注意事项

### 1. 路径更新

由于文件位置发生了变化，如果您的脚本中有相对路径引用，可能需要更新：

```python
# 旧路径（从根目录引用）
from uilt.utils import *

# 新路径
from utils.utils import *
```

### 2. 向后兼容

所有原有的脚本仍然可以正常使用，只是位置发生了变化。建议：
- 使用新的统一入口 `main.py` 进行训练和预测
- 如需使用原有脚本，请使用新的路径

### 3. 旧版本文件

旧版本的GUI文件已归档到 `archive/gui_old_versions/`，如需使用可以从该目录找到。

---

## 整理统计

- **移动的文件数**: 36 个
- **重命名的文件数**: 7 个
- **重命名的目录数**: 3 个
- **新建的目录数**: 8 个
- **归档的文件数**: 6 个

---

## 总结

本次文件夹整理大幅提升了项目的可维护性和可读性：

1. **清晰的结构**: 文件按功能分类，易于查找
2. **标准化命名**: 所有文件和目录使用英文命名
3. **输出集中管理**: 所有输出统一在 outputs/ 目录
4. **版本控制**: 旧版本文件归档，保留历史
5. **易于扩展**: 清晰的结构便于添加新功能

---

**整理完成日期**: 2026-01-18
