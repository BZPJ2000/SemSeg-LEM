# SemSeg-LEM 项目重构计划

## 概述

本文档记录了 SemSeg-LEM 项目的系统性重构计划，旨在解决代码组织、路径管理、版本混乱等问题。

## 已完成的工作

### 1. 配置管理系统 ✅
- 创建 `config/` 目录
- 实现 `config/config.py` - 全局配置管理
- 实现 `config/paths.py` - 路径配置管理
- 支持 YAML 配置文件加载

### 2. 模型注册系统 ✅
- 创建 `models/` 目录结构
- 实现 `models/registry.py` - 模型注册器
- 支持动态模型加载

### 3. 统一入口脚本 ✅
- 创建 `main.py` - 统一的命令行入口
- 支持模型选择、训练、预测、评估、可视化等模式

### 4. 损失函数模块修复 ✅
- 修复 `losses/__init__.py`
- 正确导出所有损失函数类

## 待完成的工作

### 5. 硬编码路径修复 (进行中)

**问题**: 48个文件包含硬编码的绝对路径 `E:\A_workbench\...`

**受影响的文件类别**:
- 训练脚本: 15个文件
- 预测脚本: 6个文件
- 图像处理脚本: 12个文件
- 数据集文件: 4个文件
- 工具文件: 3个文件
- 界面文件: 5个文件
- 测试文件: 2个文件
- 其他: 1个文件

**解决方案**:
1. 使用 `config.PathConfig` 替代所有硬编码路径
2. 创建示例配置文件 `config.example.yaml`
3. 更新所有受影响文件的导入和路径引用

### 6. 多版本脚本整合

**问题**: 存在大量重复的版本文件

**需要整合的文件**:
- QT界面: 7个版本 (QT界面.py ~ QT界面_V7.py)
- Salvage脚本: 3个版本
- Markers脚本: 3个版本
- Image_computing_processing: 2个版本
- Compositing脚本: 2个版本

**解决方案**:
1. 保留最新版本作为主版本
2. 将旧版本移动到 `archive/` 目录
3. 提取公共功能到工具模块

### 7. 文件夹结构重组

**新结构**:
```
SemSeg-LEM/
├── config/              ✅ 已创建
├── models/              ✅ 已创建
├── data/                ⏳ 待创建
├── losses/              ✅ 已存在
├── utils/               ⏳ 待创建
├── core/                ⏳ 待创建
├── scripts/             ⏳ 待创建
├── gui/                 ⏳ 待创建
├── tests/               ⏳ 待创建
├── outputs/             ⏳ 待创建
├── archive/             ⏳ 待创建 (存放旧版本)
└── main.py              ✅ 已创建
```

### 8. 导入路径更新

**需要更新的导入模式**:
- `from model.UNet.Unet import UNet` → `from models.unet import UNet`
- `from dataset.dataset import MyDataset` → `from data.datasets import MyDataset`
- `from uilt.utils import *` → `from utils.common import *`

### 9. 中文文件名重命名

**需要重命名的文件**:
- `Interface_display/QT界面*.py` → `gui/main_window*.py`
- `Interface_display/标记.py` → `gui/marker.py`
- `Interface_display/分解.py` → `gui/decompose.py`
- `Interface_display/合成.py` → `gui/compose.py`
- `Interface_display/视频读取显示.py` → `gui/video_viewer.py`
- `uilt/标签处理.py` → `utils/label_processing.py`
- `model/查看模型结构.py` → `scripts/model_viewer.py`

### 10. 核心功能模块创建

**需要创建的模块**:
- `core/trainer.py` - 统一训练器
- `core/predictor.py` - 统一预测器
- `core/evaluator.py` - 评估器
- `data/datasets.py` - 统一数据集类
- `data/transforms.py` - 数据增强
- `utils/common.py` - 通用工具函数
- `utils/visualization.py` - 可视化工具

## 实施步骤

### 阶段1: 基础设施 (已完成)
1. ✅ 创建配置管理系统
2. ✅ 创建模型注册系统
3. ✅ 创建统一入口脚本
4. ✅ 修复损失函数模块

### 阶段2: 路径修复 (进行中)
1. ⏳ 创建示例配置文件
2. ⏳ 修复关键训练脚本的路径
3. ⏳ 修复预测脚本的路径
4. ⏳ 修复数据集文件的路径

### 阶段3: 文件重组
1. ⏳ 创建新的目录结构
2. ⏳ 移动文件到新位置
3. ⏳ 重命名中文文件名
4. ⏳ 归档旧版本文件

### 阶段4: 导入更新
1. ⏳ 更新所有文件的导入语句
2. ⏳ 测试导入是否正常工作

### 阶段5: 核心模块创建
1. ⏳ 创建统一的训练器
2. ⏳ 创建统一的预测器
3. ⏳ 创建数据处理模块

### 阶段6: 文档和测试
1. ⏳ 创建项目结构说明文档
2. ⏳ 创建使用指南
3. ⏳ 添加基本测试

## 注意事项

1. **备份**: 在进行大规模重构前，确保代码已提交到 Git
2. **渐进式**: 逐步进行重构，每次修改后测试
3. **兼容性**: 保留旧版本文件在 archive/ 目录中
4. **文档**: 及时更新文档，记录所有变更

## 预期收益

1. **可维护性**: 清晰的目录结构，易于理解和维护
2. **可移植性**: 消除硬编码路径，可在任何环境运行
3. **可扩展性**: 模块化设计，易于添加新功能
4. **可读性**: 统一的代码风格和命名规范
5. **易用性**: 统一的命令行接口，简化使用流程
