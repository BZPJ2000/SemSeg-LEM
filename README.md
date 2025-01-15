
# 🎨 语义分割与特定方向扩展项目

> 本项目的目标是实现语义分割模型的训练，并在分割结果基础上进行额外的特定方向扩展，包括长度计算、可视化与标记处理。

---

## 🌟 项目功能概述

1. **模型网络结构**  
   - 模型核心代码位于 `model` 和 `model_V2` 文件中。
   - 这些文件定义了语义分割网络结构。
   - 开头为train或者predict的文件中，为训练与预测逻辑的实现。
2. **语义分割处理流程**  
   - 支持多类别分割。
   - 对分割区域进行进一步的区域分块与长度计算处理。
   - 所有处理结果会进行标记并保存。
   - 具体处理流程：
     - 对分割区域的长度进行计算。
     - 在图片中标记结果并保存中间过程。
   - 处理结果按如下目录结构存储：
     ```
     result_A/
       ├── 分块数量1/
       ├── 分块数量2/
       └── 分块数量N/
     result_B/
       ├── 线段标记
     result_C/
       ├── 长度标记
     ```

3. **特定方向扩展**  
   - 本项目中 "特定方向" 指对分割结果的长度计算以及一系列可视化标记操作。
   - 生成的图片包含长度计算的标记，同时对每个分块进行单独处理与保存。

4. **额外图片处理功能**  
   - `datacommn` 文件夹中包含以下两项功能：
     - 图像格式转换。
     - 自定义裁剪功能。

5. **模型训练数据**  
   - 数据加载与预处理逻辑位于 `dataset` 文件夹中。

6. **可视化工具**  
   - 项目中使用了 **PyQt** 实现了用户友好的交互式可视化界面。
   - 可视化相关代码位于 `Interface_display` 文件夹中。

7. **特别说明**  
   - `SETA` 文件夹中的模型文件 **不会随项目提供**。

---

## 📂 项目结构

```plaintext
├── model/                     # 模型结构定义
├── model_V2/                  # 模型扩展版本结构
├── train.py                   # 模型训练主脚本
├── predict.py                 # 模型预测主脚本
├── dataset/                   # 数据加载与预处理
│   └── dataset.py             # 数据加载与预处理
│   └── dataset_V1.py          # 数据加载与预处理
│   └── Unet_dataset_add.py    # 数据加载与预处理
│   └── Unet_plus_dataset.py   # 数据加载与预处理
├── datacommn/                 # 图像处理模块
│   ├── format_converter.py    # 格式转换工具
│   └── image_cropper.py       # 自定义裁剪工具
├── Interface_display/         # PyQt 可视化界面
│   └── display.py             # 交互界面主文件
├── result_A/                  # 处理结果保存（包含分块数量的分类存储）
├── result_B/                  # 第二类结果存储
├── result_C/                  # 第三类结果存储
└── SETA/                      # 模型文件（不随项目提供）
```

---

## 🚀 快速开始

### 1️⃣ 环境配置

请确保您已经安装以下依赖：
- Python 3.8+
- PyTorch 1.10+
- torchvision
- PyQt5
- 其他依赖可通过以下命令安装：
  ```bash
  pip install -r requirements.txt
  ```

### 2️⃣ 模型训练

运行以下命令进行模型训练：
```bash
python train.py --epochs 50 --batch_size 16 --lr 0.001
```
- 也可以直接启动train开头的文件，多看几个，你也就知道怎么换模型和改参数了
### 3️⃣ 模型预测

运行以下命令进行语义分割结果预测：
```bash
python predict.py --input_dir ./data/test_images --output_dir ./result_A
```

### 4️⃣ 数据格式转换与裁剪

- 格式转换：
  ```bash
  python datacommn/format_converter.py --input_dir ./data --output_dir ./formatted_data
  ```
- 自定义裁剪：
  ```bash
  python datacommn/image_cropper.py --input_dir ./data --output_dir ./cropped_data
  ```

### 5️⃣ 可视化界面

运行以下命令启动 PyQt 可视化界面：
```bash
python Interface_display/display.py
```


## 💡 项目特点

- **模块化设计**：各功能模块分离，便于维护与扩展。
- **高可视化支持**：支持 PyQt 图形化界面操作，同时对分割结果生成清晰的标记与可视化文件。
- **灵活性**：支持多类别分割与自定义图像处理，新手比较容易介绍。
---

## 📌 注意事项

- **数据路径**：请确保输入数据和输出路径正确配置。
- **SETA 模型**：`SETA` 文件夹中的模型文件 **不会随项目提供**。
- **环境要求**：请使用支持 CUDA 的环境以加速模型训练与预测。

---

## 🤝 贡献指南

欢迎贡献代码或提出改进建议！请提交 Pull Request 或 Issue。

---

## 🙌 致谢

感谢 **Microsoft CSWin Transformer** 提供的底层算法支持！
