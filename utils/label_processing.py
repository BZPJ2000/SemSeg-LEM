import os
from PIL import Image
import numpy as np
from collections import Counter

# 过滤类函数：只保留像素最多的三个类，其他的替换为0
def filter_classes(label, top_k=3):
    label_np = np.array(label)  # 将标签转换为 NumPy 数组
    original_shape = label_np.shape  # 获取标签的形状
    label_np = label_np.flatten()  # 展平数组
    counter = Counter(label_np)
    most_common = counter.most_common(top_k)
    top_classes = [cls for cls, _ in most_common]

    filtered_label = np.where(np.isin(label_np, top_classes), label_np, 0)
    return filtered_label.reshape(original_shape)  # 重新调整为原来的形状

# 输入和输出路径
input_label_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\label'
output_label_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\processed_label'
os.makedirs(output_label_folder, exist_ok=True)

# 处理每个标签文件
for filename in os.listdir(input_label_folder):
    if filename.endswith(".png"):
        label_path = os.path.join(input_label_folder, filename)
        label_image = Image.open(label_path).convert("L")  # 打开并转换为灰度图像
        filtered_label = filter_classes(label_image)  # 过滤类

        # 将处理后的标签保存为新图像文件
        output_path = os.path.join(output_label_folder, filename)
        Image.fromarray(filtered_label.astype(np.uint8)).save(output_path)

print("处理完成，所有标签已保存到:", output_label_folder)
