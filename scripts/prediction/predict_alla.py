import os
import torch
from PIL import Image
import numpy as np
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention

# 确定使用的设备（GPU或CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置路径
model_path = r'checkpoints/Unet_SETA_108150.pth'  # 这是你训练好的模型文件路径
input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'  # 需要进行预测的图像文件夹路径
output_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result2'  # 保存预测结果的文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载模型
num_classes = 3  # 类别数为3
model = UNetWithAttention(num_classes=num_classes).to(device)  # 初始化模型并移动到指定设备
model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型的权重参数
model.eval()  # 将模型设置为评估模式

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_folder, filename)  # 构建输入图像的完整路径
        output_path = os.path.join(output_folder, filename)  # 构建输出图像的保存路径

        # 加载图像
        image = Image.open(image_path)

        # 转换图像为模型输入格式
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # 如果图像是灰度图
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:  # 如果是RGB图像
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        image_tensor = image_tensor.to(device)  # 将图像张量移动到指定设备

        # 预测
        with torch.no_grad():  # 在不计算梯度的上下文中进行预测
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()

        # 将预测结果转换为图像
        predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))  # 根据类别映射到灰度值

        # 保存预测结果
        predicted_img.save(output_path)

        print(f"已处理 {filename}，预测结果保存至 {output_path}")

print("所有图像均已处理完毕，预测结果已保存。")
