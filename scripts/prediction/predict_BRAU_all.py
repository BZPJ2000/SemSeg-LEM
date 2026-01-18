import os
import torch
from PIL import Image
import numpy as np
from dataset.Unet_plus_dataset import MyDataset
from model.UNet_BRAU.bra_unet import BRAUnet
import torchvision.transforms as transforms

# 首先，我们要确定使用的设备是GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 设置路径
model_path = r'checkpoints/Unet_BRAU_145.pth'  # 这是你训练好的模型文件路径
input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'  # 输入文件夹路径
output_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\predictions'  # 输出文件夹路径

# 如果输出文件夹不存在，则创建它
os.makedirs(output_folder, exist_ok=True)

# 加载模型
num_classes = 3  # 三分类问题
model = BRAUnet(img_size=224, in_chans=3, num_classes=num_classes, n_win=7).cuda(0)
model.load_state_dict(torch.load(model_path))  # 加载模型的权重参数
model.eval()  # 设置为评估模式

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 只处理特定的图像文件格式
        image_path = os.path.join(input_folder, filename)  # 获取当前图像的完整路径
        output_path = os.path.join(output_folder, filename)  # 输出文件路径

        # 加载图像
        image = Image.open(image_path)
        image = image.resize((224, 224))  # 调整大小为224x224
        image_np = np.array(image)

        if len(image_np.shape) == 2:  # 如果是灰度图
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:  # 如果是RGB图像
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        image_tensor = image_tensor.to(device)  # 移动到设备

        # 预测
        with torch.no_grad():
            output = model(image_tensor)  # 用模型进行预测
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()  # 转为NumPy格式

        # 将预测结果转换为图像
        predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))  # 映射到0, 127, 254

        # 保存预测结果
        predicted_img.save(output_path)

        print(f"Prediction saved to {output_path}")  # 打印保存路径

print("All predictions are complete.")
