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
image_path = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages\00022.png'  # 需要进行预测的图像文件路径
output_path = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\predicted.png'  # 保存预测结果的文件路径

# 加载模型
num_classes = 3  # 因为这是个三分类问题，所以我们设置类数为3
model = BRAUnet(img_size=224,in_chans=3, num_classes=num_classes, n_win=7).cuda(0)
model.load_state_dict(torch.load(model_path))  # 加载模型的权重参数
model.eval()  # 将模型设置为评估模式，这样在预测时会关闭一些不需要的操作（比如dropout）

# 加载图像
image = Image.open(image_path)  # 打开需要预测的图像
# 调整图像大小为224x224
image = image.resize((224, 224))
# 转换图像为模型输入格式
image_np = np.array(image)  # 先把图像转换成NumPy数组
if len(image_np.shape) == 2:  # 如果图像是灰度图
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加两个维度，形状变为 (1, 1, H, W)
else:  # 如果是RGB图像
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 调整通道顺序，然后再添加一个维度，变成 (1, C, H, W)

image_tensor = image_tensor.to(device)  # 把处理好的图像移动到指定设备

# 预测
with torch.no_grad():  # 在这个块内，我们不计算梯度，减少内存占用
    output = model(image_tensor)  # 用模型进行预测
    _, predicted = torch.max(output, 1)  # 获取每个像素预测的类别
    predicted = predicted.squeeze().cpu().numpy()  # 把结果转回NumPy数组，并去掉多余的维度

# 将预测结果转换为图像
predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))  # 乘以127是为了把类别0, 1, 2分别映射到0, 127, 254，这样更容易区分

# 保存预测结果
predicted_img.save(output_path)  # 把结果图像保存到指定的文件路径

print(f"Prediction saved to {output_path}")  # 打印出保存路径，方便查看结果
