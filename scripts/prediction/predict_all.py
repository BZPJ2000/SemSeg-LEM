import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
from dataset.Unet_plus_dataset import MyDataset
from model.UNet_BRAU.bra_unet import BRAUnet
import tqdm

# 我们先来做一些基本的设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 看看有没有GPU，没有的话就用CPU
num_classes = 3  # 这里是三分类问题，所以类数设置为3
net = MyDataset(num_classes=num_classes).to(device)  # 初始化我们的Unet++模型，并放到设备上

# 接下来我们要加载已经训练好的模型参数
weight_path = r'E:\A_workbench\A-lab\Unet\Unet_complit\checkpoints\Unet_BRAU_145.pth'
net.load_state_dict(torch.load(weight_path))  # 读取模型的权重参数
net.eval()  # 设置模型为评估模式，这样可以关闭dropout等影响推理的操作

# 定义一些数据路径
input_image_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'  # 这是我们的输入图像文件夹
result_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result2'  # 预测结果保存的位置
os.makedirs(result_dir, exist_ok=True)  # 如果结果保存文件夹不存在，那就创建一个

# 我们现在要准备数据集和数据加载器
dataset = MyDataset(input_image_folder, input_image_folder, num_classes=num_classes)  # 初始化数据集，这里只用了输入图像路径
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 使用DataLoader按批次加载数据，这里每次加载一张图

# 下面就是推理过程了
with torch.no_grad():  # 不需要计算梯度，减少内存消耗
    for idx, (image, _) in enumerate(tqdm.tqdm(data_loader)):  # tqdm用来显示进度条
        image = image.to(device)  # 把图像数据转移到设备上（GPU或者CPU）
        out_image = net(image)  # 通过模型得到输出
        preds = torch.argmax(out_image, dim=1).cpu().numpy().squeeze()  # 找到每个像素所属的类别，然后转成numpy数组

        # 我们需要从文件路径中提取图像的名称
        image_name = os.path.basename(dataset.images_fps[idx]).split('.')[0]  # 取出文件名，不包括扩展名

        # 把预测结果转换成图像格式，为了可视化，我们把类别值乘以127
        predicted_img = Image.fromarray((preds * 127).astype(np.uint8))

        # 保存预测结果到我们指定的文件夹
        output_path = os.path.join(result_dir, image_name + '_pred.png')
        predicted_img.save(output_path)
        print(f"Prediction saved to {output_path}")  # 打印出保存的文件路径，方便查看

print('推理完成，灰度结果已保存。')  # 推理完成后给出提示


