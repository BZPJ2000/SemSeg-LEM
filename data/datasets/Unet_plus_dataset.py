import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, num_classes=3):
        # 获取图像文件夹中的所有文件名
        self.ids = os.listdir(images_dir)
        # 构建图像和标签的完整文件路径列表
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.num_classes = num_classes  # 设置分类的类别数量

    def __getitem__(self, i):
        # 加载图像
        image = Image.open(self.images_fps[i])
        image =image.resize((224, 224))
        # 加载标签
        mask = Image.open(self.masks_fps[i])

        # 将图像和标签转换为 NumPy 数组格式，方便后续处理
        image = np.array(image)
        mask = np.array(mask)

        # 检查图像是灰度图还是RGB图
        if len(image.shape) == 2:  # 如果图像是灰度图像 (H, W)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 加一个通道维度，变为 (1, H, W)
        else:  # 如果图像是 RGB 图像 (H, W, C)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 调整维度顺序，变为 (C, H, W)

        # 把标签也转换为 Tensor 格式
        mask = torch.tensor(mask, dtype=torch.long)

        # 没有做预先的处理就可以在这里处理
        # # 根据标签值创建新的三类分类标签
        # mask_new = torch.zeros_like(mask)  # 初始化一个全0的标签
        # mask_new[mask == 181] = 1  # 如果原标签是181，就把新标签设为1
        # mask_new[mask == 218] = 2  # 如果原标签是218，就把新标签设为2

        # 返回图像和新的标签
        return image, mask

    def __len__(self):
        # 返回数据集的总长度
        return len(self.ids)

if __name__ == '__main__':
    # 设置图像和标签的路径
    data_path_images = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'
    data_path_labels = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedLabels'
    # 初始化数据集
    data = MyDataset(data_path_images, data_path_labels, num_classes=3)
    # 获取第一个数据样本
    image, label = data[0]
    # 打印图像和标签的形状，以及标签的唯一值
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')
    print(f'Label unique values: {torch.unique(label)}')
