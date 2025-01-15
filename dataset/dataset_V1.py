import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, num_classes=3, resize_shape=None):
        """
        初始化数据集
        :param images_dir: 图像文件夹路径
        :param masks_dir: 标签文件夹路径
        :param num_classes: 分类类别数量
        :param resize_shape: 图像和标签的尺寸 (H, W)
        """
        # 获取图像文件夹中的所有文件名
        self.ids = os.listdir(images_dir)
        # 构建图像和标签的完整文件路径列表
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.num_classes = num_classes  # 设置分类的类别数量
        self.resize_shape = resize_shape  # 设置图像和标签的尺寸，如果为 None 则不调整大小

    def __getitem__(self, i):
        # 加载图像
        image = Image.open(self.images_fps[i])
        # 加载标签
        mask = Image.open(self.masks_fps[i])

        # 如果指定了尺寸，则调整图像和标签的大小
        if self.resize_shape is not None:
            image = image.resize(self.resize_shape, Image.BILINEAR)  # 双线性插值调整图像大小
            mask = mask.resize(self.resize_shape, Image.NEAREST)  # 使用最近邻插值调整标签大小

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

        # 返回图像和标签
        return image, mask

    def __len__(self):
        # 返回数据集的总长度
        return len(self.ids)

if __name__ == '__main__':
    # 设置图像和标签的路径
    data_path_images = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'
    data_path_labels = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedLabels'
    # 初始化数据集，指定图像和标签的尺寸
    data = MyDataset(data_path_images, data_path_labels, num_classes=3, resize_shape=(224, 224))
    # 获取第一个数据样本
    image, label = data[0]
    # 打印图像和标签的形状，以及标签的唯一值
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')
    print(f'Label unique values: {torch.unique(label)}')
