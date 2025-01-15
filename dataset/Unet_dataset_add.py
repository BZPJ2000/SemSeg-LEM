import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, images_dir, masks_dir, num_classes=3, resize_shape=None, augment=False, expansion_factor=1):
        """
        初始化数据集
        :param images_dir: 图像文件夹路径
        :param masks_dir: 标签文件夹路径
        :param num_classes: 分类类别数量
        :param resize_shape: 图像和标签的尺寸 (H, W)
        :param augment: 是否进行数据增强
        :param expansion_factor: 数据扩充倍数
        """
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.num_classes = num_classes
        self.resize_shape = resize_shape
        self.augment = augment
        self.expansion_factor = expansion_factor  # 数据扩充倍数

        # 定义数据增强的变换
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 调整亮度、对比度等
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),  # 随机高斯模糊
        ])

    def __getitem__(self, i):
        index = i // self.expansion_factor  # 对扩展后的索引进行还原

        # 加载图像和标签
        image = Image.open(self.images_fps[index])
        mask = Image.open(self.masks_fps[index])

        # 如果指定了尺寸，则调整图像和标签的大小
        if self.resize_shape is not None:
            image = image.resize(self.resize_shape, Image.BILINEAR)
            mask = mask.resize(self.resize_shape, Image.NEAREST)

        # 数据增强
        if self.augment and random.random() > 0.5:
            image, mask = self._augment_data(image, mask)

        image = np.array(image)
        mask = np.array(mask)

        # 检查图像是灰度图还是RGB图
        if len(image.shape) == 2:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def __len__(self):
        # 返回扩充后的数据集长度
        return len(self.ids) * self.expansion_factor

    def _augment_data(self, image, mask):
        """
        数据增强操作，包括旋转、添加噪声等
        """
        # 使用 torchvision 的 transforms 进行图像增强
        image = self.transforms(image)

        # 手动添加黑点（噪声）
        mask = self._add_black_dots(mask)

        return image, mask

    def _add_black_dots(self, mask):
        """
        随机在标签中添加黑点（噪声）
        """
        np_mask = np.array(mask)
        height, width = np_mask.shape
        num_dots = random.randint(5, 15)  # 随机黑点数量
        for _ in range(num_dots):
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 1)
            np_mask[y, x] = 0  # 在随机位置添加黑点
        return Image.fromarray(np_mask)


if __name__ == '__main__':
    data_path_images = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages'
    data_path_labels = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedLabels'

    # 初始化数据集，扩充数据集并进行数据增强
    data = MyDataset(data_path_images, data_path_labels, num_classes=3, resize_shape=(224, 224), augment=True,
                     expansion_factor=10)

    # 获取第一个数据样本
    image, label = data[0]

    # 打印图像和标签的形状，以及标签的唯一值
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')
    print(f'Label unique values: {torch.unique(label)}')


    # 获取数据集大小
    dataset_size = len(data)
    # 打印数据集大小
    print(f'Dataset size: {dataset_size}')
