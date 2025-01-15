import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Define the transformation for filtered_images
transform_image = transforms.Compose([
    transforms.ToTensor(),
])

# Define the transformation for filtered_labels
transform_label = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, num_classes=3, resize_scale=1):
        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.filenames = os.listdir(labels_path)
        self.resize_scale = resize_scale

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        segment_name = self.filenames[index]
        segment_path = os.path.join(self.labels_path, segment_name)

        image_name = segment_name
        image_path = os.path.join(self.images_path, image_name)

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        new_size = (int(original_size[0] * self.resize_scale), int(original_size[1] * self.resize_scale))
        resize_transform = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)

        image = resize_transform(image)
        image = transform_image(image)

        segment_image = Image.open(segment_path).convert("L")
        segment_image = np.array(segment_image)

        segment_image = torch.tensor(segment_image, dtype=torch.long)
        segment_image[segment_image >= self.num_classes] = 0
        segment_image = resize_transform(segment_image.unsqueeze(0)).squeeze(0)

        return image, segment_image
        # return image, segment_image, image_name, original_size

if __name__ == '__main__':
    data_path_images = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\image'
    data_path_labels = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\processed_label'
    data = MyDataset(data_path_images, data_path_labels, num_classes=3)
    image, label, filename, original_size = data[0]
    print(f'Image shape: {image.shape}')
    print(f'Label shape: {label.shape}')
    print(f'Filename: {filename}')
    print(f'Original size: {original_size}')
