import os
from PIL import Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt

input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result2'


def print_pixel_distribution(image_array, filename):
    unique, counts = np.unique(image_array, return_counts=True)
    pixel_distribution = dict(zip(unique, counts))
    print(f'Pixel distribution for {filename}: {pixel_distribution}')


for filename in tqdm.tqdm(os.listdir(input_folder)):
    if filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_array = np.array(image)

        # 打印像素分布
        print_pixel_distribution(image_array, filename)

        # # 可视化像素分布
        # plt.figure()
        # plt.hist(image_array.ravel(), bins=256, color='orange', )
        # plt.hist(image_array.ravel(), bins=256, color='blue', alpha=0.5)
        # plt.title(f'Pixel Distribution for {filename}')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.show()

print('像素分布打印完成。')
