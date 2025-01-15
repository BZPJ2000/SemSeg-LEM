import os
from PIL import Image
import numpy as np

# 定义图片文件夹路径
images_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\Images"
output_images_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages"

# 创建输出文件夹
os.makedirs(output_images_folder, exist_ok=True)

# 转换 Images 文件夹中的 .tif 图片为 PNG 格式
def convert_images_to_png():
    for filename in os.listdir(images_folder):
        if filename.endswith(('.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  # 包含.tif格式
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path)

            # 将图片转换为 PNG 格式并保存
            new_filename = os.path.splitext(filename)[0] + ".png"
            image.save(os.path.join(output_images_folder, new_filename), 'PNG')
            print(f"转换 {filename} 为 {new_filename}")

if __name__ == "__main__":
    # 运行图片和标签转换
    convert_images_to_png()
