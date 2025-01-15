import os
import numpy as np
from PIL import Image

# 设置输入和输出文件夹的路径
input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'
output_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_C'

# 获取所有以 "marked_" 开头的 PNG 图片文件名
all_files = os.listdir(input_folder)
marked_files = [f for f in all_files if f.startswith('marked_') and f.endswith('.png')]

# 把图片按照编号分组，比如 "marked_00001_" 这一类
grouped_images = {}
for file_name in marked_files:
    # 从文件名中提取前缀，比如 "marked_00001_"
    prefix = '_'.join(file_name.split('_')[:3]) + '_'
    if prefix not in grouped_images:
        grouped_images[prefix] = []
    grouped_images[prefix].append(file_name)

# 处理每个分组的图片
for prefix, image_names in grouped_images.items():
    merged_image = None

    for name in image_names:
        image_path = os.path.join(input_folder, name)
        image = np.array(Image.open(image_path))

        # 如果这是第一张图片，初始化 merged_image
        if merged_image is None:
            merged_image = np.zeros_like(image)

        # 把当前图片的非零像素叠加到 merged_image 上
        merged_image = np.maximum(merged_image, image)

    # 黑化图片的顶部区域（比如前40个像素）
    blacken_height = 22
    merged_image[:blacken_height, :] = 0  # 将顶部40像素的区域设置为黑色

    # 保存合并后的图片
    output_filename = prefix + 'merged.png'
    output_path = os.path.join(output_folder, output_filename)
    Image.fromarray(merged_image).save(output_path)

    print(f'图片已保存到: {output_path}')
