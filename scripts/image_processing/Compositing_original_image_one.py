import os
import numpy as np
from PIL import Image

# 设置路径
input_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\output_marked'
output_folder = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\merged_images'
output_filename = 'marked_00001_merged.png'

# 获取要合并的图片名称
image_names = [
    'marked_00001_region_1_1_sub_1_part_1.png',
    'marked_00001_region_1_2_sub_1_part_1.png',
    'marked_00001_region_1_3_sub_1_part_1.png',
    'marked_00001_region_1_4_sub_1_part_1.png',
    'marked_00001_region_2_1_sub_1_part_1.png'
]

# 初始化一个空的numpy数组来存放合并的结果
merged_image = None

for name in image_names:
    image_path = os.path.join(input_folder, name)
    image = np.array(Image.open(image_path))

    # 初始化merged_image
    if merged_image is None:
        merged_image = np.zeros_like(image)

    # 将当前图片的非零像素叠加到merged_image
    merged_image = np.maximum(merged_image, image)

# 保存合并后的图片
output_path = os.path.join(output_folder, output_filename)
Image.fromarray(merged_image).save(output_path)

print(f'图片已保存到: {output_path}')
