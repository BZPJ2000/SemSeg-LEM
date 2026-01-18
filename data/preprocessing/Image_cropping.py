import os
from PIL import Image
import numpy as np

# 文件夹路径
image_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\Images"
label_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\SegmentationClassPNG"
output_image_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\ResizedImages"
output_label_folder = r"E:\A_workbench\A-lab\Unet\Unet_complit\data\ResizedLabels"

# 确保输出文件夹存在
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# 裁剪区域边界
left = 1329  # 距离左边的长度
right = 779  # 距离右边的长度
top = 407  # 距离顶部的长度
bottom = 57  # 距离底部的长度

# 目标尺寸
target_size = (1024, 1024)

# 计算裁剪区域的大小
crop_width = 4908 - left - right
crop_height = 3264 - top - bottom
print(f"裁剪后的图片大小为: {crop_width}x{crop_height}")

# 计算缩放比例
width_ratio = target_size[0] / crop_width
height_ratio = target_size[1] / crop_height
print(f"宽度缩放比例: {width_ratio}, 高度缩放比例: {height_ratio}")


# 函数用于裁剪并缩放图片
def crop_and_resize_image(image_path, left, right, top, bottom, target_size, output_path):
    image = Image.open(image_path)
    width, height = image.size

    # 检查图片尺寸是否符合预期
    if width != 4908 or height != 3264:
        print(f"跳过图像 {image_path}, 因为尺寸不匹配: {width}x{height}")
        return

    # 计算裁剪区域
    left_boundary = left
    right_boundary = width - right
    top_boundary = top
    bottom_boundary = height - bottom

    # 裁剪图片
    cropped_image = image.crop((left_boundary, top_boundary, right_boundary, bottom_boundary))

    # 打印裁剪区域坐标
    print(f"裁剪区域坐标点: 左上角({left_boundary}, {top_boundary}), 右上角({right_boundary}, {top_boundary}), "
          f"左下角({left_boundary}, {bottom_boundary}), 右下角({right_boundary}, {bottom_boundary})")

    # 缩放图片
    resized_image = cropped_image.resize(target_size, Image.ANTIALIAS)

    # 保存图片
    resized_image.save(output_path)


# 函数用于裁剪并缩放标签
def crop_and_resize_label(label_path, left, right, top, bottom, target_size, output_path):
    label = Image.open(label_path)
    width, height = label.size

    # 检查标签尺寸是否符合预期
    if width != 4908 or height != 3264:
        print(f"跳过标签 {label_path}, 因为尺寸不匹配: {width}x{height}")
        return

    # 计算裁剪区域
    left_boundary = left
    right_boundary = width - right
    top_boundary = top
    bottom_boundary = height - bottom

    # 裁剪标签
    cropped_label = label.crop((left_boundary, top_boundary, right_boundary, bottom_boundary))

    # 缩放标签
    resized_label = cropped_label.resize(target_size, Image.NEAREST)

    # 检测标签中的类别
    label_array = np.array(resized_label)
    unique_classes = np.unique(label_array)
    print(f"File: {label_path}, Classes: {unique_classes}")

    # 保存标签
    resized_label.save(output_path)


# 处理所有图片和标签
for filename in os.listdir(image_folder):
    if filename.endswith('.tif'):
        image_path = os.path.join(image_folder, filename)
        output_image_path = os.path.join(output_image_folder, filename)
        crop_and_resize_image(image_path, left, right, top, bottom, target_size, output_image_path)

for filename in os.listdir(label_folder):
    if filename.endswith('.png'):
        label_path = os.path.join(label_folder, filename)
        output_label_path = os.path.join(output_label_folder, filename)
        crop_and_resize_label(label_path, left, right, top, bottom, target_size, output_label_path)

print("处理完成！")
