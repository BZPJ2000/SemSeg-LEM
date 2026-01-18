import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体支持

# 扩展黑色区域的函数
def extend_black_regions(image, extension_pixels=20, threshold=127):
    # 将图像二值化
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_regions_found = False

    # 遍历每一个白色区域的轮廓
    for contour in contours:
        # 创建白色区域的掩码
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 仅在白色区域内寻找黑色区域
        white_region = cv2.bitwise_and(binary, mask)

        # 反转白色区域的掩码，寻找白色区域内的黑色区域
        black_region = cv2.bitwise_not(white_region)
        black_region[mask == 0] = 0  # 确保只处理白色区域内的黑色区域

        # 找到白色区域内部的黑色区域轮廓
        black_contours, _ = cv2.findContours(black_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for black_contour in black_contours:
            # 获取黑色区域的所有点坐标
            black_pixels = np.where(black_region == 255)

            # 确保有黑色像素存在
            if len(black_pixels[0]) > 0:
                black_regions_found = True
                top_pixel_y = np.min(black_pixels[0])  # 获取最顶端的黑色像素
                bottom_pixel_y = np.max(black_pixels[0])  # 获取最底端的黑色像素
                x_values = black_pixels[1]

                # 找到最顶端和最底端对应的x坐标
                top_pixel_x = x_values[np.argmin(black_pixels[0])]
                bottom_pixel_x = x_values[np.argmax(black_pixels[0])]

                # 向上延伸顶端像素
                if top_pixel_y - extension_pixels >= 0:
                    for i in range(top_pixel_y, top_pixel_y - extension_pixels, -1):
                        binary[i, top_pixel_x] = 0
                else:
                    for i in range(top_pixel_y, 0, -1):
                        binary[i, top_pixel_x] = 0

                # 向下延伸底端像素
                if bottom_pixel_y + extension_pixels < binary.shape[0]:
                    for i in range(bottom_pixel_y, bottom_pixel_y + extension_pixels):
                        binary[i, bottom_pixel_x] = 0
                else:
                    for i in range(bottom_pixel_y, binary.shape[0]):
                        binary[i, bottom_pixel_x] = 0

    return binary if black_regions_found else image  # 如果找到黑色区域则返回处理后的图像，否则返回原图

# 对白色区域进行腐蚀处理的函数
def erode_white_regions(image, erosion_size=1):
    # 创建腐蚀内核并进行腐蚀操作
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    processed_image = cv2.erode(image, kernel, iterations=1)
    return processed_image

# 分割并保存白色区域的函数
def split_and_save_white_regions(processed_image, output_dir, base_filename, region_id, sub_id, part_prefix):
    # 再次对处理后的图像进行白色区域检测并保存
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        mask = np.zeros_like(processed_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        if cv2.contourArea(contour) >= 10:  # 仅保存面积大于等于10的区域
            sub_output_path = os.path.join(output_dir, f"{base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
            io.imsave(sub_output_path, mask)

            # 计算并打印白色区域长度
            lengths_class_1, length_class_2 = calculate_lengths(mask, unit_length=1.0)
            print(f"文件名: {base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
            print("一类区域长度:", lengths_class_1)
            if length_class_2 is not None:
                print("二类区域长度:", length_class_2)
            print("--------")

# 计算白色区域长度的函数
def calculate_lengths(image, unit_length=1.0):
    binary_image = image > 0.5  # 将图像转换为二值化
    labeled_image, num_labels = measure.label(binary_image, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_image)

    if len(props) > 1:
        class_1 = sorted(props[:-1], key=lambda x: -x.major_axis_length)  # 对区域按长度排序
        class_2 = props[-1]  # 最长的为class_2
        lengths_class_1 = [prop.major_axis_length * unit_length for prop in class_1]  # 计算一类区域的长度
        min_class_1_y = min([prop.bbox[0] for prop in class_1])  # 获取最小的y坐标
        max_class_2_y = class_2.bbox[2]  # 获取最大的y坐标
        length_class_2 = (max_class_2_y - min_class_1_y) * unit_length  # 计算二类区域的长度
    else:
        lengths_class_1 = [props[0].major_axis_length * unit_length]  # 如果只有一类区域
        length_class_2 = None  # 没有二类区域

    return lengths_class_1, length_class_2

# 处理图像中指定像素值的区域
def process_image_by_color(image_path, output_dir, pixel_value, part_prefix, base_filename):
    original_image = io.imread(image_path, as_gray=True)  # 读取图像并转换为灰度图
    mask = (original_image == pixel_value)  # 创建掩码，仅保留指定像素值的区域
    labeled_image, num_labels = measure.label(mask, connectivity=2, return_num=True)  # 对掩码进行标记

    for i in range(1, num_labels + 1):
        region = (labeled_image == i).astype(np.uint8) * 255  # 将区域转换为二值图像

        processed_image = extend_black_regions(region, extension_pixels=20)  # 扩展黑色区域
        eroded_image = erode_white_regions(processed_image, erosion_size=3)  # 腐蚀白色区域
        split_and_save_white_regions(eroded_image, output_dir, base_filename, i, 1, part_prefix)  # 分割并保存处理后的区域

# 处理目录中的所有图像文件
def split_image_by_color_in_directory(input_dir, output_dir):
    # 循环处理目录中的每个图像文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]

            # 处理像素值为218的部分
            process_image_by_color(image_path, output_dir, pixel_value=254, part_prefix="region_1", base_filename=base_filename)

            # 处理像素值为181的部分
            process_image_by_color(image_path, output_dir, pixel_value=127, part_prefix="region_2", base_filename=base_filename)

# 示例用法
input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result2'
output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'  # 输出目录
split_image_by_color_in_directory(input_dir, output_dir)
