import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def extend_black_regions(image, extension_pixels=20, threshold=127):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_regions_found = False

    for contour in contours:
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        white_region = cv2.bitwise_and(binary, mask)
        black_region = cv2.bitwise_not(white_region)
        black_region[mask == 0] = 0
        black_contours, _ = cv2.findContours(black_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for black_contour in black_contours:
            black_pixels = np.where(black_region == 255)
            if len(black_pixels[0]) > 0:
                black_regions_found = True
                top_pixel_y = np.min(black_pixels[0])
                bottom_pixel_y = np.max(black_pixels[0])
                x_values = black_pixels[1]
                top_pixel_x = x_values[np.argmin(black_pixels[0])]
                bottom_pixel_x = x_values[np.argmax(black_pixels[0])]

                if top_pixel_y - extension_pixels >= 0:
                    for i in range(top_pixel_y, top_pixel_y - extension_pixels, -1):
                        binary[i, top_pixel_x] = 0
                else:
                    for i in range(top_pixel_y, 0, -1):
                        binary[i, top_pixel_x] = 0

                if bottom_pixel_y + extension_pixels < binary.shape[0]:
                    for i in range(bottom_pixel_y, bottom_pixel_y + extension_pixels):
                        binary[i, bottom_pixel_x] = 0
                else:
                    for i in range(bottom_pixel_y, binary.shape[0]):
                        binary[i, bottom_pixel_x] = 0

    return binary if black_regions_found else image

def erode_white_regions(image, erosion_size=1):
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    processed_image = cv2.erode(image, kernel, iterations=1)
    return processed_image

def split_and_save_white_regions(processed_image, output_dir, base_filename, region_id, sub_id, part_prefix):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))

    for idx, contour in enumerate(contours):
        mask = np.zeros_like(processed_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        if cv2.contourArea(contour) >= 10:
            sub_output_path = os.path.join(output_dir, f"{base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
            io.imsave(sub_output_path, mask)

            lengths_class_1, length_class_2 = calculate_lengths(mask, unit_length=1.0)
            print(f"文件名: {base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
            print("一类区域长度:", lengths_class_1)
            if length_class_2 is not None:
                print("二类区域长度:", length_class_2)
            print("--------")

def calculate_lengths(image, unit_length=1.0):
    binary_image = image > 0.5
    labeled_image, num_labels = measure.label(binary_image, return_num=True, connectivity=2)
    props = measure.regionprops(labeled_image)

    if len(props) > 1:
        class_1 = sorted(props[:-1], key=lambda x: -x.major_axis_length)
        class_2 = props[-1]
        lengths_class_1 = [prop.major_axis_length * unit_length for prop in class_1]
        min_class_1_y = min([prop.bbox[0] for prop in class_1])
        max_class_2_y = class_2.bbox[2]
        length_class_2 = (max_class_2_y - min_class_1_y) * unit_length
    else:
        lengths_class_1 = [props[0].major_axis_length * unit_length]
        length_class_2 = None

    return lengths_class_1, length_class_2

def process_image_by_color(image_path, output_dir, pixel_value, part_prefix, base_filename):
    original_image = io.imread(image_path, as_gray=True)
    mask = (original_image == pixel_value)
    labeled_image, num_labels = measure.label(mask, connectivity=2, return_num=True)

    for i in range(1, num_labels + 1):
        region = (labeled_image == i).astype(np.uint8) * 255

        processed_image = extend_black_regions(region, extension_pixels=20)
        eroded_image = erode_white_regions(processed_image, erosion_size=3)
        split_and_save_white_regions(eroded_image, output_dir, base_filename, i, 1, part_prefix)

def split_image_by_color_main(input_dir, output_dir):
    """
    主函数，用于处理输入目录中的所有图像文件，并按照颜色值分割后保存到输出目录。

    Args:
        input_dir (str): 输入文件夹路径，包含待处理的图像文件。
        output_dir (str): 输出文件夹路径，用于保存分割后的结果。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]

            # 按颜色值 254 和 127 进行分割处理
            process_image_by_color(image_path, output_dir, pixel_value=254, part_prefix="region_1", base_filename=base_filename)
            process_image_by_color(image_path, output_dir, pixel_value=127, part_prefix="region_2", base_filename=base_filename)


# input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result'
# output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'
#
# # 调用主函数处理图像
# split_image_by_color_main(input_dir, output_dir)