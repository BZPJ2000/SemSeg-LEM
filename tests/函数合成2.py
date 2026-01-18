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

def split_image_by_color_in_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]

            process_image_by_color(image_path, output_dir, pixel_value=254, part_prefix="region_1", base_filename=base_filename)
            process_image_by_color(image_path, output_dir, pixel_value=127, part_prefix="region_2", base_filename=base_filename)

input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result2'
output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'
split_image_by_color_in_directory(input_dir, output_dir)







import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
from collections import deque
import cv2

# 设置输入和输出文件夹路径
input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'
output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'
csv_file_path = os.path.join(output_dir, 'path_lengths.csv')

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def find_top_and_bottom(binary_image):
    top_most = None
    bottom_most = None

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1:
                if top_most is None:
                    top_most = (i, j)
                bottom_most = (i, j)

    return top_most, bottom_most

def bfs_path(binary_image, start, end):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        (current, path) = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        if current == end:
            return path

        x, y = current
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1] and binary_image[nx, ny] == 1:
                queue.append(((nx, ny), path + [(nx, ny)]))

    return []

def process_with_contours(image_np, output_image, draw):
    binary = (image_np > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    max_distance = 0
    point1 = None
    point2 = None

    for i in range(len(max_contour)):
        for j in range(i + 1, len(max_contour)):
            dist = cv2.norm(max_contour[i] - max_contour[j])
            if dist > max_distance:
                max_distance = dist
                point1 = tuple(max_contour[i][0])
                point2 = tuple(max_contour[j][0])

    draw.line((point1[0], point1[1], point2[0], point2[1]), fill=(0, 255, 0), width=2)
    return max_distance / 11

with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Path Length (mm)'])

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('L')
            image_np = np.array(image)

            binary_image = (image_np > 0).astype(int)
            top_most, bottom_most = find_top_and_bottom(binary_image)

            if top_most and bottom_most:
                path = bfs_path(binary_image, top_most, bottom_most)
                longest_path_length_px = len(path)
                longest_path_length_mm = longest_path_length_px / 11
            else:
                path = []
                longest_path_length_mm = 0

            output_image = Image.fromarray(image_np).convert('RGB')
            draw = ImageDraw.Draw(output_image)
            changeling = 10
            if longest_path_length_mm > changeling:
                longest_path_length_mm = process_with_contours(image_np, output_image, draw)
            else:
                for i in range(1, len(path)):
                    draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)

            font = ImageFont.load_default()
            draw.text((10, 10), f"Length: {longest_path_length_mm:.2f} mm", fill=(0, 255, 0), font=font)

            output_image_path = os.path.join(output_dir, f'marked_{filename}')
            output_image.save(output_image_path)

            csv_writer.writerow([filename, longest_path_length_mm])

            print(f"已处理文件 {filename}: 路径长度为 {longest_path_length_mm:.2f} 毫米")
            print(f"标记的路径已保存到: {output_image_path}")



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





