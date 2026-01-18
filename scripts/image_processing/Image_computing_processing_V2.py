import matplotlib.pyplot as plt
from skimage import io, measure, morphology
import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import deque, defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# %%

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

    part_count = 0
    for idx, contour in enumerate(contours):
        mask = np.zeros_like(processed_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        if cv2.contourArea(contour) >= 10:
            part_count += 1
            sub_output_path = os.path.join(output_dir,
                                           f"{base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
            io.imsave(sub_output_path, mask)

            lengths_class_1, length_class_2 = calculate_lengths(mask, unit_length=1.0)
            print(f"文件名: {base_filename}_{part_prefix}_{region_id}_sub_{sub_id}_part_{idx + 1}.png")
    return part_count


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

    total_parts = 0
    for i in range(1, num_labels + 1):
        region = (labeled_image == i).astype(np.uint8) * 255

        processed_image = extend_black_regions(region, extension_pixels=20)
        eroded_image = erode_white_regions(processed_image, erosion_size=3)
        part_count = split_and_save_white_regions(eroded_image, output_dir, base_filename, i, 1, part_prefix)
        total_parts += part_count

    return total_parts


def split_image_by_color_in_directory(input_dir, output_dir):
    file_parts_count = defaultdict(int)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]

            # Process the image and update the count of parts for each base_filename
            total_parts = process_image_by_color(image_path, output_dir, pixel_value=254, part_prefix="region_1",
                                                 base_filename=base_filename)
            total_parts += process_image_by_color(image_path, output_dir, pixel_value=127, part_prefix="region_2",
                                                  base_filename=base_filename)

            file_parts_count[base_filename] += total_parts

    # Move files to corresponding directories based on the number of parts
    for base_filename, num_parts in file_parts_count.items():
        part_dir = os.path.join(output_dir, f"{num_parts}_parts")
        if not os.path.exists(part_dir):
            os.makedirs(part_dir)

        # Move all files starting with the base_filename to the corresponding directory
        for file in os.listdir(output_dir):
            if file.startswith(base_filename):
                file_path = os.path.join(output_dir, file)
                new_file_path = os.path.join(part_dir, file)
                if os.path.exists(file_path):
                    os.rename(file_path, new_file_path)
                else:
                    print(f"文件未找到: {file_path}")
# %%

# 路径长度计算相关函数
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

# 使用广度优先搜索算法计算路径
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

# 处理轮廓并计算最长路径
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
    return max_distance / 17

def process_images(A_dir, B_dir, csv_file_path, changeling=10):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(B_dir):
        os.makedirs(B_dir)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Path Length (mm)'])

        for root, _, files in os.walk(A_dir):
            for filename in files:
                if filename.endswith(".png"):
                    image_path = os.path.join(root, filename)
                    image = Image.open(image_path).convert('L')
                    image_np = np.array(image)

                    binary_image = (image_np > 0).astype(np.uint8)
                    # 通过解析文件名获取 num_regions
                    base_filename = os.path.splitext(filename)[0]
                    try:
                        num_regions = int(base_filename.split('_')[-1])  # 假设区域数量在文件名最后一部分
                    except ValueError:
                        num_regions = 1  # 如果解析失败，默认设置为1


                    top_most, bottom_most = find_top_and_bottom(binary_image)

                    if top_most and bottom_most:
                        path = bfs_path(binary_image, top_most, bottom_most)
                        longest_path_length_px = len(path)
                        longest_path_length_mm = longest_path_length_px / 17
                    else:
                        path = []
                        longest_path_length_mm = 0

                    output_image = Image.fromarray(image_np).convert('RGB')
                    draw = ImageDraw.Draw(output_image)

                    if longest_path_length_mm > changeling:
                        longest_path_length_mm = process_with_contours(image_np, output_image, draw)
                    else:
                        for i in range(1, len(path)):
                            draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)

                    font = ImageFont.load_default()
                    draw.text((10, 10), f"Length: {longest_path_length_mm:.2f} mm", fill=(0, 255, 0), font=font)

                    output_image_path = os.path.join(B_dir, f'marked_{filename}')
                    output_image.save(output_image_path)

                    # 将文件名、路径长度、类别和区域数量写入CSV文件
                    csv_writer.writerow([filename, longest_path_length_mm])

                    print(f"已处理文件 {filename}: 路径长度为 {longest_path_length_mm:.2f} 毫米")
                    print(f"标记的路径已保存到: {output_image_path}")


# %%
def merge_and_blacken_images(B_dir, C_dir, blacken_height=5):
    all_files = os.listdir(B_dir)
    marked_files = [f for f in all_files if f.startswith('marked_') and f.endswith('.png')]
    grouped_images = {}
    for file_name in marked_files:
        prefix = '_'.join(file_name.split('_')[:3]) + '_'
        if prefix not in grouped_images:
            grouped_images[prefix] = []
        grouped_images[prefix].append(file_name)
    # 处理每个分组的图片
    for prefix, image_names in grouped_images.items():
        merged_image = None
        for name in image_names:
            image_path = os.path.join(B_dir, name)
            image = np.array(Image.open(image_path))
            if merged_image is None:
                merged_image = np.zeros_like(image)
            merged_image = np.maximum(merged_image, image)
        merged_image[:blacken_height, :] = 0
        # 保存合并后的图片
        output_filename = prefix + 'merged.png'
        output_path = os.path.join(C_dir, output_filename)
        Image.fromarray(merged_image).save(output_path)
        print(f'图片已保存到: {output_path}')


input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result'
A_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'
B_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'
C_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_C'
csv_file_path = os.path.join(A_dir, 'path_lengths.csv')


split_image_by_color_in_directory(input_dir, A_dir)
process_images(A_dir, B_dir, csv_file_path)
merge_and_blacken_images(B_dir, C_dir)
