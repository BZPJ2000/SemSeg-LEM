import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
from collections import deque
import cv2

def process_images_main(input_dir, output_dir, resolution_factor=17):
    """
    主函数：处理输入目录中的图像，计算路径长度，标注路径并保存结果。

    Args:
        input_dir (str): 输入文件夹路径，包含待处理的图像。
        output_dir (str): 输出文件夹路径，用于保存处理后的图像和路径长度 CSV 文件。
        resolution_factor (float): 像素到毫米的转换因子，默认为 17。
    """
    def process_images(input_dir, output_dir, resolution_factor):
        """
        内部处理函数，用于计算路径、标注图像和保存结果。

        Args:
            input_dir (str): 输入文件夹路径。
            output_dir (str): 输出文件夹路径。
            resolution_factor (float): 像素到毫米的转换因子。
        """
        csv_file_path = os.path.join(output_dir, 'path_lengths.csv')

        # 如果输出文件夹不存在，则创建
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

        def process_with_contours(image_np, output_image, draw, mark_color=(255, 0, 0)):
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

            # 绘制路径并标注长度
            midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
            draw.line((point1[0], point1[1], point2[0], point2[1]), fill=mark_color, width=2)
            length_mm = max_distance / resolution_factor
            draw.text(midpoint, f"{length_mm:.2f} mm", fill=mark_color, font=ImageFont.load_default())
            return length_mm

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
                        longest_path_length_mm = longest_path_length_px / resolution_factor
                    else:
                        path = []
                        longest_path_length_mm = 0

                    # 如果路径长度小于 1.5 mm，不显示也不保存
                    if longest_path_length_mm < 1.5:
                        print(f"File {filename}: Path length is too small ({longest_path_length_mm:.2f} mm), skipping.")
                        continue

                    output_image = Image.fromarray(image_np).convert('RGB')
                    draw = ImageDraw.Draw(output_image)

                    if longest_path_length_mm > 10:
                        # 重新搜索路径，如果仍然大于 10 mm，则标记为红色
                        print(f"File {filename}: Path length > 10 mm, re-processing.")
                        longest_path_length_mm = process_with_contours(image_np, output_image, draw, mark_color=(255, 0, 0))
                        if longest_path_length_mm > 10:
                            print(f"File {filename}: Marked with red line as length > 10 mm.")

                    else:
                        # 普通路径处理，标记为绿色
                        for i in range(1, len(path)):
                            draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)
                        # 在路径中间标注长度
                        midpoint = ((path[0][1] + path[-1][1]) // 2, (path[0][0] + path[-1][0]) // 2)
                        draw.text(midpoint, f"{longest_path_length_mm:.2f} mm", fill=(255, 0, 0), font=ImageFont.load_default())

                    output_image_path = os.path.join(output_dir, f'marked_{filename}')
                    output_image.save(output_image_path)

                    csv_writer.writerow([filename, longest_path_length_mm])

                    print(f"Processed file {filename}: Path length is {longest_path_length_mm:.2f} mm")
                    print(f"Marked path saved to: {output_image_path}")

    process_images(input_dir, output_dir, resolution_factor)
