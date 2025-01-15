import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
from collections import deque
import cv2

def process_images(input_dir, output_dir, resolution_factor=17):
    """
    Process images in the input directory, calculate path lengths, mark paths on images, 
    and save results to the output directory and a CSV file.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory for processed images and CSV file.
        resolution_factor (float): Factor to convert pixel length to mm (default is 17).
    """
    csv_file_path = os.path.join(output_dir, 'path_lengths.csv')

    # If the output directory does not exist, create it
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
        return max_distance / resolution_factor

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

                print(f"Processed file {filename}: Path length is {longest_path_length_mm:.2f} mm")
                print(f"Marked path saved to: {output_image_path}")

input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'
output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'

process_images(input_dir, output_dir, resolution_factor=17)
