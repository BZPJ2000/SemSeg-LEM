import shutil
import sys
import os
import numpy as np
import torch
import cv2
import csv
from PIL import Image, ImageDraw, ImageFont
from skimage import io, measure
from collections import deque

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt5.QtGui import QPixmap
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention

class ImagePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 加载模型
        self.model = self.load_model()

    def initUI(self):
        self.setWindowTitle('图像预测应用')
        self.setGeometry(100, 100, 1200, 600)  # 调整窗口大小

        # 原始图像显示
        self.original_image_label = QLabel(self)
        self.original_image_label.setFixedSize(400, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black;")

        # 预测结果显示
        self.prediction_image_label = QLabel(self)
        self.prediction_image_label.setFixedSize(400, 400)
        self.prediction_image_label.setStyleSheet("border: 1px solid black;")

        # 最终处理后图像显示
        self.final_image_label = QLabel(self)
        self.final_image_label.setFixedSize(400, 400)
        self.final_image_label.setStyleSheet("border: 1px solid black;")

        # 添加中文标签
        self.original_image_text = QLabel('原始图片', self)
        self.original_image_text.setAlignment(Qt.AlignCenter)
        self.prediction_image_text = QLabel('预测结果', self)
        self.prediction_image_text.setAlignment(Qt.AlignCenter)
        self.final_image_text = QLabel('最终结果', self)
        self.final_image_text.setAlignment(Qt.AlignCenter)

        # 布局设置
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_image_label)
        original_layout.addWidget(self.original_image_text)

        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.prediction_image_label)
        prediction_layout.addWidget(self.prediction_image_text)

        final_layout = QVBoxLayout()
        final_layout.addWidget(self.final_image_label)
        final_layout.addWidget(self.final_image_text)

        image_layout = QHBoxLayout()
        image_layout.addLayout(original_layout)
        image_layout.addLayout(prediction_layout)
        image_layout.addLayout(final_layout)

        # 按钮
        self.select_image_button = QPushButton('选择图片', self)
        self.select_image_button.clicked.connect(self.selectImage)

        # 中文注释
        self.annotation_label = QLabel('', self)
        self.annotation_label.setAlignment(Qt.AlignCenter)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.select_image_button)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.annotation_label)

        self.setLayout(main_layout)

    def load_model(self):
        model_path = r'checkpoints/Unet_SETA_108100.pth'
        num_classes = 3
        model = UNetWithAttention(num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def selectImage(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # 重置图像显示
            self.original_image_label.clear()
            self.prediction_image_label.clear()
            self.final_image_label.clear()
            self.annotation_label.setText('')

            # 清理临时目录
            self.cleanup_temporary_directories(['temp_input', 'temp_output'])

            # 显示新选择的图片并进行处理
            self.showOriginalImage(file_path)
            self.predictImage(file_path)

    def cleanup_temporary_directories(self, directories):
        for directory in directories:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")

    def showOriginalImage(self, file_path):
        pixmap = QPixmap(file_path)
        self.original_image_label.setPixmap(
            pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio)
        )

    def predictImage(self, file_path):
        # 第一步：模型预测
        image = Image.open(file_path).convert('RGB')
        image = image.resize((224, 224))
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()
            # 打印预测结果中的唯一像素值
            print("预测结果中的唯一像素值：", np.unique(predicted))
        predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))
        predicted_img = predicted_img
        predicted_img_path = 'predicted_image.png'
        predicted_img.save(predicted_img_path)

        pixmap = QPixmap(predicted_img_path)
        self.prediction_image_label.setPixmap(
            pixmap.scaled(self.prediction_image_label.size(), Qt.KeepAspectRatio)
        )

        # 第二步：处理预测结果，生成最终图像
        final_image_path = self.process_prediction(predicted_img_path)
        pixmap_final = QPixmap(final_image_path)
        self.final_image_label.setPixmap(
            pixmap_final.scaled(self.final_image_label.size(), Qt.KeepAspectRatio)
        )

        # 在最后显示中文注释
        self.annotation_label.setText('注释')

    def process_prediction(self, predicted_img_path):
        # 创建临时输入和输出目录
        input_dir = '../temp_input'
        output_dir = '../temp_output'
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # 将预测结果图像保存到输入目录
        base_filename = 'predicted_result'
        input_image_path = os.path.join(input_dir, base_filename + '.png')
        predicted_image = Image.open(predicted_img_path)
        predicted_image.save(input_image_path)

        # 调用处理函数
        self.split_and_process_image(input_dir, output_dir, base_filename)

        # 最终的合并图像路径
        final_image_path = os.path.join(output_dir, base_filename + '_merged.png')


        return final_image_path

    def split_and_process_image(self, input_dir, output_dir, base_filename):

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

        def process_image_by_color(image_path, output_dir, pixel_value, part_prefix, base_filename):
            original_image = io.imread(image_path, as_gray=True)
            mask = (original_image == pixel_value)
            labeled_image, num_labels = measure.label(mask, connectivity=2, return_num=True)
            for i in range(1, num_labels + 1):
                region = (labeled_image == i).astype(np.uint8) * 255
                processed_image = extend_black_regions(region, extension_pixels=20)
                eroded_image = erode_white_regions(processed_image, erosion_size=3)
                # 保存处理后的图像
                output_path = os.path.join(output_dir, f"{base_filename}_{part_prefix}_{i}.png")
                io.imsave(output_path, eroded_image)

        # 调用处理函数
        input_image_path = os.path.join(input_dir, base_filename + '.png')
        process_image_by_color(input_image_path, output_dir, pixel_value=254, part_prefix="region_1", base_filename=base_filename)
        process_image_by_color(input_image_path, output_dir, pixel_value=127, part_prefix="region_2", base_filename=base_filename)

        # 第二部分：标记图像
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
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
            # Draw the line
            draw.line((point1[0], point1[1], point2[0], point2[1]), fill=(0, 255, 0), width=2)
            # Calculate midpoint for text
            mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
            # Convert distance to mm (example scale factor)
            distance_mm = max_distance / 13
            # Draw text near the line
            font = ImageFont.load_default()
            draw.text((mid_point[0] + 5, mid_point[1]), f"{distance_mm:.2f} mm", fill=(255, 0, 0), font=font)
            return distance_mm

        # 标记图像并保存
        for filename in os.listdir(output_dir):
            if filename.startswith(base_filename) and filename.endswith(".png"):
                image_path = os.path.join(output_dir, filename)
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

        # 第三部分：合并图像
        merged_image = None
        marked_files = [f for f in os.listdir(output_dir) if f.startswith('marked_') and f.endswith('.png')]
        for name in marked_files:
            image_path = os.path.join(output_dir, name)
            image = np.array(Image.open(image_path))
            if merged_image is None:
                merged_image = np.zeros_like(image)
            merged_image = np.maximum(merged_image, image)
        # 黑化顶部区域
        blacken_height = 22
        merged_image[:blacken_height, :] = 0
        # 保存合并后的图像
        output_filename = base_filename + '_merged.png'
        output_path = os.path.join(output_dir, output_filename)
        Image.fromarray(merged_image).save(output_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_prediction_app = ImagePredictionApp()
    image_prediction_app.show()
    sys.exit(app.exec_())
