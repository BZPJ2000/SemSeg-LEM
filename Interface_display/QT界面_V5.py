import sys
import torch
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
import cv2
from collections import deque
import csv

# 图像分割函数
def predict_segmentation(model_path, image_path, output_dir, num_classes=3, image_size=(224, 224)):
    # 确认使用的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = UNetWithAttention(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载并预处理图像
    image = Image.open(image_path)
    image = image.resize(image_size)
    image_np = np.array(image)

    # 转换为模型输入格式
    if len(image_np.shape) == 2:  # 灰度图
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:  # RGB图
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()

    # 保存预测的整张图像
    predicted_image = Image.fromarray((predicted * 127).astype(np.uint8))  # 映射到不同的灰度值
    predicted_image_path = os.path.join(output_dir, "predicted_output.png")
    predicted_image.save(predicted_image_path)
    print(f"Predicted image saved to {predicted_image_path}")

    # 分割结果的每个部分保存为独立的图像
    unique_values = np.unique(predicted)
    output_paths = []
    for value in unique_values:
        if value != 0:  # 忽略背景
            part_mask = (predicted == value).astype(np.uint8) * 255
            part_img = Image.fromarray(part_mask)
            output_path = os.path.join(output_dir, f"predicted_output_part_{int(value)}.png")
            part_img.save(output_path)
            output_paths.append(output_path)
            print(f"Prediction part saved to {output_path}")

    return predicted_image_path, output_paths

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

# 使用轮廓处理找到最长路径
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

    if max_distance / 17 >= 0.5:
        draw.line((point1[0], point1[1], point2[0], point2[1]), fill=(0, 255, 0), width=2)
        draw.text(((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2), f"Length: {max_distance / 17:.2f} mm", fill=(0, 255, 0))
    return max_distance / 17

# 计算并在图像上标注长度
def calculate_and_draw_length(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    binary_image = (image_np[:, :, 0] > 0).astype(np.uint8)  # 使用红色通道的二值化
    num_labels, labeled_image = cv2.connectedComponents(binary_image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for label in range(1, num_labels):
        region_mask = (labeled_image == label).astype(np.uint8)
        top_most, bottom_most = find_top_and_bottom(region_mask)
        if top_most and bottom_most:
            path = bfs_path(region_mask, top_most, bottom_most)
            longest_path_length_px = len(path)
            longest_path_length_mm = longest_path_length_px / 17

            if longest_path_length_mm > 10:
                longest_path_length_mm = process_with_contours(region_mask, image, draw)
            elif longest_path_length_mm >= 0.5:
                for i in range(1, len(path)):
                    draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)
                midpoint = ((path[0][1] + path[-1][1]) // 2, (path[0][0] + path[-1][0]) // 2)
                draw.text(midpoint, f"Length: {longest_path_length_mm:.2f} mm", fill=(0, 255, 0), font=font)

    image.save(image_path)
    print(f"Length calculated and annotated on {image_path}")



# 创建Qt界面
class ImageSegmentationApp(QtWidgets.QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像分割工具')
        self.setGeometry(100, 100, 1000, 600)

        # 布局设置
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        # 原始图像标签
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.original_label)

        # 预测图像标签
        self.predicted_label = QLabel("预测图像")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.predicted_label)

        # 分割图像标签容器
        self.split_labels = []

        # 导入图像按钮
        self.upload_button = QPushButton('导入图像')
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        # 预测按钮
        self.predict_button = QPushButton('预测分割')
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)
        button_layout.addWidget(self.predict_button)

        # 分割图像按钮
        self.split_button = QPushButton('分割图像并计算长度')
        self.split_button.clicked.connect(self.split_image)
        self.split_button.setEnabled(False)
        button_layout.addWidget(self.split_button)

        # 设置主窗口布局
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.image_layout = image_layout

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(self.image_path)
            self.original_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
            self.predict_button.setEnabled(True)

    def predict_image(self):
        if hasattr(self, 'image_path'):
            output_dir = "../output_parts"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.predicted_image_path, self.output_paths = predict_segmentation(self.model_path, self.image_path, output_dir)

            # 显示预测的整张图像
            pixmap = QtGui.QPixmap(self.predicted_image_path)
            self.predicted_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))

            self.split_button.setEnabled(True)

    def split_image(self):
        if hasattr(self, 'output_paths'):
            # 清除之前的分割图像标签
            for label in self.split_labels:
                self.image_layout.removeWidget(label)
                label.deleteLater()
            self.split_labels = []

            # 显示所有分割的图像部分并计算长度
            for output_path in self.output_paths:
                calculate_and_draw_length(output_path)
                label = QLabel("分割部分图像")
                label.setAlignment(Qt.AlignCenter)
                pixmap = QtGui.QPixmap(output_path)
                label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
                self.image_layout.addWidget(label)
                self.split_labels.append(label)

# 主函数
def main():
    app = QApplication(sys.argv)
    model_path = "checkpoints/Unet_SETA_108150.pth"
    window = ImageSegmentationApp(model_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
