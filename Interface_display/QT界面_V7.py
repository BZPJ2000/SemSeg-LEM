import sys
import torch
import os
import shutil  # 引入shutil模块
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
from 分解 import split_image_by_color_main
from 合成 import merge_images_by_group
from 标记 import process_images_main


# 模型预测函数
def predict_segmentation(model_path, image_path, output_dir, num_classes=3, image_size=(224, 224)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = UNetWithAttention(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(image_path)
    # 加载并预处理图像
    image = Image.open(image_path)
    image = image.resize(image_size)
    image_np = np.array(image)

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

    return predicted_image_path, []


# 辅助函数：清空指定目录
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


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

    def clear_results(self):
        """清除所有显示的图片和结果。"""
        self.original_label.clear()
        self.predicted_label.clear()

        for label in self.split_labels:
            self.image_layout.removeWidget(label)
            label.deleteLater()

        self.split_labels = []

    def upload_image(self):
        """导入新图像并清除之前的结果。"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像文件", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            # 清除之前的结果
            self.clear_results()

            self.image_path = file_path
            pixmap = QtGui.QPixmap(self.image_path)
            self.original_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
            self.predict_button.setEnabled(True)

    def predict_image(self):
        if hasattr(self, 'image_path'):
            output_dir = "output_parts"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 清空 split_results 目录
            self.split_dir = os.path.join(output_dir, 'split_results')
            clear_directory(self.split_dir)

            # 模型预测
            self.predicted_image_path, _ = predict_segmentation(self.model_path, self.image_path, output_dir)

            # 调用分割函数
            split_image_by_color_main(output_dir, self.split_dir)

            # 显示预测的整张图像
            pixmap = QtGui.QPixmap(self.predicted_image_path)
            self.predicted_label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
            self.split_button.setEnabled(True)

    def split_image(self):
        if hasattr(self, 'split_dir'):
            # 标注路径长度
            processed_dir = os.path.join(self.split_dir, 'processed_results')
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
            process_images_main(self.split_dir, processed_dir, resolution_factor=17)

            # 合并标注结果
            final_dir = os.path.join(processed_dir, 'final_results')
            if not os.path.exists(final_dir):
                os.makedirs(final_dir)
            merge_images_by_group(processed_dir, final_dir, blacken_height=1)

            # 显示最终合并结果
            final_images = [f for f in os.listdir(final_dir) if f.endswith('.png')]
            for img_name in final_images:
                img_path = os.path.join(final_dir, img_name)

                # 显示结果到 Qt 界面
                label = QLabel("最终结果")
                label.setAlignment(Qt.AlignCenter)
                pixmap = QtGui.QPixmap(img_path)
                label.setPixmap(pixmap.scaled(256, 256, aspectRatioMode=1))
                self.image_layout.addWidget(label)
                self.split_labels.append(label)


# 主函数
def main():
    app = QApplication(sys.argv)
    model_path = r"E:\A_workbench\A-lab\Unet\Unet_complit\checkpoints\Unet_SETA_108150.pth"
    window = ImageSegmentationApp(model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
