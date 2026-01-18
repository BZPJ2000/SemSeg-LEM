import sys
import torch
import os
import shutil  # 引入shutil模块
import numpy as np
from PIL import Image
import cv2  # 引入OpenCV模块
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
from 分解 import split_image_by_color_main
from 合成 import merge_images_by_group
from 标记 import process_images_main


# 模型预测函数
def predict_segmentation(model, device, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()

    # 将预测结果映射为可视化图像
    predicted_image = Image.fromarray((predicted * 127).astype(np.uint8))  # 映射到不同的灰度值
    return predicted_image


# 辅助函数：清空指定目录
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# 视频处理线程
class VideoThread(QThread):
    original_frame_signal = pyqtSignal(QtGui.QImage)
    predicted_frame_signal = pyqtSignal(QtGui.QImage)

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.running = False

        # 设置设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 加载模型
        self.model = UNetWithAttention(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print("模型已加载")

    def run(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            # 原始帧处理
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = original_frame.shape
            bytes_per_line = 3 * width
            original_qimg = QtGui.QImage(original_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.original_frame_signal.emit(original_qimg)

            # 预测帧处理
            resized_image = cv2.resize(original_frame, (224, 224))
            image_np = np.array(resized_image)

            if len(image_np.shape) == 2:  # 灰度图
                image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            else:  # RGB图
                image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            # 进行预测
            predicted_image = predict_segmentation(self.model, self.device, image_tensor)

            # 将预测结果转换为OpenCV图像
            predicted_np = np.array(predicted_image)
            predicted_color = cv2.applyColorMap(predicted_np, cv2.COLORMAP_JET)
            predicted_color = cv2.cvtColor(predicted_color, cv2.COLOR_BGR2RGB)
            predicted_qimg = QtGui.QImage(predicted_color.data, predicted_color.shape[1], predicted_color.shape[0],
                                         3 * predicted_color.shape[1], QtGui.QImage.Format_RGB888)
            self.predicted_frame_signal.emit(predicted_qimg)

            # 控制帧率
            self.msleep(30)  # 大约33帧每秒

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


# 创建Qt界面
class ImageSegmentationApp(QtWidgets.QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.video_thread = None  # 添加视频线程属性
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像分割工具')
        self.setGeometry(100, 100, 1200, 800)  # 调整窗口大小以适应视频显示

        # 布局设置
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()
        button_layout = QVBoxLayout()


        # 添加视频显示标签
        self.video_original_label = QLabel("原始视频")
        self.video_original_label.setAlignment(Qt.AlignCenter)
        self.video_original_label.setFixedSize(560, 420)
        top_layout.addWidget(self.video_original_label)

        self.video_predicted_label = QLabel("预测视频")
        self.video_predicted_label.setAlignment(Qt.AlignCenter)
        self.video_predicted_label.setFixedSize(560, 420)
        top_layout.addWidget(self.video_predicted_label)

        # 分割图像标签容器
        self.split_labels = []


        # 启动视频按钮
        self.start_video_button = QPushButton('启动视频')
        self.start_video_button.clicked.connect(self.start_video)
        button_layout.addWidget(self.start_video_button)

        # 停止视频按钮
        self.stop_video_button = QPushButton('停止视频')
        self.stop_video_button.clicked.connect(self.stop_video)
        self.stop_video_button.setEnabled(False)
        button_layout.addWidget(self.stop_video_button)

        # 设置按钮布局
        bottom_layout.addLayout(button_layout)

        # 设置主窗口布局
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def clear_results(self):
        """清除所有显示的图片和结果。"""
        self.original_label.clear()
        self.predicted_label.clear()

        for label in self.split_labels:
            self.layout().itemAt(0).layout().removeWidget(label)
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
            self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio))
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
            self.predicted_label.setPixmap(pixmap.scaled(self.predicted_label.size(), Qt.KeepAspectRatio))
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
                label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
                self.layout().itemAt(0).layout().addWidget(label)
                self.split_labels.append(label)

    def start_video(self):
        """启动视频捕捉和预测"""
        if self.video_thread is None:
            self.video_thread = VideoThread(self.model_path)
            self.video_thread.original_frame_signal.connect(self.update_original_video)
            self.video_thread.predicted_frame_signal.connect(self.update_predicted_video)
            self.video_thread.start()
            self.start_video_button.setEnabled(False)
            self.stop_video_button.setEnabled(True)

    def stop_video(self):
        """停止视频捕捉和预测"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
            self.start_video_button.setEnabled(True)
            self.stop_video_button.setEnabled(False)

    def update_original_video(self, qimg):
        """更新原始视频显示"""
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.video_original_label.setPixmap(pixmap.scaled(self.video_original_label.size(), Qt.KeepAspectRatio))

    def update_predicted_video(self, qimg):
        """更新预测视频显示"""
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.video_predicted_label.setPixmap(pixmap.scaled(self.video_predicted_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        """确保线程在关闭时被正确终止"""
        if self.video_thread is not None:
            self.video_thread.stop()
        event.accept()


# 主函数
def main():
    app = QApplication(sys.argv)
    model_path = r"E:\A_workbench\A-lab\Unet\Unet_complit\checkpoints\Unet_SETA_108150.pth"
    window = ImageSegmentationApp(model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
