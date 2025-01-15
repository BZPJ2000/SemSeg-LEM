import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image
import numpy as np
import torch
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention

class ImagePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像预测应用')
        self.setGeometry(100, 100, 800, 600)

        # 原始图像显示
        self.original_image_label = QLabel(self)
        self.original_image_label.setFixedSize(400, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black;")

        # 预测结果显示
        self.prediction_image_label = QLabel(self)
        self.prediction_image_label.setFixedSize(400, 400)
        self.prediction_image_label.setStyleSheet("border: 1px solid black;")

        # 添加中文标签
        self.original_image_text = QLabel('原始图片', self)
        self.original_image_text.setAlignment(Qt.AlignCenter)
        self.prediction_image_text = QLabel('预测结果', self)
        self.prediction_image_text.setAlignment(Qt.AlignCenter)

        # 布局设置
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_image_label)
        original_layout.addWidget(self.original_image_text)

        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.prediction_image_label)
        prediction_layout.addWidget(self.prediction_image_text)

        image_layout = QHBoxLayout()
        image_layout.addLayout(original_layout)
        image_layout.addLayout(prediction_layout)

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

    def selectImage(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.showOriginalImage(file_path)
            self.predictImage(file_path)

    def showOriginalImage(self, file_path):
        pixmap = QPixmap(file_path)
        self.original_image_label.setPixmap(pixmap.scaled(self.original_image_label.size(), aspectRatioMode=True))

    def predictImage(self, file_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = r'checkpoints/Unet_SETA_108100.pth'
        num_classes = 3
        model = UNetWithAttention(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        image = Image.open(file_path)
        image = image.resize((224, 224))
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()

        predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))
        predicted_img = predicted_img.resize((400, 400))
        predicted_img_path = 'predicted_image.png'
        predicted_img.save(predicted_img_path)

        pixmap = QPixmap(predicted_img_path)
        self.prediction_image_label.setPixmap(pixmap)

        # 在最后显示中文注释
        self.annotation_label.setText('这里是注释。')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_prediction_app = ImagePredictionApp()
    image_prediction_app.show()
    sys.exit(app.exec_())
