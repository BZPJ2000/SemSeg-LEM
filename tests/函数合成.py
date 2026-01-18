import torch
from PIL import Image
import numpy as np
from model.UNet_SETA.Unet_ASE_V3 import UNetWithAttention
import cv2



def predict_segmentation(model_path, image_path, output_path, num_classes=3, image_size=(224, 224)):
    # 首先，确定使用的设备是GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载模型
    model = UNetWithAttention(num_classes=num_classes).to(device)  # 初始化模型并移动到指定设备
    model.load_state_dict(torch.load(model_path))  # 加载模型的权重参数
    model.eval()  # 设置模型为评估模式
    # 加载并预处理图像
    image = Image.open(image_path)  # 打开输入图像
    image = image.resize(image_size)  # 调整图像大小
    image_np = np.array(image)  # 转换为NumPy数组
    # 将图像转换为模型输入格式
    if len(image_np.shape) == 2:  # 如果是灰度图
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加维度 (1, 1, H, W)
    else:  # 如果是RGB图像
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, C, H, W)

    image_tensor = image_tensor.to(device)  # 将图像数据移动到设备（GPU/CPU）
    # 预测
    with torch.no_grad():  # 不计算梯度以减少内存占用
        output = model(image_tensor)  # 通过模型进行预测
        _, predicted = torch.max(output, 1)  # 获取每个像素预测的类别
        predicted = predicted.squeeze().cpu().numpy()  # 将结果转为NumPy数组
    # 将预测结果转换为图像
    predicted_img = Image.fromarray((predicted * 127).astype(np.uint8))  # 将类别0, 1, 2分别映射为0, 127, 254
    # 保存预测结果
    predicted_img.save(output_path)
    print(f"Prediction saved to {output_path}")
    # 返回预测结果的数组
    return predicted

def process_image(image_path, output_path, scale_factor=13):
    # 读取图像
    image = cv2.imread(image_path)

    # 检查图像是否成功读取
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像中的所有唯一像素值（排除背景值0）
    unique_pixel_values = set(np.unique(gray)) - {0}

    # 存储每个区域的计算结果
    results = []

    # 遍历每个唯一像素值，逐个处理
    for pixel_value in unique_pixel_values:
        # 生成当前像素值的二值掩码
        binary = np.where(gray == pixel_value, 255, 0).astype(np.uint8)

        # 找到该掩码中的所有轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个轮廓（即当前像素值对应的区域）
        for contour in contours:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)

            # 初始化最大距离和对应的点
            max_distance = 0
            point1 = None
            point2 = None

            # 找到轮廓中距离最远的两个点
            for i in range(len(contour)):
                for j in range(i + 1, len(contour)):
                    dist = cv2.norm(contour[i] - contour[j])
                    if dist > max_distance:
                        max_distance = dist
                        point1 = tuple(contour[i][0])
                        point2 = tuple(contour[j][0])

            # 将距离转换为毫米
            distance_mm = max_distance / scale_factor

            # 保存该轮廓的结果
            line_coordinates = (point1, point2) if point1 and point2 else None

            # 将结果添加到列表中
            results.append({
                'pixel_value': pixel_value,
                'perimeter': perimeter,
                'max_distance': distance_mm,
                'point1': point1,
                'point2': point2,
                'line_coordinates': line_coordinates
            })

    # 返回所有区域的计算结果
    return results

def overlay_lines_on_image(original_image_path, line_data, output_image_path, processed_image_size):

    # 读取原始图像
    original_image = cv2.imread(original_image_path)

    # 检查图像是否成功读取
    if original_image is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    # 获取原始图像尺寸
    original_height, original_width = original_image.shape[:2]

    # 处理后的图像尺寸 (width, height)
    processed_width, processed_height = processed_image_size

    # 计算缩放比例
    scale_x = original_width / processed_width
    scale_y = original_height / processed_height

    # 遍历每个区域的数据，绘制线段
    for data in line_data:
        # 检查 'line_coordinates' 是否存在且不为 None
        if data.get('line_coordinates'):
            point1, point2 = data['line_coordinates']
            if point1 and point2:
                # 缩放线段坐标
                scaled_point1 = (int(point1[0] * scale_x), int(point1[1] * scale_y))
                scaled_point2 = (int(point2[0] * scale_x), int(point2[1] * scale_y))

                # 在原图上绘制缩放后的线段
                cv2.line(original_image, scaled_point1, scaled_point2, (0, 255, 0), 2)

                # 计算中点，用于显示线段长度
                mid_point = ((scaled_point1[0] + scaled_point2[0]) // 2, (scaled_point1[1] + scaled_point2[1]) // 2)

                # 在原图上标注长度信息
                cv2.putText(original_image, f"{data['max_distance']:.2f}", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 222, 0), 2, cv2.LINE_AA)

    # 保存处理后的图像
    cv2.imwrite(output_image_path, original_image)
    print(f"Overlay image saved at {output_image_path}")


# 第一步：预测分割结果
predicted = predict_segmentation(
    model_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\checkpoints\Unet_SETA_108150.pth',
    image_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages\00022.png',
    output_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\predicted.png'
)

# 第二步：处理预测结果，找到每个非0区域的最大距离线段
line_data = process_image(
    image_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\predicted.png',
    output_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\processed.png',
    scale_factor=17  # 使用的像素到毫米的转换比例
)

# 第三步：将计算出的线段叠加到原始图像上并保存
overlay_lines_on_image(
    original_image_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\ConvertedImages\00022.png',
    line_data=line_data,
    output_image_path=r'E:\A_workbench\A-lab\Unet\Unet_complit\data\final_output.png',
    processed_image_size=(224, 224)  # 和分割模型输入的尺寸一致
)


