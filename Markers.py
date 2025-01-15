import cv2
import numpy as np

# 读取图片
image = cv2.imread(r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A\00003_pred_region_2_1_sub_1_part_1.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用二值化以获得白色区域
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# 找到轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓
max_contour = max(contours, key=cv2.contourArea)

# 计算轮廓的周长
perimeter = cv2.arcLength(max_contour, True)
print(f"White region length: {perimeter}")

# 找到距离最远的两个点
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

# 画出绿色的线段连接这两个点
cv2.line(image, point1, point2, (0, 255, 0), 1)

# 保存结果图片
cv2.imwrite('output_image.png', image)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
