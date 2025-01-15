import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv

# 设置输入和输出文件夹路径
input_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_A'  # 输入图像文件夹路径
output_dir = r'E:\A_workbench\A-lab\Unet\Unet_complit\result_B'  # 输出图像文件夹路径
csv_file_path = os.path.join(output_dir, 'path_lengths.csv')  # CSV文件保存路径

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 找到图像中最上方和最下方的白色像素
def find_top_and_bottom(binary_image):
    top_most = None  # 用于存储最上方的白色像素位置
    bottom_most = None  # 用于存储最下方的白色像素位置

    # 遍历二值化图像的每个像素
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1:  # 如果当前像素是白色
                if top_most is None:
                    top_most = (i, j)  # 记录第一个遇到的白色像素作为最上方的像素
                bottom_most = (i, j)  # 持续更新，最终得到最下方的白色像素

    return top_most, bottom_most

# 使用深度优先搜索(DFS)找到从最上方到最下方的路径
def dfs_path(binary_image, start, end):
    stack = [(start, [start])]  # 初始化栈，存储起点及路径
    visited = set()  # 记录已访问的像素

    while stack:
        (current, path) = stack.pop()  # 从栈中取出当前像素及其路径
        if current in visited:
            continue  # 如果已经访问过，跳过
        visited.add(current)  # 记录当前像素已访问

        if current == end:
            return path  # 如果找到终点，返回路径

        x, y = current
        # 检查8个可能的方向（上下左右和四个对角线方向）
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 如果新位置在图像范围内且为白色像素，则加入栈
            if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1] and binary_image[nx, ny] == 1:
                stack.append(((nx, ny), path + [(nx, ny)]))

    return []  # 如果没有找到路径，返回空列表

# 创建并打开CSV文件
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Path Length (mm)'])  # 写入CSV表头

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # 检查文件是否为PNG格式
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('L')  # 打开图像并转换为灰度图
            image_np = np.array(image)

            # 将图像二值化（假设白色为对象，黑色为背景）
            binary_image = (image_np > 0).astype(int)

            # 找到图像中最上方和最下方的白色像素
            top_most, bottom_most = find_top_and_bottom(binary_image)

            # 查找从最上方到最下方的路径
            if top_most and bottom_most:
                path = dfs_path(binary_image, top_most, bottom_most)
                longest_path_length_px = len(path)  # 路径的像素长度
                longest_path_length_mm = longest_path_length_px / 13  # 将像素长度转换为毫米长度
            else:
                path = []
                longest_path_length_px = 0
                longest_path_length_mm = 0

            # 在图像上标记路径
            output_image = Image.fromarray(image_np).convert('RGB')  # 转换为RGB格式以便绘制彩色路径
            draw = ImageDraw.Draw(output_image)

            # 用绿色线条绘制路径
            for i in range(1, len(path)):
                draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)

            # 添加路径长度的文本（以毫米为单位）
            font = ImageFont.load_default()
            draw.text((10, 10), f"Length: {longest_path_length_mm:.2f} mm", fill=(0, 255, 0), font=font)

            # 保存输出图像
            output_image_path = os.path.join(output_dir, f'marked_{filename}')
            output_image.save(output_image_path)

            # 将文件名和路径长度写入CSV
            csv_writer.writerow([filename, longest_path_length_mm])

            print(f"已处理文件 {filename}: 最长路径长度为 {longest_path_length_mm:.2f} 毫米")
            print(f"标记的路径已保存到: {output_image_path}")

'''
查找最上方和最下方的白色像素：
find_top_and_bottom 函数用于遍历二值化后的图像，
找出图像中最上方和最下方的白色像素。这两个像素将用作后续路径查找的起点和终点。

深度优先搜索(DFS)路径查找：
dfs_path 函数使用深度优先搜索算法，从最上方像素开始，寻找通向最下方像素的路径。
路径是由相邻的白色像素组成的，算法检查八个方向（上下左右以及四个对角线方向）来寻找路径。
'''
