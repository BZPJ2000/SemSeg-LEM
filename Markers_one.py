from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 加载图像
image_path = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\regions\00001_region_1_2_sub_1_part_1.png'
image = Image.open(image_path).convert('L')  # 打开图像并转换为灰度图
image_np = np.array(image)  # 将图像转换为NumPy数组

# 将图像二值化（假设白色为对象，黑色为背景）
binary_image = (image_np > 0).astype(int)

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

# 调用函数找到图像中的最上方和最下方的白色像素
top_most, bottom_most = find_top_and_bottom(binary_image)

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

# 查找从最上方到最下方的路径
if top_most and bottom_most:
    path = dfs_path(binary_image, top_most, bottom_most)  # 获取路径
    longest_path_length_px = len(path)  # 路径的像素长度
    longest_path_length_mm = longest_path_length_px / 11  # 将像素长度转换为毫米长度
else:
    path = []
    longest_path_length_px = 0
    longest_path_length_mm = 0

# 在图像上标记路径
output_image = Image.fromarray(image_np).convert('RGB')  # 将图像转换为RGB格式以便绘制彩色路径
draw = ImageDraw.Draw(output_image)

# 用绿色线条绘制路径
for i in range(1, len(path)):
    draw.line((path[i - 1][1], path[i - 1][0], path[i][1], path[i][0]), fill=(0, 255, 0), width=2)

# 添加路径长度的文本（以毫米为单位）
font = ImageFont.load_default()
draw.text((10, 10), f"Length: {longest_path_length_mm:.2f} mm", fill=(0, 255, 0), font=font)

# 保存并显示输出图像
output_image_path = r'E:\A_workbench\A-lab\Unet\Unet_complit\data\marked_path.png'
output_image.save(output_image_path)  # 保存标记后的图像
output_image.show()  # 显示图像

# 输出路径信息
print(f"Longest path length: {longest_path_length_mm:.2f} mm")
print(f"Marked path saved to: {output_image_path}")
