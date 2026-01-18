import sys

import torch
from PIL import Image
import re
from io import StringIO

from torchinfo import summary


def keep_image_size_open(path):
    img = Image.open(path)
    return img

def keep_image_size_open_rgb(path):
    img = Image.open(path).convert('RGB')
    return img









def check_memory_requirement(model, input_size, gpu_memory_available):
    # 将标准输出重定向到StringIO对象
    old_stdout = sys.stdout
    sys.stdout = summary_str_io = StringIO()

    # 获取模型的summary信息
    summary(model, input_size=input_size)

    # 恢复标准输出
    sys.stdout = old_stdout

    # 获取summary信息的字符串
    summary_str = summary_str_io.getvalue()

    # 提取summary中的内存使用信息
    input_size, fwd_bwd_size, params_size, total_size = 0, 0, 0, 0
    lines = summary_str.split('\n')
    for line in lines:
        if "Input size" in line:
            input_size = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        elif "Forward/backward pass size" in line:
            fwd_bwd_size = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        elif "Params size" in line:
            params_size = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        elif "Estimated Total Size" in line:
            total_size = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

    # 计算实际所需显存
    actual_mem_1_5 = total_size * 1.5
    actual_mem_2 = total_size * 2

    # 打印内存使用信息
    print(f"Estimated Total Size: {total_size} MB")
    print(f"Actual Memory Required (1.5x): {actual_mem_1_5} MB")
    print(f"Actual Memory Required (2x): {actual_mem_2} MB")

    # 检查显存是否足够
    if actual_mem_1_5 <= gpu_memory_available:
        print("显存足够 (1.5倍)")
    else:
        print("显存不足 (1.5倍)")

    if actual_mem_2 <= gpu_memory_available:
        print("显存足够 (2倍)")
    else:
        print("显存不足 (2倍)")

    return actual_mem_1_5, actual_mem_2

def test_input_size(model, max_size=(3384, 1710), step=32, batch_size=1):
    """
    检测模型允许输入的尺寸，并输出允许的尺寸列表。

    参数：
        model: 要测试的模型。
        max_size: 测试的最大尺寸，默认为(1024, 1024)。
        step: 尺寸递增的步长，默认为32。
        batch_size: 测试时的批量大小，默认为1。

    返回：
        允许输入的尺寸列表。
    """
    allowed_sizes = []
    for h in range(step, max_size[0] + 1, step):
        for w in range(step, max_size[1] + 1, step):
            try:
                # 创建测试输入数据
                input_data = torch.randn(batch_size, 3, h, w).cuda()
                # 尝试前向传播
                model(input_data)
                # 如果成功，记录尺寸
                allowed_sizes.append((h, w))
                print(f"Allowed size: ({h}, {w})")
            except RuntimeError as e:
                # 捕捉内存不足错误并继续
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    print(f"Size ({h}, {w}) is too large: {e}")
                else:
                    raise e
    return allowed_sizes










