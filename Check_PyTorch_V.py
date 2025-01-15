import torch
import sys
import platform
import numpy as np
import matplotlib
def print_system_info():
    # 打印 Python 版本
    print(f"Python 版本: {platform.python_version()}")

    # 打印 NumPy 版本
    print(f"NumPy 版本: {np.__version__}")

    # 打印 Matplotlib 版本
    print(f"Matplotlib 版本: {matplotlib.__version__}")

    # 打印 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"CUDA 可用: 是")
        print(f"CUDA 版本: {torch.version.cuda}")

        # 打印 GPU 的信息
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA 可用: 否")

    # 打印操作系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")


if __name__ == "__main__":
    print_system_info()