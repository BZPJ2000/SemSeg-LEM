import torch
from model.UNet_Swin.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys  # 根据实际情况调整导入路径

# 实例化模型
model = SwinTransformerSys(img_size=224, num_classes=3)

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 创建一个随机输入
x = torch.randn(1, 3, 224, 224).to(device)

# 清空缓存并记录初始内存使用量
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device)
    start_memory = torch.cuda.memory_allocated(device)
else:
    import psutil
    import os
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss

# 在不计算梯度的情况下运行模型
with torch.no_grad():
    output = model(x)

# 查看输出形状
print('Output shape:', output.shape)

# 打印模型运行后消耗的内存
if device.type == 'cuda':
    end_memory = torch.cuda.memory_allocated(device)
    max_memory = torch.cuda.max_memory_allocated(device)
    print(f'模型运行前占用的内存: {start_memory / 1024 ** 2:.2f} MB')
    print(f'模型运行后占用的内存: {end_memory / 1024 ** 2:.2f} MB')
    print(f'模型运行过程中最大占用的内存: {max_memory / 1024 ** 2:.2f} MB')
else:
    end_memory = process.memory_info().rss
    print(f'模型运行前占用的内存: {start_memory / 1024 ** 2:.2f} MB')
    print(f'模型运行后占用的内存: {end_memory / 1024 ** 2:.2f} MB')
    print(f'模型运行过程中增加的内存: {(end_memory - start_memory) / 1024 ** 2:.2f} MB')
