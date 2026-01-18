import torch
from DATransUNet import DA_Transformer as ViT_seg
from DATransUNet import CONFIGS as CONFIGS_ViT_seg

# 配置模型参数
vit_name = 'R50-ViT-B_16'  # 模型名称，可以根据实际情况修改
img_size = 224             # 输入图像大小
num_classes = 3            # 分类类别数（例如对于医学分割任务）
vit_patches_size = 16      # ViT patch大小

# 加载模型配置
config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = 3  # Skip-connection的数目，可以根据原始代码修改

# 对于ResNet基础的ViT模型，设置grid大小
if vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

# 创建模型
model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()

# 打印模型结构
print(model)

# 随机生成一个输入张量（批次大小为1，通道数为3，图像大小为224x224）
input_data = torch.randn(1, 3, img_size, img_size).cuda()

# 执行前向传播
output = model(input_data)

# 打印输出形状，确保输出维度正确
print(f"Output shape: {output.shape}")
