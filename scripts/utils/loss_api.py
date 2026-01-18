import torch
from torch import nn
from losses.focal_loss import FocalLoss
# %%

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


# FocalLoss 和 CrossEntropyLoss
focal_loss = FocalLoss(apply_nonlin=softmax_helper)
cross_entropy_loss = nn.CrossEntropyLoss()
def focal_and_cross_entropy_loss(pred, target):
    fl_loss = focal_loss(pred, target)
    ce_loss = cross_entropy_loss(pred, target)
    return 0.5 * fl_loss + 0.5 * ce_loss  #0.5 权重可以调整

# %%

from losses.lovasz_loss import LovaszSoftmax
lovasz_loss = LovaszSoftmax(reduction='mean')

# %%
from losses.ND_Crossentropy import CrossentropyND

cross_entropy_loss = CrossentropyND()

# %%
from losses.focal_loss import FocalLoss
import torch.nn.functional as F

# 实例化 FocalLoss
focal_loss = FocalLoss(
    apply_nonlin=F.softmax,  # 根据您的模型输出选择合适的非线性函数
    gamma=2,                 # 聚焦参数 gamma，可根据需要调整
    alpha=None               # 类别权重 alpha，可设置为 None 或指定权重
)


# %%
from losses.dice_loss import SoftDiceLoss
import torch.nn.functional as F

# 实例化 SoftDiceLoss
soft_dice_loss = SoftDiceLoss(
    apply_nonlin=F.softmax,
    batch_dice=False,
    do_bg=True,
    smooth=1.0,
    square=False
)

# %%
from losses.dice_loss import TverskyLoss
import torch.nn.functional as F

# 实例化 TverskyLoss
tversky_loss = TverskyLoss(
    apply_nonlin=F.softmax,
    alpha=0.3,       # 控制假阳性的重要性
    beta=0.7,        # 控制假阴性的重要性
    smooth=1.0,
    do_bg=True,
    batch_dice=False,
    square=False
)

# %%
from losses.dice_loss import FocalTversky_loss
import torch.nn.functional as F

# 定义 TverskyLoss 参数
tversky_kwargs = {
    'apply_nonlin': F.softmax,
    'alpha': 0.3,
    'beta': 0.7,
    'smooth': 1.0,
    'do_bg': True,
    'batch_dice': False,
    'square': False
}

# 实例化 FocalTversky_loss
focal_tversky_loss = FocalTversky_loss(
    tversky_kwargs=tversky_kwargs,
    gamma=0.75
)

# %%
from losses.dice_loss import GDiceLoss
import torch.nn.functional as F

# 实例化 GDiceLoss
gdice_loss = GDiceLoss(
    apply_nonlin=F.softmax,
    smooth=1e-5
)
