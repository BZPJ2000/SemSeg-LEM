from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from Unet.Unet_complit.model.UNet_KAN.kan import KANLinear
import torch.nn.functional as F

class KANConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs
    ):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups

        assert self.groups == 1, "Grouped convolutions are not supported with KANConv2d."

        self.kan_linear = KANLinear(
            in_features=in_channels * self.kernel_size[0] * self.kernel_size[1],
            out_features=out_channels,
            **kwargs
        )

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x_unfolded = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        x_unfolded = x_unfolded.transpose(1, 2)
        x_patches = x_unfolded.reshape(
            -1, self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        outputs = self.kan_linear(x_patches)
        H_out = (
            (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        W_out = (
            (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
            + 1
        )
        outputs = outputs.view(batch_size, H_out * W_out, self.out_channels)
        outputs = outputs.transpose(1, 2).reshape(batch_size, self.out_channels, H_out, W_out)
        return outputs

# Modify ConvLayer to use KANConv2d
class KANConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(KANConvLayer, self).__init__()
        self.conv = nn.Sequential(
            KANConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            KANConv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

# Modify D_ConvLayer to use KANConv2d
class KAND_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(KAND_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            KANConv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            KANConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

# Modify OverlapPatchEmbed to use KANConv2d
class KANOverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding using KANConv2d
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, **kwargs):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = KANConv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
            **kwargs
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W