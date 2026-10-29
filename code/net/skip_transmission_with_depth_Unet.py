from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from kornia.filters import gaussian_blur2d
import torch


def exists(x):
    """判断变量是否存在"""
    return x is not None

def Downsample(dim):
    """下采样模块，利用卷积和 ReLU实现"""
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )

def Upsample(dim):
    """上采样模块，利用转置卷积和 ReLU实现"""
    return nn.Sequential(
        nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=0),
        nn.ReLU()
    )

class FCB(nn.Module):
    """频带分解与特征融合模块"""
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.ks = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.sigma_rate = 1
        self.params = nn.Parameter(torch.ones((4, 1), requires_grad=True))

    def forward(self, x):
        _, _, h, w = x.shape
        def get_valid_kernel_size(size):
            size = min(size, h, w)
            return max(3, size if size % 2 == 1 else size - 1)
        kernel_size1 = get_valid_kernel_size(self.ks)
        kernel_size2 = get_valid_kernel_size(self.ks * 2 - 1)
        kernel_size3 = get_valid_kernel_size(self.ks * 4 - 1)
        x1 = gaussian_blur2d(x, (kernel_size1, kernel_size1), (1 * self.sigma_rate, 1 * self.sigma_rate))
        x2 = gaussian_blur2d(x, (kernel_size2, kernel_size2), (2 * self.sigma_rate, 2 * self.sigma_rate))
        x3 = gaussian_blur2d(x, (kernel_size3, kernel_size3), (4 * self.sigma_rate, 4 * self.sigma_rate))
        R1 = x - x1
        R2 = x1 - x2
        R3 = x2 - x3
        R1 = F.interpolate(R1, size=(h, w), mode='bilinear', align_corners=False)
        R2 = F.interpolate(R2, size=(h, w), mode='bilinear', align_corners=False)
        R3 = F.interpolate(R3, size=(h, w), mode='bilinear', align_corners=False)
        R_cat = torch.cat([R1.unsqueeze(-1), R2.unsqueeze(-1), R3.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)
        if self.params.shape != (4, 1):
            self.params = nn.Parameter(torch.ones((4, 1), requires_grad=True).to(x.device))
        sum_ = torch.matmul(R_cat, self.params).squeeze(dim=-1)
        return sum_

class DynamicConvBlock(nn.Module):
    """支持任意尺寸特征处理的卷积块"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.GELU(),
            ChannelAttention(dim_out)
        )
    def forward(self, x):
        return self.proj(x)

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.fc(max).view(b, c, 1, 1)
        return x * self.sigmoid(avg + max)

class Residual(nn.Module):
    """残差模块，将输入与输出相加"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    """自定义层归一化"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    """先归一化后处理的模块封装"""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Block(nn.Module):
    """基本卷积块，包含卷积、归一化及激活"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """残差卷积块，无时间嵌入信息"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """线性注意力机制"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    """常规自注意力机制"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class ResidualBlockCnn(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockCnn, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)

class Mask_Unet(nn.Module):
    def __init__(self, channels=1, out_dim=3, dim=32, dim_mults=(1, 2), resnet_block_groups=8):
        super(Mask_Unet, self).__init__()
        self.channels = channels
        init_dim = dim
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.downs = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, groups=resnet_block_groups),
                ResnetBlock(dim_out, dim_out, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, groups=resnet_block_groups)

        self.ups = nn.ModuleList([])
        reversed_in_out = list(reversed(in_out))
        for idx, (dim_in, dim_out) in enumerate(reversed_in_out):
            is_last = idx == (len(reversed_in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, groups=resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.fcb = nn.ModuleList([FCB(dim) for dim in dims])

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x):
        orig_size = x.shape[-2:]
        x = self.init_conv(x)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
            skip = h.pop()
            skip = self.fcb[idx](skip)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x