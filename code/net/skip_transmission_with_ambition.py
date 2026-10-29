import torch
import torch.nn as nn
from .layers import bn, conv, act  # 确保这些函数与 skip 模块中一致


class SkipTransmissionWithAmb(nn.Module):
    def __init__(self,
                 noise_channels=8,  # 噪声输入通道数（8个噪声）
                 amb_channels=3,  # 深度图输入通道数（单通道）
                 num_output_channels=3,  # 传输图输出通常为单通道
                 num_channels_down=[8, 16, 32, 64, 128],
                 num_channels_up=[8, 16, 32, 64, 128],
                 num_channels_skip=[0, 0, 0, 4, 4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 need_sigmoid=True,
                 need_bias=True,
                 pad='reflection',
                 upsample_mode='bilinear',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 need1x1_up=True):
        super(SkipTransmissionWithAmb, self).__init__()

        # 针对深度图建立卷积分支，输出8通道
        self.amb_branch = nn.Sequential(
            nn.Conv2d(amb_channels, 8, kernel_size=3, padding=1, bias=need_bias),
            bn(8),
            act(act_fun),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=need_bias),
            bn(8),
            act(act_fun)
        )

        # 可学习的权重因子，用于调节深度图特征的贡献
        self.amb_weight = nn.Parameter(torch.ones(1) * 10)  # 初始值可以设为2，确保开始时深度信息有较大贡献

        # 后续 skip 网络
        from .skip_model import skip
        self.skip_net = skip(num_input_channels=noise_channels + 8,
                             num_output_channels=num_output_channels,
                             num_channels_down=num_channels_down,
                             num_channels_up=num_channels_up,
                             num_channels_skip=num_channels_skip,
                             filter_size_down=filter_size_down,
                             filter_size_up=filter_size_up,
                             filter_skip_size=filter_skip_size,
                             need_sigmoid=need_sigmoid,
                             need_bias=need_bias,
                             pad=pad,
                             upsample_mode=upsample_mode,
                             downsample_mode=downsample_mode,
                             act_fun=act_fun,
                             need1x1_up=need1x1_up)

    def forward(self, noise_input, amb_input):
        """
        参数:
            noise_input: 噪声特征，形状为 [B, 8, H, W]，包括8个噪声通道
            depth_input: 深度图，形状为 [B, 1, H, W]，单通道的深度图
        返回:
            预测的传输图，形状为 [B, num_output_channels, H, W]
        """
        # 提取深度图的8通道特征
        amb_feat = self.amb_branch(amb_input)  # 输出形状 [B, 8, H, W]

        # 将深度特征与噪声特征拼接
        fused_input = torch.cat([noise_input, self.amb_weight * amb_feat], dim=1)  # 形状 [B, 8 + 8, H, W]

        # 后续传入 skip 网络进行编码解码操作
        out = self.skip_net(fused_input)
        return out
