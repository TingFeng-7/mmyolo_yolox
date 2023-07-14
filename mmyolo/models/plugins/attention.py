# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


class ChannelAttention(BaseModule):
    """ChannelAttention.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self,
                 channels: int,
                 reduce_ratio: int = 16,
                 act_cfg: dict = dict(type='ReLU')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avgpool_out = self.fc(self.avg_pool(x))
        maxpool_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avgpool_out + maxpool_out)
        return out


class SpatialAttention(BaseModule):
    """SpatialAttention
    Args:
         kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=dict(type='Sigmoid'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    #-n x*sigmoid(硬件加速版)
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

@MODELS.register_module()
class CoordAtt(BaseModule):
    def __init__(self,
                 in_channels: int,
                 reduce_ratio: int = 32,
                 act_cfg: dict = dict(type='ReLU')):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduce_ratio)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
@MODELS.register_module()
class SEAttention(BaseModule):
    def __init__(self,
                 in_channels: int,
                 reduce_ratio: int = 16,
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduce_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduce_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)