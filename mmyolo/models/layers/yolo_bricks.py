# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, MaxPool2d,
                      build_norm_layer)
from mmdet.models.layers.csp_layer import \
    DarknetBottleneck as MMDET_DarknetBottleneck
# note
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmdet.models.layers.se_layer import ChannelAttention
from mmpretrain.models.backbones.mobileone import MobileOneBlock
# note

import math
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from torch import Tensor
from mmyolo.registry import MODELS

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    MODELS.register_module(module=nn.SiLU, name='SiLU')
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=True):
            super().__init__()

        def forward(self, inputs) -> Tensor:
            return inputs * torch.sigmoid(inputs)

    MODELS.register_module(module=SiLU, name='SiLU')

#! add yolov5 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

#! add yolov5 
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation 默认silu

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
#! add yolov5 
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

#! add yolov5 
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X

#! add shuffleNet
def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

#! add GhostNetV2
class DFC_Attention(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, 1, act=act) # kxk 降维，group=1
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)# 5x5 的深度卷积 效果就不是很好 group=channels

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) #channel维度叠加

#! add GhostNetV1
class GhostMoudule(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, 1, act=act) # kxk 降维，group=1
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)# 5x5 的深度卷积 效果就不是很好 group=channels

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) #channel维度叠加
    
class GhostModuleV2(nn.Module): 
    def __init__(self, inp, oup, kernel_size=1,\
                 stage_id=0, dw_size=3, stride=1, act=None,mode='attn',args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()
        self.oup = oup
        self.stage_id = stage_id
        init_channels = math.ceil(oup / 2)  #! 里面自己变
        new_channels = init_channels*(2-1)

        if self.mode in ['original'] or stage_id<2:
            logger.info(f'--- stage:{stage_id}, original mode')
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                act(inplace=True) if act else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                act(inplace=True) if act else nn.Sequential(),
            )
        elif self.mode in ['attn'] or stage_id>=2: 
            logger.info(f'--- stage:{stage_id}, attention mode')
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                act(inplace=True) if act else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                act(inplace=True) if act else nn.Sequential(),
            )
            # -n dfc特有
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False), #1x1
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original'] or self.stage_id<2:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn'] or self.stage_id >=2:  
            # ? 1.全局平均池化进行特征下采样
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            # ? 2. 1x1 通道锁定，水平和垂直上计算卷积，最后最近邻插值还原
            # F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 
            # 上面的操作是 dfc_attention_output
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1) # 2.ghost_output

            return out[:,:self.oup,:,:] * F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 
    
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))

        self.c2 = c2

class MobileOneBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1,num_conv_branches=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.MoblieOneDepthSeprate = nn.Sequential(
            # num_convs=num_conv_branches 指的分支
            # dw+pw  cba还没传进去
            MobileOneBlock(c1,c1,3,num_convs=num_conv_branches,\
                           stride=1,padding=1,groups=c1),#dw
            MobileOneBlock(c1,c2,1,num_convs=num_conv_branches,\
                           stride=1,padding=0,groups=1)#pw
            )
    def forward(self, x):
        #+ self.shortcut(x)
        return self.MoblieOneDepthSeprate(x)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, e=0.5,
                 k=3, pad=(1,0),s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        #stride=1 or 2 is different
        self.conv = nn.Sequential(
            GhostMoudule(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostMoudule(c_, c2, 1, 1, act=False))  # pw-linear
        
        self.shortcut = nn.Sequential( #深度可分离做 shortcut
            DWConv(c1, c1, k, s, act=False),
            Conv(c1, c2, 1, 1,act=False)) if s == 2 else nn.Identity()
    #stride foerver 1
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    
class GhostBottleneckV2(nn.Module): 
    def __init__(self, in_chs, out_chs, e=1, dw_kernel_size=3, has_dfc=False,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=0,args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        mid_chs = in_chs // e
        logger.info(f'Ghost BottleneckV2 cin:{in_chs} cout:{mid_chs}')

        # Point-wise expansion
        ghost1_dws, ghost2_dws=3,3
        logger.info(f'GBNv2 ghost1_Cheap_Operation:{ghost1_dws}x{ghost1_dws}')

        #! close dfc
        if has_dfc == False:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, dw_size=ghost1_dws, stage_id=layer_id, \
                            act=act_layer, mode='original',args=args)
        else:
        #! 第1个ghost bn+relu
            if layer_id<=1 or layer_id==None:   # layer_id：0-1 都是采用原始方式
                self.ghost1 = GhostModuleV2(in_chs, mid_chs, dw_size=ghost1_dws, stage_id=layer_id, \
                                            act=act_layer, mode='original',args=args)
            else:                               # layer_id: 2-3 开始加强
                logger.info('ghost1 implementing DFC Attention')
                self.ghost1 = GhostModuleV2(in_chs, mid_chs, dw_size=ghost1_dws, stage_id=layer_id, \
                                            act=act_layer, mode='attn',args=args) 

        # Depth-wise convolution\
        # If > 1
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        # -n ghost2 默认这样
        # ghost2_dws=3 # | 7.922M  11.496G  30.513M    3+3
        # ghost2_dws=5 # | 7.929M  11.518G  30.513M    3+5 ;5+3 都一样
        #5+5  | 7.936M  | 11.54G   | 30.513M   |
        logger.info(f'GBNv2 ghost2_Cheap_Operation:{ghost2_dws}x{ghost2_dws}')
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, dw_size=ghost2_dws, act=None, mode='original',args=args) 
        
        # shortcut
        #todo 1.stride = 1 直连结构
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential() 
        else:
        #todo 2.stride = 2 skip DW 5X5
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def __str__(self):
        return 'GhostBottleneck-v2'
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1: #如果 stride == 2
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

# note end
class DarknetBottleneck(MMDET_DarknetBottleneck):
    """The basic bottleneck block used in Darknet. 再封装一下
    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of k1Xk1 and the second one has the
    filter size of k2Xk2.
    """
#cin cout exp kz pad add_shorcut, usedp ,convcfg,normcfg act_cfg,init_cfg
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2
        # 稍微改写一下 改动写在这也可以
        # TODO 1X1 + 3X3
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels
    def __str__(self):
        return 'DarkNet-Bottlenecck'
        
class MobileOneBottleneckMM(DarknetBottleneck):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = False,
                 use_depthwise: bool = False,
                 num_branchs: int = 1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Relu', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        # num_branchs=1
        print(f'MobileONe in branch: {num_branchs}')
        self.num_branchs = num_branchs
        print(f'MobileONe branch: {self.num_branchs}')
        self.MoblieOneDepthSeprate = nn.Sequential(
            # num_convs = num_conv_branches 指的分支
            # dw+pw  cba还没传进去
            MobileOneBlock(in_channels, hidden_channels,3, #dw 降维
                                num_convs=self.num_branchs,\
                                stride=1,padding=1,groups=in_channels), #group太大 退化到深度卷积
            MobileOneBlock(hidden_channels,out_channels,1,
                                num_convs=self.num_branchs,\
                                stride=1,padding=0,groups=1) #pw 通道
            )

        self.add_identity = \
            add_identity and in_channels == out_channels
        
    def forward(self, x):
        identity = x
        out = self.MoblieOneDepthSeprate(x)

        if self.add_identity:
            return out + identity
        else:
            return out
        

@MODELS.register_module()
class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = MODELS.build(act_cfg)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


@MODELS.register_module()
class BepC3StageBlock(nn.Module):
    """Beer-mug RepC3 Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        num_blocks (int): Number of blocks. Defaults to 1
        hidden_ratio (float): Hidden channel expansion.
            Default: 0.5
        concat_all_layer (bool): Concat all layer when forward calculate.
            Default: True
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 hidden_ratio: float = 0.5,
                 concat_all_layer: bool = True,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        hidden_channels = int(out_channels * hidden_ratio)

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            2 * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.block = RepStageBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_blocks,
            block_cfg=block_cfg,
            bottle_block=BottleRep)
        self.concat_all_layer = concat_all_layer
        if not concat_all_layer:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x):
        if self.concat_all_layer is True:
            return self.conv3(
                torch.cat((self.block(self.conv1(x)), self.conv2(x)), dim=1))
        else:
            return self.conv3(self.block(self.conv1(x)))


class BottleRep(nn.Module):
    """Bottle Rep Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        adaptive_weight (bool): Add adaptive_weight when forward calculate.
            Defaults False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 adaptive_weight: bool = False):
        super().__init__()
        conv1_cfg = block_cfg.copy()
        conv2_cfg = block_cfg.copy()

        conv1_cfg.update(
            dict(in_channels=in_channels, out_channels=out_channels))
        conv2_cfg.update(
            dict(in_channels=out_channels, out_channels=out_channels))

        self.conv1 = MODELS.build(conv1_cfg)
        self.conv2 = MODELS.build(conv2_cfg)

        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if adaptive_weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


@MODELS.register_module()
class ConvWrapper(nn.Module):
    """Wrapper for normal Conv with SiLU activation.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): Conv bias. Default: True.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = dict(type='SiLU')):
        super().__init__()
        self.block = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2, act_cfg=dict(type='ReLU')):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvModule(c, c_mid, kernel_size=1,act_cfg=act_cfg) #silu relu
        #!注意 padding
        self.enc = ConvModule(c_mid, (scale*k_up)**2,padding=1,kernel_size=k_enc) 
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale #认证图尺寸
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                     # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_
        # X = X.view(b, -1, c, h_, w_) 

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X

@MODELS.register_module()
class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation.

    From `CenterMask : Real-Time Anchor-Free Instance Segmentation`
    arxiv (https://arxiv.org/abs/1911.06667)
    This code referenced to
    https://github.com/youngwanLEE/CenterMask/blob/72147e8aae673fcaf4103ee90a6a6b73863e7fa1/maskrcnn_benchmark/modeling/backbone/vovnet.py#L108-L121  # noqa

    Args:
        channels (int): The input and output channels of this Module.
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='HSigmoid').
    """

    def __init__(self,
                 channels: int,
                 act_cfg: ConfigType = dict(type='HSigmoid')):
        super().__init__()
        assert isinstance(act_cfg, dict)
        self.fc = ConvModule(channels, channels, 1, act_cfg=None)

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = MODELS.build(act_cfg_)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.activate(x_se)


class PPYOLOESELayer(nn.Module):
    """Squeeze-and-Excitation Attention Module for PPYOLOE.
        There are some differences between the current implementation and
        SELayer in mmdet:
            1. For fast speed and avoiding double inference in ppyoloe,
               use `F.adaptive_avg_pool2d` before PPYOLOESELayer.
            2. Special ways to init weights.
            3. Different convolution order.

    Args:
        feat_channels (int): The input (and output) channels of the SE layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 feat_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvModule(
            feat_channels,
            feat_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self._init_weights()

    def _init_weights(self):
        """Init weights."""
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat: Tensor, avg_feat: Tensor) -> Tensor:
        """Forward process
         Args:
             feat (Tensor): The input tensor.
             avg_feat (Tensor): Average pooling feature tensor.
         """
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


@MODELS.register_module()
class ELANBlock(BaseModule):
    """Efficient layer aggregation networks for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        middle_ratio (float): The scaling ratio of the middle layer
            based on the in_channels.
        block_ratio (float): The scaling ratio of the block layer
            based on the in_channels.
        num_blocks (int): The number of blocks in the main branch.
            Defaults to 2.
        num_convs_in_block (int): The number of convs pre block.
            Defaults to 1.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 middle_ratio: float,
                 block_ratio: float,
                 num_blocks: int = 2,
                 num_convs_in_block: int = 1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        assert num_blocks >= 1
        assert num_convs_in_block >= 1

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * block_ratio)
        final_conv_in_channels = int(
            num_blocks * block_channels) + 2 * middle_channels

        self.main_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.short_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_convs_in_block == 1:
                internal_block = ConvModule(
                    middle_channels,
                    block_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            else:
                internal_block = []
                for _ in range(num_convs_in_block):
                    internal_block.append(
                        ConvModule(
                            middle_channels,
                            block_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg))
                    middle_channels = block_channels
                internal_block = nn.Sequential(*internal_block)

            middle_channels = block_channels
            self.blocks.append(internal_block)

        self.final_conv = ConvModule(
            final_conv_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.final_conv(x_final)


@MODELS.register_module()
class EELANBlock(BaseModule):
    """Expand efficient layer aggregation networks for YOLOv7.

    Args:
        num_elan_block (int): The number of ELANBlock.
    """

    def __init__(self, num_elan_block: int, **kwargs):
        super().__init__()
        assert num_elan_block >= 1
        self.e_elan_blocks = nn.ModuleList()
        for _ in range(num_elan_block):
            self.e_elan_blocks.append(ELANBlock(**kwargs))

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for elan_blocks in self.e_elan_blocks:
            outs.append(elan_blocks(x))
        return sum(outs)


class MaxPoolAndStrideConvBlock(BaseModule):
    """Max pooling and stride conv layer for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        maxpool_kernel_sizes (int): kernel sizes of pooling layers.
            Defaults to 2.
        use_in_channels_of_middle (bool): Whether to calculate middle channels
            based on in_channels. Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 maxpool_kernel_sizes: int = 2,
                 use_in_channels_of_middle: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        middle_channels = in_channels if use_in_channels_of_middle \
            else out_channels // 2

        self.maxpool_branches = nn.Sequential(
            MaxPool2d(
                kernel_size=maxpool_kernel_sizes, stride=maxpool_kernel_sizes),
            ConvModule(
                in_channels,
                out_channels // 2,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.stride_conv_branches = nn.Sequential(
            ConvModule(
                in_channels,
                middle_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                middle_channels,
                out_channels // 2,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        maxpool_out = self.maxpool_branches(x)
        stride_conv_out = self.stride_conv_branches(x)
        return torch.cat([stride_conv_out, maxpool_out], dim=1)


@MODELS.register_module()
class TinyDownSampleBlock(BaseModule):
    """Down sample layer for YOLOv7-tiny.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        middle_ratio (float): The scaling ratio of the middle layer
            based on the in_channels. Defaults to 1.0.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
             sizes of pooling layers. Defaults to 3.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            middle_ratio: float = 1.0,
            kernel_sizes: Union[int, Sequence[int]] = 3,
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='LeakyReLU', negative_slope=0.1),
            init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        middle_channels = int(in_channels * middle_ratio)

        self.short_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.main_convs = nn.ModuleList()
        for i in range(3):
            if i == 0:
                self.main_convs.append(
                    ConvModule(
                        in_channels,
                        middle_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.main_convs.append(
                    ConvModule(
                        middle_channels,
                        middle_channels,
                        kernel_sizes,
                        padding=(kernel_sizes - 1) // 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        self.final_conv = ConvModule(
            middle_channels * 4,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x) -> Tensor:
        short_out = self.short_conv(x)

        main_outs = []
        for main_conv in self.main_convs:
            main_out = main_conv(x)
            main_outs.append(main_out)
            x = main_out

        return self.final_conv(torch.cat([*main_outs[::-1], short_out], dim=1))

# last level
class SPPFBottleneck(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True), #note 默认是 silu 
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        # todo 1. int 整数
        if isinstance(kernel_sizes, int): 
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        # todo 2. list 列表
        else: 
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x
    
@MODELS.register_module()
class SPPFCSPBlock(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer with CSP for
     YOLOv7

     Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         expand_ratio (float): Expand ratio of SPPCSPBlock.
            Defaults to 0.5.
         kernel_sizes (int, tuple[int]): Sequential or number of kernel
             sizes of pooling layers. Defaults to 5.
         is_tiny_version (bool): Is tiny version of SPPFCSPBlock. If True,
            it means it is a yolov7 tiny model. Defaults to False.
         conv_cfg (dict): Config dict for convolution layer. Defaults to None.
             which means using conv2d. Defaults to None.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.03, eps=0.001).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         init_cfg (dict or list[dict], optional): Initialization config dict.
             Defaults to None.
     """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 is_tiny_version: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.is_tiny_version = is_tiny_version

        mid_channels = int(2 * out_channels * expand_ratio)

        if is_tiny_version:
            self.main_layers = ConvModule(
                in_channels,
                mid_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.main_layers = nn.Sequential(
                ConvModule(
                    in_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])

        if is_tiny_version:
            self.fuse_layers = ConvModule(
                4 * mid_channels,
                mid_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.fuse_layers = nn.Sequential(
                ConvModule(
                    4 * mid_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.short_layer = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x1 = self.main_layers(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x1)
            y2 = self.poolings(y1)
            concat_list = [x1] + [y1, y2, self.poolings(y2)]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))
        else:
            concat_list = [x1] + [m(x1) for m in self.poolings]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))

        x2 = self.short_layer(x)
        return self.final_conv(torch.cat((x1, x2), dim=1))


class ImplicitA(nn.Module):
    """Implicit add layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 0.
        std (float): Std value of implicit module. Defaults to 0.02
    """

    def __init__(self, in_channels: int, mean: float = 0., std: float = .02):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplier layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 1.
        std (float): Std value of implicit module. Defaults to 0.02.
    """

    def __init__(self, in_channels: int, mean: float = 1., std: float = .02):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit * x


@MODELS.register_module()
class PPYOLOEBasicBlock(nn.Module):
    """PPYOLOE Backbone BasicBlock.

    Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         shortcut (bool): Whether to add inputs and outputs together
         at the end of this layer. Defaults to True.
         use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 shortcut: bool = True,
                 use_alpha: bool = False):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(
            out_channels,
            out_channels,
            use_alpha=use_alpha,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResLayer(nn.Module):
    """PPYOLOE Backbone Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of blocks in this stage.
        block_cfg (dict): Config dict for block. Default config is
            suitable for PPYOLOE+ backbone. And in PPYOLOE neck,
            block_cfg is set to dict(type='PPYOLOEBasicBlock',
            shortcut=False, use_alpha=False). Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True).
        stride (int): Stride of the convolution. In backbone, the stride
            must be set to 2. In neck, the stride must be set to 1.
            Defaults to 1.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict, optional): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        use_spp (bool): Whether to use `SPPFBottleneck` layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_block: int,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
                 stride: int = 1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 attention_cfg: OptMultiConfig = dict(
                     type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
                 use_spp: bool = False):
        super().__init__()

        self.num_block = num_block
        self.block_cfg = block_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_spp = use_spp
        assert attention_cfg is None or isinstance(attention_cfg, dict)

        if stride == 2:
            # todo 如果是降维的stage
            conv1_in_channels = conv2_in_channels = conv3_in_channels = (
                in_channels + out_channels) // 2
            blocks_channels = conv1_in_channels // 2
            self.conv_down = ConvModule(
                in_channels,
                conv1_in_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            conv1_in_channels = conv2_in_channels = in_channels
            conv3_in_channels = out_channels
            blocks_channels = out_channels // 2
            self.conv_down = None

        self.conv1 = ConvModule(
            conv1_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            conv2_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = self.build_blocks_layer(blocks_channels)

        self.conv3 = ConvModule(
            conv3_in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if attention_cfg:
            attention_cfg = attention_cfg.copy()
            attention_cfg['channels'] = blocks_channels * 2
            self.attn = MODELS.build(attention_cfg)
        else:
            self.attn = None

    def build_blocks_layer(self, blocks_channels: int) -> nn.Module:
        """Build blocks layer.

        Args:
            blocks_channels: The channels of this Module.
        """
        blocks = nn.Sequential()
        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(in_channels=blocks_channels, out_channels=blocks_channels))
        block_cfg.setdefault('norm_cfg', self.norm_cfg)
        block_cfg.setdefault('act_cfg', self.act_cfg)

        for i in range(self.num_block):
            blocks.add_module(str(i), MODELS.build(block_cfg))

            if i == (self.num_block - 1) // 2 and self.use_spp:
                blocks.add_module(
                    'spp',
                    SPPFBottleneck(
                        blocks_channels,
                        blocks_channels,
                        kernel_sizes=[5, 9, 13],
                        use_conv_first=False,
                        conv_cfg=None,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))

        return blocks

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


@MODELS.register_module()
class RepStageBlock(nn.Module):
    """RepStageBlock is a stage block with rep-style basic block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_blocks (int, tuple[int]): Number of blocks.  Defaults to 1.
        bottle_block (nn.Module): Basic unit of RepStage.
            Defaults to RepVGGBlock.
        block_cfg (ConfigType): Config of RepStage.
            Defaults to 'RepVGGBlock'.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 bottle_block: nn.Module = RepVGGBlock,
                 block_cfg: ConfigType = dict(type='RepVGGBlock')):
        super().__init__()
        block_cfg = block_cfg.copy()

        block_cfg.update(
            dict(in_channels=in_channels, out_channels=out_channels))

        self.conv1 = MODELS.build(block_cfg)

        block_cfg.update(
            dict(in_channels=out_channels, out_channels=out_channels))

        self.block = None
        if num_blocks > 1:
            self.block = nn.Sequential(*(MODELS.build(block_cfg)
                                         for _ in range(num_blocks - 1)))

        if bottle_block == BottleRep:
            self.conv1 = BottleRep(
                in_channels,
                out_channels,
                block_cfg=block_cfg,
                adaptive_weight=True)
            num_blocks = num_blocks // 2
            self.block = None
            if num_blocks > 1:
                self.block = nn.Sequential(*(BottleRep(
                    out_channels,
                    out_channels,
                    block_cfg=block_cfg,
                    adaptive_weight=True) for _ in range(num_blocks - 1)))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class CSPLayerWithTwoConv(BaseModule):
    """Cross Stage Partial Layer with 2 convolutions.
    # c2f

    Args:
        in_channels (int): The input channels of the CSP layer .
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


bottleneck = [GhostBottleneck,GhostBottleneckV2,DarknetBottleneck,
        CSPNeXtBlock,MobileOneBottleneckMM,]  

class Gghost_bottom_up(BaseModule):
    """Cross Stage Partial Layer. with g-ghost mix
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 1,#note :这里无
                 stage_id: int =0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 1,
                 num_branchs: int =1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck 
        # 根据索引来制定
        block = bottleneck[use_custom_block]
        print(f'Gghost_bottom_up: Stage:{stage_id}, CHOICE {use_custom_block}',block)
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        cheap_ratio = 0.5 # 50% 去生成
        self.raw_channels = int(mid_channels * (1 - cheap_ratio))
        self.cheap_channels = mid_channels - self.raw_channels
        self.mid_channels = mid_channels
        raw_channels = self.raw_channels
        cheap_channels = self.cheap_channels
        # print(f'AFTER {self.num_branchs}')

        self.main_conv = ConvModule( # part 1
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            out_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        #!add
        # 0: ghost == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=raw_channels,
                    c2=raw_channels,) for _ in range(num_blocks)
            ])
        # 1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    raw_channels,
                    raw_channels, # 通道数减少 MAC下降
                    dw_kernel_size=5, #因为通道数不变 这个参数没用
                    act_layer= nn.SiLU,
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                DarknetBottleneck( #堆叠bottleneck提高深度
                    raw_channels,
                    raw_channels, # 通道数减少 MAC下降
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])

        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        #! add
        # 3. mix operation 
        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels + raw_channels*(num_blocks), cheap_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_channels, cheap_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_channels),
        ) 
        self.cheap = nn.Sequential(
        # 4 这里的 cheap操作区别c-ghost 的 3x3 dw  这里是 1x1 pw
                    nn.Conv2d(cheap_channels, cheap_channels,
                            kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(cheap_channels),
            )
        
        self.cheap_relu = nn.ReLU(inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        # 1 直接降通道
        x_main = self.main_conv(x) # 
        # 3 custom
        # bottleneck(block)堆叠 包含的操作[先1x1 再3x3]
        m_list=[x_main]
        e = x_main[:,:self.raw_channels]
        for l in self.blocks:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list, 1)
        m = self.merge(m)
        # cheap_shortcut
        c = x_main[:, self.raw_channels:]
        c = self.cheap_relu(self.cheap(c)+m) # short & 中间特征图
        x_main = e # 3 csp_part2: raw

        x_final = torch.cat((x_main, c), dim=1) #合并
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) 
    
class CSPLayerG(BaseModule):
    """Cross Stage Partial Layer. with g-ghost mix
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 stage_id: int =0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 0,
                 num_branchs: int =1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层
        block = bottleneck[use_custom_block]
        print(f'Stage:{stage_id}, CHOICE {use_custom_block}', block)
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        # todo 一样 part 1
        self.main_conv = ConvModule( 
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        #! 开始不一样
        self.short_conv = ConvModule( # part 2
            mid_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels, # mid x 2 = out
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # ! add
        cheap_ratio = 0.5 # 50% 去生成
        self.raw_channels = int(mid_channels * (1 - cheap_ratio))
        self.cheap_channels = in_channels - self.raw_channels
        self.mid_channels = mid_channels
        raw_channels = self.raw_channels
        cheap_channels = self.cheap_channels

        # 0: ghost == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=mid_channels,
                    c2=mid_channels,) for _ in range(num_blocks)
            ])
        # 1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    # mid_channels,
                    # mid_channels,
                    raw_channels,
                    raw_channels, # 通道数减少 MAC下降
                    dw_kernel_size=5, #因为通道数不变 这个参数没用
                    act_layer= nn.SiLU,
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                DarknetBottleneck( #堆叠bottleneck提高深度
                    raw_channels,
                    raw_channels, # 通道数减少 MAC下降
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])

        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        #! add
        # 3. mix operation 

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels + raw_channels*(num_blocks), cheap_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_channels, cheap_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_channels),
        ) 
        self.cheap = nn.Sequential(
        # 4 这里的 cheap操作区别c-ghost 的 3x3 dw  这里是 1x1 pw
                    nn.Conv2d(cheap_channels, cheap_channels,
                            kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(cheap_channels),
            )
        
        self.cheap_relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        x_main = self.main_conv(x) # 1 csp_part1
        # 3 custom
        # bottleneck(block)堆叠 包含的操作[先1x1 再3x3]
        m_list=[x_main]
        e = x_main[:,:self.raw_channels]
        for l in self.blocks:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list, 1)
        m = self.merge(m)
        c = x_main[:, self.raw_channels:]
        c = self.cheap_relu(self.cheap(c)+m) # 2 csp_part2: cheap
        x_main = e # 3 csp_part2: raw

        x_final = torch.cat((x_main,c, x_short), dim=1) #channelx2
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) #channelx2

#old-version
class CSPLayer1(BaseModule):
    """ 自定义的CSP层  与YOLOV5 的C3结构一致
       bottleneck = [GhostBottleneck,GhostBottleneckV2,DarknetBottleneck,
              CSPNeXtBlock,MobileOneBottleneckMM,]  
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 stage_id: int = 0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 0,
                 use_dfc_attention: bool = True,
                 num_branchs: int = 1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck 
        block = bottleneck[use_custom_block]#根据索引来制定
        from loguru import logger
        logger.info(f'stage:{stage_id+1},bottleNeck:{block}')
        mid_channels = int(out_channels * expand_ratio)

        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        self.use_dfc_attention = use_dfc_attention
        #! 与 C3 一样
        self.main_conv = ConvModule( # part 1
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule( # part 2
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule( # part 3
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        logger.info(f'num_blocks:{num_blocks}')

        # ? 0: ghostV1 == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=mid_channels,
                    c2=mid_channels,) for _ in range(num_blocks)
            ])
        # ? 1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    e=1,
                    dw_kernel_size=3,
                    has_dfc=self.use_dfc_attention, #默认是开启
                    act_layer= nn.SiLU, #激活函数
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        elif use_custom_block==3:
            self.blocks = nn.Sequential(*[
                MobileOneBottleneckMM( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    num_branchs=self.num_branchs,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                block( # todo 堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        #! 1 跳连
        x_short = self.short_conv(x)
        #! 2 卷一下
        x_main = self.main_conv(x)
        #! 3 custom bottleneck(block)堆叠 包含的操作[先1x1 再3x3]
        x_main = self.blocks(x_main) 
        x_final = torch.cat((x_main, x_short), dim=1) #channelx2
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) #channelx2
    
class CSPLayerCustom(BaseModule):
    """ 自定义的CSP层  与YOLOV5 的C3结构一致
       bottleneck = [GhostBottleneck,GhostBottleneckV2,DarknetBottleneck,
              CSPNeXtBlock,MobileOneBottleneckMM,]  
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 stage_id: int = 0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 0,
                 use_dfc_attention: bool = True,
                 num_branchs: int = 1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck 
        block = bottleneck[use_custom_block]#根据索引来制定
        from loguru import logger
        logger.info(f'stage:{stage_id+1},bottleNeck:{block}')
        mid_channels = int(out_channels * expand_ratio)

        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        self.use_dfc_attention = use_dfc_attention
        #! 与 C3 一样
        self.main_conv = ConvModule( # part 1
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule( # part 2
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule( # part 3
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        logger.info(f'num_blocks:{num_blocks}')

        # ? 0: ghostV1 == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=mid_channels,
                    c2=mid_channels,) for _ in range(num_blocks)
            ])
        # ? 1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    e=1,
                    dw_kernel_size=3,
                    has_dfc=self.use_dfc_attention, #默认是开启
                    act_layer= nn.SiLU, #激活函数
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        elif use_custom_block==3:
            self.blocks = nn.Sequential(*[
                MobileOneBottleneckMM( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    num_branchs=self.num_branchs,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                block( # todo 堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        #! 1 跳连
        x_short = self.short_conv(x)
        #! 2 卷一下
        x_main = self.main_conv(x)
        #! 3 custom bottleneck(block)堆叠 包含的操作[先1x1 再3x3]
        x_main = self.blocks(x_main) 
        x_final = torch.cat((x_main, x_short), dim=1) #channelx2
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) #channelx2

class CSPLayer_Ghostv2_conv(BaseModule):
    """Cross Stage Partial Layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 stage_id: int = 0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 2,#默认darknet
                 num_branchs: int = 1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层 
        block = bottleneck[use_custom_block]
        #根据索引来制定
        logger.info(f'Stage:{stage_id+1}, CHOICE {use_custom_block}', block)
        mid_channels = int(out_channels * expand_ratio)

        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        # 与C3 一样
        self.main_conv = GhostModuleV2( # ! part 1
            in_channels,
            mid_channels,
            1,
            stage_id=stage_id,
)
        self.short_conv = GhostModuleV2( # ! part 2
            in_channels,
            mid_channels,
            1,
            stage_id=stage_id
)
        self.final_conv = GhostModuleV2( # ! part 3
            2 * mid_channels,
            out_channels,
            1,
            stage_id=stage_id
)
        # ?0: ghostV1 == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=mid_channels,
                    c2=mid_channels,) for _ in range(num_blocks)
            ])
        # ?1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    e=1,
                    has_dfc = self.use_dfc_attentiion,
                    dw_kernel_size=5,
                    act_layer= nn.SiLU,
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                block( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)


    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        # 3 custom
        # bottleneck(block)堆叠 包含的操作[先1x1 再3x3]
        x_main = self.blocks(x_main) 
        x_final = torch.cat((x_main, x_short), dim=1) #channelx2
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) #channelx2

class CSPLayer_Ghostv2_split_channel(BaseModule):
    """Cross Stage Partial Layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 stage_id: int = 0,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 use_custom_block: int = 2,#默认darknet
                 num_branchs: int = 1,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # standard bottleneck 和 NextBlock 有一个大卷积层 
        block = bottleneck[use_custom_block]
        #根据索引来制定
        from loguru import logger
        logger.info(f'Stage:{stage_id+1}, CHOICE {use_custom_block}',block)
        mid_channels = int(out_channels * expand_ratio)

        self.channel_attention = channel_attention
        self.num_branchs = num_branchs
        self.mid_channels = mid_channels

        self.short_conv = ConvModule( # ! part 2
            mid_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.final_conv = GhostModuleV2( # ! part 3
            2 * mid_channels,
            out_channels,
            1,
            stage_id=stage_id, #好加attention
        )
        # ? 0: ghostV1 == silu + 5x5dw
        if use_custom_block==0:
            self.blocks = nn.Sequential(*[
                GhostBottleneck( #堆叠bottleneck提高深度
                    c1=mid_channels,
                    c2=mid_channels,) for _ in range(num_blocks)
            ])
        # ? 1: ghostv2 == silu + 5x5dw
        elif use_custom_block==1:
            self.blocks = nn.Sequential(*[
                GhostBottleneckV2( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    e=1,
                    dw_kernel_size=5, #
                    act_layer= nn.SiLU,
                    layer_id=stage_id) for _ in range(num_blocks)
            ])
        else:# 1:默认
            self.blocks = nn.Sequential(*[
                block( #堆叠bottleneck提高深度
                    mid_channels,
                    mid_channels,
                    1.0,
                    (1,3),
                    (0,1),
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_blocks = x[:, :self.mid_channels]
        x_cheap = x[:, self.mid_channels:]
        x_main = self.blocks(x_blocks)
        x_short = self.short_conv(x_cheap)
        
        x_final = torch.cat((x_main, x_short), dim=1) #channelx2
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final) #channelx2
     
class  MobileOneBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1,num_conv_branches=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.MoblieOneDepthSeprate = nn.Sequential(
            MobileOneBlock(c1,c1,3,num_convs=num_conv_branches,\
                           stride=1,padding=1,groups=c1),#dw
            MobileOneBlock(c1,c2,1,num_convs=num_conv_branches,\
                           stride=1,padding=0,groups=1)#pw
            )
    def forward(self, x):
        return self.MoblieOneDepthSeprate(x)
## ---------- speed test
import torch, time, math, thop, tqdm, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from prettytable import PrettyTable

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

if __name__ == '__main__':
    warmup, test_times = 1000, 3000
    bs, h, w = 8, 128, 128
    inc, ouc, kernel_size = 128, 128, 3
    cuda, half = True, True
    module_list = [
                    GhostBottleneckV2(inc,ouc,1),
                    DarknetBottleneck(inc,ouc,1.0),
                   ]
    
    device = torch.device("cuda:0") if cuda else torch.device("cpu")
    # device = torch.device("cpu")
    inputs = torch.randn((bs, inc, h, w)).to(device)
    if half:
        inputs = inputs.half()
    table = PrettyTable()
    table.title = 'Conv Family Speed'
    table.field_names = ['Name', 'All_Time', 'Mean_Time', 'FPS', "FLOPs", "Params"]
    for module in module_list:
        module = module.to(device)
        if half:
            module = module.half()
        for i in tqdm.tqdm(range(warmup), desc=f'{str(module)} Warmup....'):
            module(inputs)
        all_time = 0
        for i in tqdm.tqdm(range(test_times), desc=f'{str(module)} Calculate Speed....'):
            begin = time_synchronized()
            module(inputs)
            all_time += time_synchronized() - begin
        FLOPs, Params = thop.profile(module, inputs=(inputs, ), verbose=False)
        FLOPs, Params = thop.clever_format([FLOPs, Params], "%.3f")
        # print(f'{str(module)} all_time:{all_time:.5f} mean_time:{all_time / test_times:.5f} fps:{1 / (all_time / test_times)} FLOPs:{FLOPs} Params:{Params}')
        table.add_row([str(module), f'{all_time:.5f}', f'{all_time / test_times:.5f}', f'{1 / (all_time / test_times)}', f'{FLOPs}', f'{Params}'])
    print(table)