# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer
#note
from mmyolo.models.layers.yolo_bricks import CSPLayerCustom, CSPLayerG,\
    Gghost_bottom_up,CARAFE
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.ops.carafe import CARAFEPack
from mmyolo.registry import MODELS
from .base_yolo_neck import BaseYOLONeck
from loguru import logger


@MODELS.register_module()
class YOLOXPAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 use_depthwise: bool = False,
                 freeze_all: bool = False,
                 num_features: int = 3,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.use_depthwise = use_depthwise
        self.num_features = num_features
        self.light_p2 = 0 # vanlila
        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2: 
            # 1x1 卷积降通道 stride =1 reduce channel
            layer = ConvModule(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        # if idx == 1:
        #     #1.对齐 reduced输出
        #     return CARAFE(self.in_channels[idx-1] , act_cfg=self.act_cfg)
        #     # return CARAFEPack(self.in_channels[idx-1]*2,2)
        # if idx == 2:
        #     #2.对齐 reduced输出
        #     return CARAFE(self.in_channels[idx-2]*2,  act_cfg=self.act_cfg)
            # return CARAFEPack(self.in_channels[idx-2]*2,2)
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            return CSPLayer(
                self.in_channels[idx - 1] * 2,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif idx == 2:
            return nn.Sequential(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=self.num_csp_blocks,
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer. stride=2的普通卷积
        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The downsample layer.
        """
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        return conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer. 
         因为和低层concat,所以是两倍
        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(
            self.in_channels[idx] * 2,
            self.in_channels[idx + 1],
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return ConvModule(
            self.in_channels[idx],
            self.out_channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

@MODELS.register_module()
class YOLOXPAFPNMY(BaseYOLONeck):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 num_features: int = 3,
                 use_depthwise: bool = True,
                 freeze_all: bool = False,
                 bottom_up_mode: str = 'original',
                 upsample_mode: str ='original',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        # num_csp_blocks 这玩意控制
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.use_depthwise = use_depthwise
        # self.use_custom_block = use_custom_block # note
        self.bottom_up_mode = bottom_up_mode
        self.upsample_mode = upsample_mode
        print(f'csp_num_blocks:{self.num_csp_blocks}')
        print(f'自底向上bottom_up_mode:{self.bottom_up_mode}')
        print(f'upsample_mode:{self.upsample_mode}')
        # self.in_channels_new = in_channels.append(in_channels[-1])
        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        # note： 
        # 如果输出特征尺度等于4 开启P2
        self.num_features = num_features
        self.light_p2 = 0
        if self.num_features == 4 and len(in_channels)==3:
            if in_channels[0] == 256: #p3-p6
                self.light_p2 = 1#开启
                print('P2 Start')
        # self.p5_out_to_p6_top=ConvModule(self.out_channels, self.out_channels, kernel_size=3,\
        #                        stride=2,padding=1,act_cfg=act_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx, *args, **kwargs) -> nn.Module:
        """build upsample layer."""

        if self.upsample_mode == 'carafe':
            if idx == 1:
                #1.对齐 reduced输出
                return CARAFE(self.in_channels[idx-1] , act_cfg=self.act_cfg)
                # return CARAFEPack(self.in_channels[idx-1]*2,2)
            if idx == 2:
                #2.对齐 reduced输出
                return CARAFE(self.in_channels[idx-2]*2,  act_cfg=self.act_cfg)
        else:    
                return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        logger.info(f'=='*20)
        logger.info(f'building top_down_layer_{idx}')
        if idx == 1:
            return CSPLayer1(
                self.in_channels[idx - 1] * 2,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                use_custom_block= 1, # note
                stage_id=2, #dfc
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif idx == 2:
            return nn.Sequential(
                CSPLayer1(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1], # todo 通道数减半，特征图不变
                    num_blocks=self.num_csp_blocks,
                    use_custom_block= 1, # note
                    add_identity=False,
                    stage_id=2, #dfc
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule( #1x1 通道降维
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer. depth_seperate
        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The downsample layer.
        """
        # logger.info(f'=='*20)
        # logger.info(f'building downsample_layer_{idx}')
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        return conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        # CSPLayerG#
        logger.info(f'=='*20)
        logger.info(f'building bottom_up_layer_{idx}')
        if self.bottom_up_mode == 'original':
            return CSPLayer1(
                self.in_channels[idx] * 2,
                self.in_channels[idx + 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        
        elif self.bottom_up_mode == 'gbnv2':
            return CSPLayer1(
                self.in_channels[idx] * 2,
                self.in_channels[idx + 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                stage_id = 3, # 开启dfc
                use_custom_block=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else: #ghostv2
            return Gghost_bottom_up(
                self.in_channels[idx] * 2,
                self.in_channels[idx + 1],
                num_blocks=self.num_csp_blocks,
                use_custom_block= 1,
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return ConvModule(
            self.in_channels[idx],
            self.out_channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
