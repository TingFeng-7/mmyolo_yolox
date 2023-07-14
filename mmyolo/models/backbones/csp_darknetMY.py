# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
# from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.models.backbones.csp_darknet import Focus, CSPLayer
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayer_Ghostv2_split_channel, \
    SPPFBottleneck, CSPLayerCustom, CSPLayerG, CSPLayer_Ghostv2_conv,GhostModuleV2
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone
from loguru import logger


@MODELS.register_module()
class YOLOXCSPDarknetMYY(BaseBackbone):
    # CSP-Darknet backbone used in YOLOX.
    # From left to right: in_channels, out_channels, num_blocks, add_identity, use_spp
    # yolov5: 3 6 9 3   yolox: 3 9 9 3
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'G': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'last6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 6, False, True]],
        'last9': [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 9, True, False], [512, 1024, 9, False, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 stage_mode: str = 'C3',
                 frozen_stages: int = -1,
                 use_depthwise: bool = False,
                 use_ghostv2: bool = False,
                 use_cspnext_block: bool = False,
                 num_branchs: int = 1,
                 use_custom_block: int = 0, # 0: ghost note 父类继承有问题
                 use_dfc_attention: bool = True,
                 spp_kernal_sizes: Tuple[int] = (5, 9, 13),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.use_dfc_attention = use_dfc_attention
        self.use_depthwise = use_depthwise
        self.use_cspnext_block = use_cspnext_block
        self.use_custom_block = use_custom_block # note
        self.spp_kernal_sizes = spp_kernal_sizes
        self.num_branchs = num_branchs
        self.use_ghostv2 = use_ghostv2
        self.stage_mode =stage_mode

        logger.info(f'当前结构是{arch}\n{self.arch_settings[arch]}')
        super().__init__(self.arch_settings[arch], deepen_factor, widen_factor,
                         input_channels, out_indices, frozen_stages, plugins, #note
                         norm_cfg, act_cfg, norm_eval, init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return Focus(
            3,
            make_divisible(64, self.widen_factor),
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer. 

        Args:
            stage_idx (int): The index of a stage layer. stage的索引
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        # note 定卷积样式
        #todo 1. Main-conv
        if self.use_ghostv2:
            #! 说明第一个卷积使用 ghost卷积
            conv_layer = GhostModuleV2(
                in_channels,
                out_channels,
                kernel_size=3, # 不需要attention
                stride=2,
                )
        else:
            conv = DepthwiseSeparableConvModule \
                if self.use_depthwise else ConvModule
            #! 说明第一个卷积使用 common 卷积
            conv_layer = conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        stage.append(conv_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=self.spp_kernal_sizes,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        # todo 参数 通道数不变 卷积个数
        logger.info(f'=='*20)
        if self.stage_mode == 'C3':
            logger.info(f'{stage_idx+1} : C3 stage')
        # todo 2. CSP  bottleneck x n 的结构
            csp_layer = CSPLayerCustom(
                out_channels,
                out_channels,
                stage_id = stage_idx, #-n 传 stage_id 
                num_blocks=num_blocks, #-n 堆叠的数量
                add_identity=add_identity,
                num_branchs=self.num_branchs,
                use_dfc_attention=self.use_dfc_attention,
                use_cspnext_block= self.use_cspnext_block,
                use_custom_block= self.use_custom_block, #-n 选择哪种block
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        elif self.stage_mode == 'C3_g-ghost':
            logger.info('MIX CSP stage')
            csp_layer = CSPLayerG(
                out_channels,
                out_channels,
                stage_id = stage_idx, 
                num_blocks=num_blocks,
                add_identity=add_identity,
                num_branchs=self.num_branchs,
                use_cspnext_block= self.use_cspnext_block,
                use_custom_block= self.use_custom_block, # note
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        elif self.stage_mode == 'C3_ALL_GHOST':
            logger.info('C3_ALL_GHOST') #-n 全部使用ghost卷积
            csp_layer = CSPLayer_Ghostv2_conv(
                out_channels,
                out_channels,
                stage_id = stage_idx,  
                num_blocks=num_blocks, 
                add_identity=add_identity,
                num_branchs=self.num_branchs,
                use_cspnext_block= self.use_cspnext_block,
                use_custom_block= self.use_custom_block, # note
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        elif self.stage_mode == 'C3_main_split':
            logger.info('CSP main_split ghostv2')
            csp_layer = CSPLayer_Ghostv2_split_channel(
                out_channels,
                out_channels,
                stage_id = stage_idx, # 传id 和
                num_blocks=num_blocks, # note
                add_identity=add_identity,
                num_branchs=self.num_branchs,
                use_cspnext_block= self.use_cspnext_block,
                use_custom_block= self.use_custom_block, # note
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        stage.append(csp_layer)

        return stage
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)