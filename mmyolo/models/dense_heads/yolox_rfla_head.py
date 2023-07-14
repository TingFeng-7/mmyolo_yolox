# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from .yolov5_head import YOLOv5Head
# from mmyolo.utils import register_all_modules
# register_all_modules

# @MODELS.register_module()
class YOLOXHeadModule(BaseModule):
    """YOLOXHead head module used in `YOLOX.
    `<https://arxiv.org/abs/2107.08430>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid
        stacked_convs (int): Number of stacking convs of the head.
            Defaults to 2.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Defaults to False.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Union[int, Sequence],
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: Union[bool, str] = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs 
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.num_base_priors = num_base_priors

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            # 以上三个方法来定义
            self.multi_level_conv_cls.append(conv_cls) #[channel num_class]
            self.multi_level_conv_reg.append(conv_reg) #[channel 4]
            self.multi_level_conv_obj.append(conv_obj) #[channel 1]

    def _build_stacked_convs(self) -> nn.Sequential:
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        #只有x是一个列表, output of neck
        # for l in x:
        #     print(l.shape)
        return multi_apply(self.forward_single, x, self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj)

    def forward_single(self, x: Tensor, cls_convs: nn.Module,
                       reg_convs: nn.Module, conv_cls: nn.Module,
                       conv_reg: nn.Module,
                       conv_obj: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x) #分类出口
        reg_feat = reg_convs(x) #回归出口
        #第二层
        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness


@MODELS.register_module()
class RFLA_YOLOXHead(YOLOv5Head): #loss 部分
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.
    Args:
        head_module(ConfigType): Base module used for YOLOXHead
        prior_generator: Points generator feature maps in
            2D points-based detectors.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        loss_bbox_aux (:obj:`ConfigDict` or dict): Config of bbox aux loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='YOLOXBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True, #ce + sigmoid
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox_aux: ConfigType = dict(
                     type='mmdet.L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        self.use_bbox_aux = False
        self.loss_bbox_aux = loss_bbox_aux
        # prior_generator['stride']=self.featmap_strides # note: 把stride传进来
        # print('1',self.prior_generator)

        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # print('2',self.prior_generator)
        

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        self.loss_bbox_aux: nn.Module = MODELS.build(self.loss_bbox_aux)
        if self.train_cfg:
            # note: 创建assginer
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            # YOLOX does not support sampling
            self.sampler = PseudoSampler()

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        return self.head_module(x)

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head. 损失计算

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W). 特征图上每个point的质量
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # cls_scores 存放每个level的point的score:  [[b,numc,h1,w1],[b,numc,h2,w2],[b,numc,h3,w3]]
        #1.取出三个特征图的尺寸 [torch.Size([4, 10, 96, 160]), 
        # torch.Size([4, 10, 48, 80]), torch.Size([4, 10, 24, 40])]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores] 
        # 2.生成anchors
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
        # [torch.Size([15360, 4]), torch.Size([3840, 4]), torch.Size([960, 4])]  [hxw,4]
        #!2.add 每层anchor另外生成
        # mlvl_priors_erf = self.mlvl_priors_trf_to_erf_level(mlvl_priors)
        #! end
        #3.不同featuremap分别进行操作
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ] #[torch.Size([4, 15360]), torch.Size([4, 3840]), torch.Size([4, 960])]
        
        #4.  按照维度1， 将不同特征图的结果堆叠起来
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1) # torch.Size([4, 20160, 10]) 结果还没放缩
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)# torch.Size([4, 20160, 4]) 数字比较小 tensor([1.3350, 0.7671, 1.4141, 0.5449]
        #? 4.1 flatten_bbox_preds 预测的是什么呢
        flatten_objectness = torch.cat(flatten_objectness, dim=1) # torch.Size([4, 20160])  每个batch 一个质量估计
        #? 4.2 anchors堆叠
        flatten_priors = torch.cat(mlvl_priors)
        # torch.Size([4, 20160, 4])--- anchor points [left ,top ,stride, stride]
        #? 4.3 使用bboxcoder进行预测框解码尺度还原
        flatten_bboxes = self.bbox_coder.decode(flatten_priors[..., :2],
                                                flatten_bbox_preds,
                                                flatten_priors[..., 2]) #stride
        

        #! 5.标签分配 获取taget; cls obj bbox;target是什么意思
        (pos_masks, cls_targets, obj_targets, bbox_targets, bbox_aux_target,
         num_fg_imgs) = \
            multi_apply(
             self._get_targets_single, 
             # single func 对一张图片进行label assign，range(len(flatten_priors.size[0]-1)),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_cls_preds.detach(), flatten_bboxes.detach(),
             flatten_objectness.detach(), 
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             )

        # The experimental results show that 'reduce_mean' can improve performance on the COCO dataset.
        #6. 正样本统计
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0) # torch.Size([1750, 10])正样本的预测
        obj_targets = torch.cat(obj_targets, 0) # torch.Size([80640, 1]) torch.sum(obj_targets)
        bbox_targets = torch.cat(bbox_targets, 0) # torch.Size([1750, 4])正样本的预测
        if self.use_bbox_aux:
            bbox_aux_target = torch.cat(bbox_aux_target, 0)

        #! 7. loss计算部分
        # loss obj：ce， loss cls：ce，loss bbox：iou loss
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), 
                                 obj_targets) / num_total_samples # todo 质量估计loss
        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                cls_targets) / num_total_samples # todo 分类loss
            loss_bbox = self.loss_bbox(
                flatten_bboxes.view(-1, 4)[pos_masks],
                bbox_targets) / num_total_samples # todo 回归框loss
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_bbox_aux:
            if num_pos > 0:
                loss_bbox_aux = self.loss_bbox_aux(
                    flatten_bbox_preds.view(-1, 4)[pos_masks],
                    bbox_aux_target) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_bbox_aux = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_bbox_aux=loss_bbox_aux)

        return loss_dict

    def mlvl_priors_trf_to_erf_level(self, mlvl_priors):
        rfields = []
        trfs = self.gen_trf()
        self.fraction = 2, #! 除以2
        for num in range(len(mlvl_priors)):
            rfield=[]
            if len(mlvl_priors) ==3 : #从p3开始 [p3 p4 p5]
                rfnum = num +1
            else:
                rfnum = num
            if rfnum == 0:
                rf = trfs[0]//self.fraction[0]
            elif rfnum == 1:
                rf = trfs[1]//self.fraction[0]
            elif rfnum == 2:
                rf = trfs[2]//self.fraction[0]
            elif rfnum == 3:
                rf = trfs[3]//self.fraction[0]
            elif rfnum == 4:
                rf = trfs[4]//self.fraction[0]
            else:
                rf = trfs[5]//self.fraction[0]
            # center format[0]
            # 理论感受野 to 有效感受野
            point = mlvl_priors[num]
            px1 = point[...,0] - rf/2
            py1 = point[...,1] - rf/2
            px2 = point[...,0] + rf/2
            py2 = point[...,1] + rf/2
            rfield = torch.cat((px1[...,None], py1[...,None]), dim=1)
            rfield = torch.cat((rfield, px2[...,None]), dim=1)
            rfield = torch.cat((rfield, py2[...,None]), dim=1)
            rfields.append(rfield)
        rfields = torch.cat(rfields, dim=0)
        return rfields

    @torch.no_grad()
    def _get_targets_single( # 出单个Target
            self,
            priors: Tensor,
            cls_preds: Tensor,
            decoded_bboxes: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            batch_i: int=1,
            gt_instances_ignore: Optional[InstanceData] = None,
            trf: list=[]) -> tuple:
        """Compute classification, regression, and objectness targets for
        priors in a single image. 每个batch的东西

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                bbox_aux_target (int): BBox aux targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """
        # 每个batch 的
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            bbox_aux_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    bbox_aux_target, 0)
        # YOLOX uses center priors with 0.5 offset to assign targets,
        # 分配标签的时候用的是anchor的中心点
        # but use center priors without offset to regress bboxes.
        # 但是回归的时候 没用

        # prior 还是left top的形式 -- torch.Size([20160, 4])
        # 1.[0,0,8,8]->[4,4,8,8] ::: left top -->cx cy
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1) # torch.Size([20160, 4])


        # 2. 计算score（两个相乘）scores = cls*objectness 算实际得分
        # --- objectness.unsqueeze(1) row to column
        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid() # torch.Size([20160, 10])
        pred_instances = InstanceData( #scores,boxes,priors 字典
            bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors,)
            # priors_erf=priors_erf) 
        
        # 3. 标签分配 label assign _get_targets_single ==> assigner
        # priors_erf
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        # assign_result ==>>> Assign_Result 
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        # print(f'one image pos sample:{num_pos_per_img}')

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        bbox_aux_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_bbox_aux:
            bbox_aux_target = self._get_bbox_aux_target(
                bbox_aux_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                bbox_aux_target, num_pos_per_img)

    def _get_bbox_aux_target(self,
                             bbox_aux_target: Tensor,
                             gt_bboxes: Tensor,
                             priors: Tensor,
                             eps: float = 1e-8) -> Tensor:
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        bbox_aux_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        bbox_aux_target[:,2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return bbox_aux_target
    # 移植
    def gen_trf(self):
        '''
        Calculate the theoretical receptive field from P2-p7 of a standard ResNet-50-FPN.
        计算理论感受野
        # ref: https://distill.pub/2019/computing-receptive-fields/
        '''
        j_i = [1]
        for i in range(7):
            j = j_i[i]*2
            j_i.append(j)
        r0 = 1
        r1 = r0 + (7-1)*j_i[0]
        r2 = r1 + (3-1)*j_i[1]
        trf_p2 = r2 + (3-1)*j_i[2]*3
        r3 = trf_p2 + (3-1)*j_i[2]
        trf_p3 = r3 + (3-1)*j_i[3]*3

        r4 = trf_p3 + (3-1)*j_i[3]
        trf_p4 = r4 + (3-1)*j_i[4]*5

        r5 = trf_p4 + (3-1)*j_i[4]
        trf_p5 = r5 + (3-1)*j_i[5]*2
 
        trf_p6 = trf_p5 + (3-1)*j_i[6]

        trf_p7 = trf_p6 + (3-1)*j_i[7]

        trfs = [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]

        return trfs
