# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.structures.bbox import HorizontalBoxes

from mmyolo.registry import MODELS

# 1.服务于IOU loss 计算
def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  iou_mode: str = 'ciou',
                  bbox_format: str = 'xywh',
                  siou_theta: float = 4.0,
                  eps: float = 1e-7) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou') #标配四个
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)
    #-n 统一转成xyxy格式
    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if iou_mode == 'ciou':
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) -
                           (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == 'giou':
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w)**2  # ρ_x
        rho_y = (sigma_ch / enclose_h)**2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
            1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w),
                               siou_theta) + torch.pow(
                                   1 - torch.exp(-1 * omiga_h), siou_theta)

        ious = ious - ((distance_cost + shape_cost) * 0.5)

    return ious.clamp(min=-1.0, max=1.0)

def bbox_overlaps_kld(bboxes1, bboxes2,\
                        mode='kl', is_aligned=False,\
                        eps=1e-6, weight=2):
    """用来计算iou和分布距离的

    Args:
        bboxes1 (_type_): xyxy
        bboxes2 (_type_): xyxy
        is_aligned (bool, optional): _description_. Defaults to False.
        weight (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """    
    assert mode in ['iou', 'iof', 'giou', 'wd', 'kl','center_distance2','exp_kl','kl_10'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2) #预测框
    cols = bboxes2.size(-2) #真实框

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if mode in ['box1_box2']:
        box1_box2 = area1[...,None] / area2[None,...]
        return box1_box2

    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # left和top挑最大，[B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # right和bottom挑最小，[B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1] # 交集

    union = area1[..., None] + area2[..., None, :] - overlap + eps # 并集

    if mode in ['giou']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        

    eps = union.new_tensor([eps]) # eps是一个很小的数
    union = torch.max(union, eps)
    ious = overlap / union # 交集/并集
    
    if mode in ['iou', 'iof']:
        return ious

    # calculate gious
    if mode in ['giou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

    if mode == 'giou':
        return gious

    if mode == 'center_distance2':
        # box1 , box2: x1, y1, x2, y2
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6 #
        #distance = torch.sqrt(center_distance2)
        return center_distance2

   
    if mode == 'kl':
        # KL 散度举例
        # 两个概率分布之间的距离，定义如下：
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+\
            4*whs[..., 0]**2/w1**2 + 4*whs[..., 1]**2/h1**2+\
            torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(1+kl)

        return ious, kld #add

    if mode == 'kl_10':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(10+kl)

        return kld

    if mode == 'exp_kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = torch.exp(-kl/10)

        return kld

    if mode == 'wd':
        # wasserstein_loss 瓦瑟斯坦
        # Wasserstein距离度量两个概率分布之间的距离，定义如下：
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        wasserstein = center_distance + wh_distance # 距离

        wd = 1/(1+wasserstein)

        return wd
# note:
def wasserstein_loss(pred, target, eps=1e-7, mode='exp', gamma=1, constant=12.8):
    """_summary_

    Args:
        pred (_type_): _description_
        target (_type_): _description_
        eps (_type_, optional): _description_. Defaults to 1e-7.
        mode (str, optional): _description_. Defaults to 'exp'.
        gamma (int, optional): _description_. Defaults to 1.
        constant (float, optional): _description_. Defaults to 12.8.

    Returns:
        _type_: _description_
    """
    center1 = (pred[:, :2] + pred[:, 2:]) / 2 
    center2 = (target[:, :2] + target[:, 2:]) / 2
    #todo 两个中心
    whs = center1[:, :2] - center2[:, :2] 
    #todo 两个中心的宽高距离

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps # pow(x,2) + pow(y,2)

    w1 = pred[:, 2] - pred[:, 0]  + eps
    h1 = pred[:, 3] - pred[:, 1]  + eps
    w2 = target[:, 2] - target[:, 0]  + eps
    h2 = target[:, 3] - target[:, 1]  + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance

    if mode == 'exp':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2)/constant)
        wloss = 1 - normalized_wasserstein
    if mode == 'exp_square':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2)/constant)
        wloss = 1 - normalized_wasserstein**2
    if mode == 'sqrt':
        wloss = torch.sqrt(wasserstein_2)
    
    if mode == 'log':
        wloss = torch.log(wasserstein_2 + 1)

    if mode == 'norm_sqrt':
        wloss = 1 - 1 / (gamma + torch.sqrt(wasserstein_2))

    if mode == 'w2':
        wloss = wasserstein_2

    return wloss
# note:
@MODELS.register_module()
class WassersteinLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, mode='exp', gamma=2, constant=12.8):
        super(WassersteinLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.gamma = gamma
        self.constant = constant # constant = 12.8 for AI-TOD

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * wasserstein_loss(
            pred,
            target,
            eps=self.eps,
            mode=self.mode,
            gamma=self.gamma,
            constant=self.constant,
            **kwargs)
        return loss
    
@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self,
                 iou_mode: str = 'ciou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_iou: bool = True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight,
                                                     reduction, avg_factor)

        if self.return_iou:
            return loss, iou
        else:
            return loss
