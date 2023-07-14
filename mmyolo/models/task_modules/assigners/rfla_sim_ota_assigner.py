# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from mmdet.utils import ConfigType
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from .utils import yolox_iou_rfla_calculator #add

INF = 100000.0
EPS = 1.0e-7

@TASK_UTILS.register_module()
class SimOTAAssignerWithRF(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Ground truth center size
            to judge whether a prior is in center. Defaults to 2.5.
        candidate_topk (int): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Defaults to 10.
        iou_weight (float): The scale factor for regression
            iou cost. Defaults to 3.0.
        cls_weight (float): The scale factor for classification
            cost. Defaults to 1.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 mode: str='iou',
                 iou_weight: float = 3.0, # iou3
                 cls_weight: float = 1.0, # cls1
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
                 rfla_calculator: ConfigType = dict(type='BboxOverlaps2D')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        # self.rfields = rfields
        self.mode = mode
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.iou_calculator = yolox_iou_rfla_calculator

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors using SimOTA.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        #预测框、预测得分、anchor
        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ), 0, dtype=torch.long) #用·0来填满数组
        # 如果gt为0
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # valid and center候选
        # 1.还是要正常采样一下,中心和inside采样,priors只用于采样
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)
        # valid_mas 布尔数组， is_in_boxes_and_center更严格一点

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        # 2.如果初筛的时候没有有效的正样本
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        
        # 3.预测框 和 GT 去计算iou
        pairwise_ious, kld = self.iou_calculator(valid_decoded_bbox, gt_bboxes, mode='kl')
        #!rfla
        # overlaps = self.rfla_calculator(gt_bboxes, rfields, mode='kl')
        # priors_rescale = self.anchor_rescale(rfields, self.ratio) #缩放anchor
        # pairwise_kld = self.rfla_calculator(gt_bboxes, priors, mode='kl')
        #!rfla end
        #用中文解释下这段代码“-torch.log(pairwise_ious + EPS)”
        iou_cost = -torch.log(pairwise_ious + EPS)
        kld_cost = -torch.log(kld + EPS)
        gt_onehot_label = (F.one_hot(gt_labels.to(torch.int64),
                            pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                            num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1) #指定维度乘以几倍
        # b表示对a的对应维度进行乘以1，乘以num_gt，乘以1的操作，
        # 所以b：torch.Size([valid_preds, num_gt, num_cls])
        # disable AMP autocast and calculate BCE with FP32 to avoid overflow 防止溢出
        with torch.cuda.amp.autocast(enabled=False):
            cls_cost = (
                F.binary_cross_entropy(
                    valid_pred_scores.to(dtype=torch.float32),
                    gt_onehot_label,
                    reduction='none',
                ).sum(-1).to(dtype=valid_pred_scores.dtype))
            
        #! 4.消耗矩阵 pair-wise matching degree
        if self.mode=='iou':
            self.iou_weight = self.iou_weight
            self.kld_weight= 0
        elif self.mode == 'kld':
            self.kld_weight= self.iou_weight
            self.iou_weight = 0

        cost_matrix = ( 
            cls_cost * self.cls_weight + 
            iou_cost * self.iou_weight + 
            kld_cost * self.kld_weight +
            (~is_in_boxes_and_center) * INF) #不在gt内部也不在中心的 自动
        
        # simota
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format

        # 1、 特征图anchors(预测框) 分配到 对应的 gt_id
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1 
        # new_full 用-1 填充 label矩阵
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1) 
        # 2、 特征图anchors(预测框) 分配到 对应的 gt_label
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        #最大的覆盖
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        # 每个gt动态决定样本数目
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        # 第一步分配
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            #pos_idx 是一批样本的id
            # print(f'当前gt_{gt_idx} 根据cost分配, 正样本的id:{pos_idx}')
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # 进行一个去重的操作
        # 第二步分配
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        # 在第1维度上进行求和，看这个预测框 有没有分配给多个gt，求和大于1说明是有几个gt都匹配到了。
        # print('------------------'*5)
        # print(f'priors which match gt totally: {prior_match_gt_mask[matching_matrix.sum(1) > 0].size(0)}')
        # print(f'priors which match many gt: {prior_match_gt_mask[prior_match_gt_mask==True].size(0)}')
        # 对这些个预测框的归属再决定
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            # 对于这种重复分配的，全部清零
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1 
            # 把这个框分给令他损失最小的gt_id，也就是最接近
        # ?如果是密集预测的话 可能存在这种prior 符合 两个密集gt的情形    
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        
        # 基于匹配矩阵
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # 这段代码的作用是：首先根据一个布尔类型的数组 fg_mask_inboxes 从匹配矩阵 matching_matrix 
        # 中选择一部分行，然后在这些行中找到每行最大值所在的列的索引，将这些索引保存在 matched_gt_inds 中。
        matched_pred_ious = (matching_matrix *pairwise_ious).sum(1)[fg_mask_inboxes]
        #! debug
        # print(f'gt_nums:{num_gt}, priors matched_gt_inds_num: {len(matched_gt_inds)}')
        counts = torch.histc(matched_gt_inds, bins=num_gt, min=1, max=num_gt)
        index = torch.nonzero(counts==0)
        # print(counts)
        # print(f'gt_nums_no_matched_anchor:{index.size(0)} :\n{index.squeeze()},')
        return matched_pred_ious, matched_gt_inds
    
    def get_in_gt_and_in_center_info(
        self, priors: Tensor, gt_bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors. 
        1. prior center 是否在 gt 框的内部
        2. prior center 是否在 gt center的半径范围之内 """
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # ? 1.is prior centers in gt bboxes, shape: [n_prior, n_gt]
        # prior到四条边的距离
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # ? 2.is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers
