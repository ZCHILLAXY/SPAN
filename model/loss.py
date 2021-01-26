#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: loss.py
@Author: Zhuang Yi
@Date: 2020/8/25
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.anchor_utils import delta_to_boxes3d


def rpn_cross_entropy_balance(input, target, num_pos=16, num_neg=48):
    r"""
    :param input: (N,1125,2)
    :param target: (15x15x5,)
    :return:
    """

    # if ohem:
    #     final_loss = rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=True,
    #                                                     num_threads=4)
    # else:
    # print(input.shape)
    # print(target.shape)
    print(target.size())
    print(target.lt(0.0).float().sum())
    target = target.view(target.shape[0], -1)
    target = torch.repeat_interleave(target.unsqueeze(2), repeats=2, dim=2)
    loss_all = []
    pos_loss = []
    neg_loss = []
    for batch_id in range(target.shape[0]):
        min_pos = min(
            len(np.where(target[batch_id].cpu() >= 0.99)[0]), num_pos)
        min_neg = min(len(np.where((target[batch_id].cpu() <= 0.1) & (
            target[batch_id].cpu() >= 0))[0]), num_neg)
        pos_index = np.where(target[batch_id].cpu() >= 0.99)[0].tolist()
        neg_index = np.where((target[batch_id].cpu() <= 0.1) & (
            target[batch_id].cpu() >= 0))[0].tolist()

        # if ohem_pos:
        #     if len(pos_index) > 0:
        #         pos_loss_bid = F.cross_entropy(input=input[batch_id][pos_index],
        #                                        target=target[batch_id][pos_index], reduction='none')
        #         selected_pos_index = nms(anchors[pos_index], pos_loss_bid.cpu().detach().numpy(), min_pos)
        #         pos_loss_bid_final = pos_loss_bid[selected_pos_index]
        #     else:
        #         pos_loss_bid = torch.FloatTensor([0]).cuda()
        #         pos_loss_bid_final = pos_loss_bid
        #
        # else:
        pos_index_random = random.sample(pos_index, min_pos)
        # print(input.shape)
        # print(target.shape)
        if len(pos_index) > 0:
            pos_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][pos_index_random],
                                                                    target=target[batch_id][pos_index_random], reduce=False)
        else:
            pos_loss_bid_final = torch.FloatTensor([0]).cuda()

        # if ohem_neg:
        #     if len(pos_index) > 0:
        #         neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
        #                                        target=target[batch_id][neg_index], reduction='none')
        #         selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), min_neg)
        #         neg_loss_bid_final = neg_loss_bid[selected_neg_index]
        #     else:
        #         neg_loss_bid = F.cross_entropy(input=input[batch_id][neg_index],
        #                                        target=target[batch_id][neg_index], reduction='none')
        #         selected_neg_index = nms(anchors[neg_index], neg_loss_bid.cpu().detach().numpy(), num_neg)
        #         neg_loss_bid_final = neg_loss_bid[selected_neg_index]
        # else:
        if len(neg_index) > 0:
            neg_index_random = random.sample(
                np.where((target[batch_id].cpu() <= 0.1) & (target[batch_id].cpu() >= 0))[0].tolist(), min_neg)
            neg_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][neg_index_random],
                                                                    target=target[batch_id][neg_index_random], reduce=False)
        else:
            neg_index_random = random.sample(
                np.where((target[batch_id].cpu() <= 0.1) & (target[batch_id].cpu() >= 0))[0].tolist(), num_neg)
            neg_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][neg_index_random],
                                                                    target=target[batch_id][neg_index_random], reduce=False)
        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all.append(loss_bid)
        pos_loss.append(pos_loss_bid_final.mean())
        neg_loss.append(neg_loss_bid_final.mean())
    final_loss = torch.stack(loss_all).mean()
    pos_final = torch.stack(pos_loss).mean()
    neg_final = torch.stack(neg_loss).mean()
    return final_loss, pos_final, neg_final

def focal_loss(input, target):
    target = target.view(target.shape[0], -1)
    target = torch.repeat_interleave(target.unsqueeze(2), repeats=2, dim=2)
    input = input.sigmoid()
    loss_all = []
    pos_loss = []
    neg_loss = []
    for batch_id in range(target.shape[0]):
        pos_inds = target[batch_id].gt(0.99).float()
        neg_inds = target[batch_id].lt(0.1).float() * target[batch_id].ge(0.0).float()
        # print(target[batch_id].lt(0.1).float().sum())
        # print(target[batch_id].gt(0.0).float().sum())

        neg_weights = torch.pow(1 - target[batch_id], 4)

        loss_bid = 0
        # print(pos_inds.sum())
        # print(neg_inds.sum())
        pos_loss_bid_final = torch.log(input[batch_id]) * torch.pow(1 - input[batch_id], 2) * pos_inds
        neg_loss_bid_final = torch.log(1 - input[batch_id]) * torch.pow(input[batch_id], 2) * neg_weights * neg_inds

        # print(torch.log(pred) * torch.pow(1 - pred, 2))
        # print(pos_loss.max())
        # print(pos_loss.min())
        # exit()
        num_pos = pos_inds.float().sum()


        if num_pos == 0:
            loss_bid = loss_bid - neg_loss_bid_final
        else:
            loss_bid = loss_bid - (pos_loss_bid_final + neg_loss_bid_final)
        loss_all.append(loss_bid)
        pos_loss.append(pos_loss_bid_final.mean())
        neg_loss.append(neg_loss_bid_final.mean())

    final_loss = torch.stack(loss_all).mean()
    pos_final = torch.stack(pos_loss).mean()
    neg_final = torch.stack(neg_loss).mean()
    return final_loss, pos_final, neg_final


def rpn_smoothL1(input, target, label, num_pos=16):
    r'''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''

    loss_all = []
    target = target.reshape(target.shape[0], -1, 8)
    label = label.reshape(label.shape[0], -1)
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() >= 0.99)[0]), num_pos)
        # if ohem:
        #     pos_index = np.where(label[batch_id].cpu() == 1)[0]
        #     if len(pos_index) > 0:
        #         loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index],
        #                                     reduction='none')
        #         sort_index = torch.argsort(loss_bid.mean(1))
        #         loss_bid_ohem = loss_bid[sort_index[-num_pos:]]
        #     else:
        #         loss_bid_ohem = torch.FloatTensor([0]).cuda()[0]
        #     loss_all.append(loss_bid_ohem.mean())
        # else:
        pos_index = np.where(label[batch_id].cpu() >= 0.99)[0]
        pos_index = random.sample(pos_index.tolist(), min_pos)
        # print(input[batch_id][pos_index].shape)
        # exit()
        # print(F.smooth_l1_loss(input[batch_id][pos_index][:, 0], target[batch_id][pos_index][:, 0], reduction='mean'))
        # print(F.smooth_l1_loss(input[batch_id][pos_index][:, 3], target[batch_id][pos_index][:, 3], reduction='mean'))
        # print(F.smooth_l1_loss(input[batch_id][pos_index][:, 7], target[batch_id][pos_index][:, 7], reduction='mean'))
        # print(F.smooth_l1_loss(input[batch_id][pos_index][:, 6], target[batch_id][pos_index][:, 6], reduction='mean'))
        # exit()
        if len(pos_index) > 0:
            loss_bid = F.smooth_l1_loss(
                input[batch_id][pos_index], target[batch_id][pos_index])
        else:
            loss_bid = torch.FloatTensor([0]).cuda()[0]
        loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def box_iou3d(input, target, anchors, label, num_pos=16):
    loss_all = []

    batch_boxes3d = delta_to_boxes3d(input, anchors).detach().cpu().numpy()
    gt_boxes3d = delta_to_boxes3d(target, anchors).detach().cpu().numpy()
    box3d_x = batch_boxes3d[:, :, 0]
    box3d_y = batch_boxes3d[:, :, 1]
    box3d_z = batch_boxes3d[:, :, 2]
    box3d_height = batch_boxes3d[:, :, 3]
    box3d_width = batch_boxes3d[:, :, 4]
    box3d_length = batch_boxes3d[:, :, 5]
    gts_x = gt_boxes3d[:, :, 0]
    gts_y = gt_boxes3d[:, :, 1]
    gts_z = gt_boxes3d[:, :, 2]
    gts_height = gt_boxes3d[:, :, 3]
    gts_width = gt_boxes3d[:, :, 4]
    gts_length = gt_boxes3d[:, :, 5]


    label = label.reshape(label.shape[0], -1)
    for batch_id in range(input.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() >= 0.99)[0]), num_pos)
        pos_index = np.where(label[batch_id].cpu() >= 0.99)[0]
        pos_index = random.sample(pos_index.tolist(), min_pos)
        if len(pos_index) > 0:
            pred_x = box3d_x[batch_id][pos_index]
            pred_y = box3d_y[batch_id][pos_index]
            pred_z = box3d_z[batch_id][pos_index]
            pred_height = box3d_height[batch_id][pos_index]
            pred_width = box3d_width[batch_id][pos_index]
            pred_length = box3d_length[batch_id][pos_index]
            gt_x = gts_x[batch_id][pos_index]
            gt_y = gts_y[batch_id][pos_index]
            gt_z = gts_z[batch_id][pos_index]
            gt_height = gts_height[batch_id][pos_index]
            gt_width = gts_width[batch_id][pos_index]
            gt_length = gts_length[batch_id][pos_index]
            inter_width = np.maximum((pred_width + gt_width) - (np.maximum(pred_x + pred_width / 2, gt_x + gt_width / 2) - np.minimum(pred_x - pred_width / 2, gt_x - gt_width / 2)), 0)
            inter_length = np.maximum((pred_length + gt_length) - (np.maximum(pred_y + pred_length / 2, gt_y + gt_length / 2) - np.minimum(pred_y - pred_length / 2, gt_y - gt_length / 2)), 0)
            inter_height = np.maximum((pred_height + gt_height) - (np.maximum(pred_z + pred_height / 2, gt_z + gt_height / 2) - np.minimum(pred_z - pred_height / 2, gt_z - gt_height / 2)), 0)
            inter_union = inter_width * inter_length * inter_height
            gt_volume = gt_width * gt_length * gt_height
            pred_volume = pred_width * pred_length * pred_height

            iou_3d = inter_union / (gt_volume + pred_volume - inter_union)
            loss_bid = 1 - iou_3d
            loss_bid = torch.FloatTensor(torch.from_numpy(loss_bid).float()).cuda()
        else:
            loss_bid = torch.FloatTensor([0]).cuda()[0]
        loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss
