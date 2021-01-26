import torch

from config import cfg
import numpy as np
from utils.box_overlaps import *
from utils.anchor_utils import *



def cal_anchors():
    # Output:
    # Anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(cfg.SCENE_X_MIN, cfg.SCENE_X_MAX, cfg.FEATURE_WIDTH)
    y = np.linspace(cfg.SCENE_Y_MIN, cfg.SCENE_Y_MAX, cfg.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y)
    # All are (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * cfg.ANCHOR_Z
    w = np.ones_like(cx) * cfg.ANCHOR_W
    l = np.ones_like(cx) * cfg.ANCHOR_L
    h = np.ones_like(cx) * cfg.ANCHOR_H
    r = np.ones_like(cx)
    r[..., 0] = 0  # 0
    r[..., 1] = np.pi / 2

    # 7 * (w, l, 2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    return anchors


def cal_rpn_target(bbox, feature_map_shape, anchors, coordinate = 'lidar'):
    # Input:
    #   labels: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)
    # Attention: cal IoU on birdview

    batch_size = bbox.shape[0]
    batch_gt_boxes3d = np.expand_dims(bbox, axis=1)
    # Defined in eq(1) in 2.2
    anchors_reshaped = anchors.reshape(-1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    pos_equal_one[...] = -1
    smooth_labeling = 0.01
    targets = np.zeros((batch_size, *feature_map_shape, 16))

    for batch_id in range(batch_size):
        # BOTTLENECK; from (x,y,w,l) to (x1,y1,x2,y2)
        anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d(center_to_corner_box2d(
            batch_gt_boxes3d[batch_id], coordinate=coordinate))

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        # iou = cal_box3d_iou(anchors_reshaped, batch_gt_boxes3d[batch_id])

        # Find anchor with highest iou (iou should also > 0)
        id_highest = np.argmax(iou.T, axis = 1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # Find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)
        #print(cfg.RPN_POS_IOU)

        # Find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < cfg.RPN_NEG_IOU, axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index = True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # Cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1 * (1 - smooth_labeling) + smooth_labeling * (1 / 2) * iou[id_pos][0]

        # ATTENTION: index_z should be np.array
        targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / anchors_reshaped[id_pos, 3]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = np.sin(
            batch_gt_boxes3d[batch_id][id_pos_gt, 6]) - np.sin(anchors_reshaped[id_pos, 6])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 7] = np.cos(
            batch_gt_boxes3d[batch_id][id_pos_gt, 6]) - np.cos(anchors_reshaped[id_pos, 6])
        

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 0 * (1 - smooth_labeling) + smooth_labeling * (1 / 2) * iou[id_neg][0]
        # To avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1
    return pos_equal_one, targets

