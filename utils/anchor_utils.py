import numpy as np
import torch
from pyquaternion import Quaternion

from utils.data_classes import Box


def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4); x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4); x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d


def center_to_corner_box2d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 4, 2)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 4, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = [box[0], box[1], box[2]]
        size = [box[4], box[5], box[3]]
        rotation = Quaternion(axis=[0, 0, 1], angle=box[6])
        pred_box = Box(translation, size, rotation)
        ret[i] = pred_box.bottom_corners().T

    return ret[:, :, 0:2]

def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    N = deltas.shape[0]
    deltas = deltas.view(N, -1, 8)
    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)

    anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]]) + anchors_reshaped[..., [2]]

    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    rax = torch.cos(anchors_reshaped[..., 6])
    ray = torch.sin(anchors_reshaped[..., 6])
    rgy = deltas[..., 6] + ray
    rgx = deltas[..., 7] + rax
    boxes3d[..., 6] = torch.atan2(rgy, rgx)

    return boxes3d

