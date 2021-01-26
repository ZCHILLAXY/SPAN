import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ModulatedDeformConvPack

from config import cfg
from model.deformconv import DeformConvAtt, CAM_Module, PAM_Module
from model.pointpillar.base_bev_backbone import BaseBEVBackbone
from model.pointpillar.pillar_vfe import PillarVFE
from model.pointpillar.pointpillar_scatter import PointPillarScatter
from utils.anchors import *
from utils.colorize import colorize
from utils.nms import nms
from model.rpn import MiddleAndRPNFeature, ConvMD
from model.rpn_deep import RPN_Deep
from config import cfg
import pdb




small_addon_for_BCE = 1e-6


class SiamPillar(nn.Module):
    def __init__(self, cls = 'Car', alpha = 1, beta = 8, sigma = 3):
        super(SiamPillar, self).__init__()

        self.cls = cls
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.template_pvfe = PillarVFE(template=True)
        self.scene_pcfe = PillarVFE()
        self.template_scatter = PointPillarScatter(template=True)
        self.scene_scatter = PointPillarScatter()
        self.template_rpn = BaseBEVBackbone()
        self.scene_rpn = BaseBEVBackbone()
        self.pam = PAM_Module()
        self.cam = CAM_Module()
        self.deep_1_RPN = RPN_Deep()
        self.deep_2_RPN = RPN_Deep()
        self.deep_3_RPN = RPN_Deep()



        self.weighted_sum_layer_alpha = nn.Conv2d(3 * 4, 4, kernel_size=1, padding=0,
            groups=4)
        self.weighted_sum_layer_beta = nn.Conv2d(3 * 16, 16, kernel_size=1, padding=0,
            groups=16)


        self.prob_conv = ConvMD(2, 2*2, 2*2, 1, (1, 1), (0, 0), bn=False, activation=False)

        self.reg_conv = ConvMD(2, 8*2, 8*2, 1, (1, 1), (0, 0), bn=False, activation=False)


        # Generate anchors
        self.anchors = cal_anchors()    # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 8]; 2 means two rotations; 8 means (cx, cy, cz, h, w, l, r)

        self.rpn_output_shape = self.template_rpn.output_shape


    def forward(self, batch_size, t_vox_feature, t_vox_number, t_vox_coordinate, s_vox_feature, s_vox_number, s_vox_coordinate):

        batchsize = batch_size

       
        template_pvfes = self.template_pvfe(t_vox_feature, t_vox_number, t_vox_coordinate)
        scene_pvfes = self.scene_pcfe(s_vox_feature, s_vox_number, s_vox_coordinate)
        template_features = self.template_scatter(template_pvfes, t_vox_coordinate, batchsize)
        scene_features = self.scene_scatter(scene_pvfes, s_vox_coordinate, batchsize)
        template_features, scene_features = self.pam(template_features, scene_features)

        template_out_1, template_out_2, template_out_3 = self.template_rpn(template_features)
        scene_out_1, scene_out_2, scene_out_3 = self.scene_rpn(scene_features)

        template_out_1, scene_out_1 = self.cam(template_out_1, scene_out_1)
        template_out_2, scene_out_2 = self.cam(template_out_2, scene_out_2)
        template_out_3, scene_out_3 = self.cam(template_out_3, scene_out_3)

        deep_1_cls_prediction, deep_1_bbox_regression_prediction = self.deep_1_RPN(
            template_out_1, scene_out_1)
        deep_2_cls_prediction, deep_2_bbox_regression_prediction = self.deep_2_RPN(
            template_out_2, scene_out_2)
        deep_3_cls_prediction, deep_3_bbox_regression_prediction = self.deep_3_RPN(
            template_out_3, scene_out_3)


        stacked_cls_prediction = torch.cat((deep_1_cls_prediction, deep_2_cls_prediction, deep_3_cls_prediction),
        2).reshape(batchsize, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT).reshape(batchsize, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT)
        stacked_regression_prediction = torch.cat((deep_1_bbox_regression_prediction, deep_2_bbox_regression_prediction, deep_3_bbox_regression_prediction),
        2).reshape(batchsize, 16, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT).reshape(batchsize, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT)


        pillar_cls_prediction = self.weighted_sum_layer_alpha(stacked_cls_prediction)
        pillar_reg_prediction = self.weighted_sum_layer_beta(stacked_regression_prediction)

        p_map = self.prob_conv(pillar_cls_prediction)
        r_map = self.reg_conv(pillar_reg_prediction)

        pred_conf = p_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_HEIGHT).permute(0, 2, 1)
        pred_reg = r_map.reshape(-1, 8, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_HEIGHT).permute(0, 2, 1)

        return pred_conf, pred_reg

    def track_init(self, t_vox_feature, t_vox_number, t_vox_coordinate):

        template_pvfes = self.template_pvfe(t_vox_feature, t_vox_number, t_vox_coordinate)
        self.template_features = self.template_scatter(template_pvfes, t_vox_coordinate, 1)




    def track(self, s_vox_feature, s_vox_number, s_vox_coordinate):
        
        scene_pvfes = self.scene_pcfe(s_vox_feature, s_vox_number, s_vox_coordinate)
        
        scene_features = self.scene_scatter(scene_pvfes, s_vox_coordinate, 1)
        template_features, scene_features = self.pam(self.template_features, scene_features)
        template_out_1, template_out_2, template_out_3 = self.template_rpn(template_features)
        scene_out_1, scene_out_2, scene_out_3 = self.scene_rpn(scene_features)

        template_out_1, scene_out_1 = self.cam(template_out_1, scene_out_1)
        template_out_2, scene_out_2 = self.cam(template_out_2, scene_out_2)
        template_out_3, scene_out_3 = self.cam(template_out_3, scene_out_3)


        deep_1_cls_prediction, deep_1_bbox_regression_prediction = self.deep_1_RPN(
            template_out_1, scene_out_1)
        deep_2_cls_prediction, deep_2_bbox_regression_prediction = self.deep_2_RPN(
            template_out_2, scene_out_2)
        deep_3_cls_prediction, deep_3_bbox_regression_prediction = self.deep_3_RPN(
            template_out_3, scene_out_3)

        stacked_cls_prediction = torch.cat((deep_1_cls_prediction, deep_2_cls_prediction, deep_3_cls_prediction),
            2).reshape(1, 4, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT).reshape(1, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT)
        stacked_regression_prediction = torch.cat(
            (deep_1_bbox_regression_prediction, deep_2_bbox_regression_prediction, deep_3_bbox_regression_prediction),
            2).reshape(1, 16, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT).reshape(1, -1, cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT)


        pillar_cls_prediction = self.weighted_sum_layer_alpha(stacked_cls_prediction)
        pillar_reg_prediction = self.weighted_sum_layer_beta(stacked_regression_prediction)

        p_map = self.prob_conv(pillar_cls_prediction)
        r_map = self.reg_conv(pillar_reg_prediction)

        pred_conf = p_map.reshape(-1, 2, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_HEIGHT).permute(0, 2, 1)
        pred_reg = r_map.reshape(-1, 8, 2 * cfg.FEATURE_WIDTH * cfg.FEATURE_HEIGHT).permute(0, 2, 1)
        
        #
        probs = F.sigmoid(pred_conf)[:, :, 1].squeeze()
        
        
        batch_boxes3d = delta_to_boxes3d(pred_reg, self.anchors)

        best_idx = torch.argmax(probs)

        ret_box3d = batch_boxes3d[:, best_idx, :]


        return ret_box3d[0]



