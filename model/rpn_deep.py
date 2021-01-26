#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from modules.modulated_deform_conv import ModulatedDeformConvPack

...
'''=================================================
@Project -> File   ：SiamVoxel -> rpn_deep.py
@Author ：Yi_Zhuang
@Time   ：2020/5/16 3:40 下午
=================================================='''


class RPN_Deep(nn.Module):
    def __init__(self):
        super(RPN_Deep, self).__init__()
        self.exam_cls = nn.Sequential(
            ModulatedDeformConvPack(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.inst_cls = nn.Sequential(
            ModulatedDeformConvPack(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.exam_reg = nn.Sequential(
            ModulatedDeformConvPack(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))
        self.inst_reg = nn.Sequential(
            ModulatedDeformConvPack(256, 256, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(256))


        self.fusion_module_cls = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_reg = nn.Sequential(nn.Conv2d(
            256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))


        self.Box_Head = nn.Sequential(
            nn.Conv2d(256, 8*2, kernel_size=1, padding=0, stride=1))
        self.Cls_Head = nn.Sequential(
            nn.Conv2d(256, 2*2, kernel_size=1, padding=0, stride=1))



    def forward(self, examplar_feature_map, instance_feature_map):
        exam_cls_output = self.exam_cls(examplar_feature_map)
        exam_cls_output = exam_cls_output.reshape(-1, 18, 18)
        exam_cls_output = exam_cls_output.unsqueeze(0).permute(1, 0, 2, 3)

        inst_cls_output = self.inst_cls(instance_feature_map)
        inst_cls_output = inst_cls_output.reshape(-1, 30, 30)
        inst_cls_output = inst_cls_output.unsqueeze(0)

        exam_reg_output = self.exam_reg(examplar_feature_map)
        exam_reg_output = exam_reg_output.reshape(-1, 18, 18)
        exam_reg_output = exam_reg_output.unsqueeze(0).permute(1, 0, 2, 3)

        inst_reg_output = self.inst_reg(instance_feature_map)
        inst_reg_output = inst_reg_output.reshape(-1, 30, 30)
        inst_reg_output = inst_reg_output.unsqueeze(0)



        depthwise_cross_cls = F.conv2d(
            inst_cls_output, exam_cls_output, bias=None, stride=1, padding=0, groups=exam_cls_output.size()[0]).squeeze()
        depthwise_cross_reg = F.conv2d(
            inst_reg_output, exam_reg_output, bias=None, stride=1, padding=0, groups=exam_reg_output.size()[0]).squeeze()
        depthwise_cross_cls = depthwise_cross_cls.reshape(-1, 256, 13, 13)
        depthwise_cross_reg = depthwise_cross_reg.reshape(-1, 256, 13, 13)



        depthwise_cross_cls = self.fusion_module_cls(depthwise_cross_cls)
        depthwise_cross_reg = self.fusion_module_reg(depthwise_cross_reg)


        cls_prediction = self.Cls_Head(depthwise_cross_cls)
        bbox_regression_prediction = self.Box_Head(depthwise_cross_reg)


        return cls_prediction, bbox_regression_prediction
