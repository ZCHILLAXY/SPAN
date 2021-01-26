#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: deformconv.py
@Author: Zhuang Yi
@Date: 2020/9/12
"""
import torch

import torch.nn as nn
import torch.nn.functional as F
from modules import ModulatedDeformConvPack


class DeformConvAtt(nn.Module):
    def __init__(self):
        super(DeformConvAtt, self).__init__()

        # self.template_1x1_w = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.scene_1x1_w = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.template_1x1_h = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.scene_1x1_h = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.template_1x1_c = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.scene_1x1_c = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.template_1x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1),
        #                                   nn.BatchNorm2d(64))
        # self.scene_1x1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(64))

        self.a = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.b = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.c = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.d = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.e = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.f = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.g = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.h = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        # self.i = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))
        # self.j = torch.nn.Parameter(torch.zeros((1), dtype=torch.float))



        # self.t_w_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.t_h_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.t_c_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.s_w_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.s_h_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.s_c_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.t_x_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.s_x_res = nn.Sequential(nn.Conv2d(
        #     64, 64, kernel_size=1, padding=0, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # self.t_hw_res = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))
        # self.s_hw_res = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1, bias=False))

        # self.linear_t_down8 = nn.Linear(32, 32)
        # self.linear_s_down8 = nn.Linear(32, 32)
        # self.linear_t = nn.Linear(64, 64)
        # self.linear_s = nn.Linear(64, 64)
        # self.template_down8_q = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1))
        # self.template_down8_k = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1))
        # self.scene_down8_q = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1))
        # self.scene_down8_k = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1))

        # self.fused_template = nn.Sequential(
        #     nn.Conv2d(4 * 64, 64, kernel_size=1, padding=0, stride=1))
        # self.fused_scene = nn.Sequential(
        #     nn.Conv2d(4 * 64, 64, kernel_size=1, padding=0, stride=1))

        #self.dcn_scene = nn.Sequential(ModulatedDeformConvPack(64, 64, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(64))
        #self.dcn_template = nn.Sequential(ModulatedDeformConvPack(64, 64, kernel_size=3, padding=0, stride=1), nn.BatchNorm2d(64))

    def forward(self, template_feature_map, scene_feature_map):

        batchsize = template_feature_map.shape[0]
        channels = template_feature_map.shape[1]
        #downchannels = int(channels / 8)
        t_h = template_feature_map.shape[2]
        t_w = template_feature_map.shape[3]
        s_h = scene_feature_map.shape[2]
        s_w = scene_feature_map.shape[3]
        hw_dim = -1
        channel_dim = -1

        template_w = template_feature_map.view(batchsize, channels * t_h, -1)
        template_corr_w = torch.transpose(template_w, 1, 2).contiguous()

        template_self_w = torch.bmm(template_corr_w, template_w)
        template_map_w = torch.bmm(template_w, template_self_w.softmax(hw_dim)).view(batchsize, -1, t_h, t_w)
        template_map_w = self.a * template_map_w + template_feature_map


        template_h = template_feature_map.transpose(2, 3).contiguous().view(batchsize, channels * t_w, -1)
        template_corr_h = torch.transpose(template_h, 1, 2).contiguous()

        template_self_h = torch.bmm(template_corr_h, template_h)
        template_map_h = torch.bmm(template_h, template_self_h.softmax(hw_dim)).view(batchsize, -1, t_w, t_h).transpose(2, 3).contiguous()
        template_map_h = self.b * template_map_h + template_feature_map


        template_c = template_feature_map.view(batchsize, channels, -1)
        template_corr_c = torch.transpose(template_c, 1, 2).contiguous()

        template_self_channels = torch.bmm(template_c, template_corr_c)
        template_self_channels_new = torch.max(template_self_channels ,-1, keepdim=True)[0].expand_as(template_self_channels) - template_self_channels
        template_map_c = torch.bmm(template_self_channels_new.softmax(channel_dim), template_c).view(batchsize, -1, t_h, t_w)
        template_map_c = self.c * template_map_c + template_feature_map

        # template_value_hw = self.template_1x1(template_feature_map).view(batchsize, channels, -1)
        # template_hw = template_feature_map.view(batchsize, channels, -1)
        # template_corr_hw = torch.transpose(template_hw, 1, 2).contiguous()
        #
        # template_self_hw = torch.bmm(template_corr_hw, template_hw)
        # template_map_hw = torch.bmm(template_hw, template_self_hw.softmax(dim=hw_dim)).view(batchsize, -1, t_h, t_w)
        # template_map_hw = self.i * template_map_hw + template_feature_map

        scene_w = scene_feature_map.view(batchsize, channels * s_h, -1)
        scene_corr_w = torch.transpose(scene_w, 1, 2).contiguous()

        scene_self_w = torch.bmm(scene_corr_w, scene_w)
        scene_map_w = torch.bmm(scene_w, scene_self_w.softmax(hw_dim)).view(batchsize, -1, s_h, s_w)
        scene_map_w = self.d * scene_map_w + scene_feature_map


        scene_h = scene_feature_map.transpose(2, 3).contiguous().view(batchsize, channels * s_w, -1)
        scene_corr_h = torch.transpose(scene_h, 1, 2).contiguous()

        scene_self_h = torch.bmm(scene_corr_h, scene_h)
        scene_map_h = torch.bmm(scene_h, scene_self_h.softmax(hw_dim)).view(batchsize, -1, s_w, s_h).transpose(2, 3).contiguous()
        scene_map_h = self.e * scene_map_h + scene_feature_map


        scene_c = scene_feature_map.view(batchsize, channels, -1)
        scene_corr_c = torch.transpose(scene_c, 1, 2).contiguous()

        scene_self_channels = torch.bmm(scene_c, scene_corr_c)
        scene_self_channels_new = torch.max(scene_self_channels ,-1, keepdim=True)[0].expand_as(scene_self_channels) - scene_self_channels
        scene_map_c = torch.bmm(scene_self_channels_new.softmax(channel_dim), scene_c).view(batchsize, -1, s_h, s_w)
        scene_map_c = self.f * scene_map_c + scene_feature_map

        # scene_value_hw = self.scene_1x1(scene_feature_map).view(batchsize, channels, -1)
        # scene_hw = scene_feature_map.view(batchsize, channels, -1)
        # scene_corr_hw = torch.transpose(scene_hw, 1, 2).contiguous()
        #
        # scene_self_hw = torch.bmm(scene_corr_hw, scene_hw)
        # scene_map_hw = torch.bmm(scene_hw, scene_self_hw.softmax(dim=hw_dim)).view(batchsize, -1, s_h, s_w)
        # scene_map_hw = self.j * scene_map_hw + scene_feature_map

        template_map_x = torch.bmm(scene_self_channels_new.softmax(dim=channel_dim), template_c).view(batchsize, -1, t_h, t_w)
        template_map_x = self.g * template_map_x + template_feature_map
        scene_map_x = torch.bmm(template_self_channels_new.softmax(dim=channel_dim), scene_c).view(batchsize, -1, s_h, s_w)
        scene_map_x = self.h * scene_map_x + scene_feature_map

        fused_template_map = template_map_w + template_map_h + template_map_c + template_map_x
        fused_scene_map = scene_map_w + scene_map_h + scene_map_c + scene_map_x

        #fused_template_dcn = self.dcn_template(fused_template_map)
        #fused_scene_dcn = self.dcn_scene(fused_scene_map)

        return fused_template_map, fused_scene_map



class PAM_Module(nn.Module):

    def __init__(self, in_dim=64):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.t_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.t_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.t_value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.s_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.s_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.s_value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, template_map, scene_map):

        m_batchsize, C, t_h, t_w = template_map.size()
        _, _, s_h, s_w = scene_map.size()

        t_proj_query_w = self.t_query_conv(template_map).view(m_batchsize, -1, t_w).permute(0, 2, 1)
        t_proj_key_w = self.t_key_conv(template_map).view(m_batchsize, -1, t_w)
        t_energy_w = torch.bmm(t_proj_query_w, t_proj_key_w)
        t_attention_w = self.softmax(t_energy_w)
        t_proj_value_w = self.t_value_conv(template_map).view(m_batchsize, -1, t_w)

        t_out_w = torch.bmm(t_proj_value_w, t_attention_w.permute(0, 2, 1))
        t_out_w = t_out_w.view(m_batchsize, C, t_h, t_w)
        t_out_w = self.alpha * t_out_w + template_map



        s_proj_query_w = self.s_query_conv(scene_map).view(m_batchsize, -1, s_w).permute(0, 2, 1)
        s_proj_key_w = self.s_key_conv(scene_map).view(m_batchsize, -1, s_w)
        s_energy_w = torch.bmm(s_proj_query_w, s_proj_key_w)
        s_attention_w = self.softmax(s_energy_w)
        s_proj_value_w = self.s_value_conv(scene_map).view(m_batchsize, -1, s_w)

        s_out_w = torch.bmm(s_proj_value_w, s_attention_w.permute(0, 2, 1))
        s_out_w = s_out_w.view(m_batchsize, C, s_h, s_w)
        s_out_w = self.beta * s_out_w + scene_map


        t_proj_query_h = self.t_query_conv(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h).permute(0, 2, 1)
        t_proj_key_h = self.t_key_conv(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)
        t_energy_h = torch.bmm(t_proj_query_h, t_proj_key_h)
        t_attention_h = self.softmax(t_energy_h)
        t_proj_value_h = self.t_value_conv(template_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, t_h)

        t_out_h = torch.bmm(t_proj_value_h, t_attention_h.permute(0, 2, 1))
        t_out_h = t_out_h.view(m_batchsize, C, t_w, t_h).permute(0, 1, 3, 2)
        t_out_h = self.gamma * t_out_h + template_map

        s_proj_query_h = self.s_query_conv(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h).permute(0, 2, 1)
        s_proj_key_h = self.s_key_conv(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)
        s_energy_h = torch.bmm(s_proj_query_h, s_proj_key_h)
        s_attention_h = self.softmax(s_energy_h)
        s_proj_value_h = self.s_value_conv(scene_map).permute(0, 1, 3, 2).contiguous().view(m_batchsize, -1, s_h)

        s_out_h = torch.bmm(s_proj_value_h, s_attention_h.permute(0, 2, 1))
        s_out_h = s_out_h.view(m_batchsize, C, s_w, s_h).permute(0, 1, 3, 2)
        s_out_h = self.omega * s_out_h + scene_map

        fused_template_map = t_out_w + t_out_h
        fused_scene_map = s_out_w + s_out_h


        return fused_template_map, fused_scene_map


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=256):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.omega = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, template_map, scene_map):
        m_batchsize, C, t_h, t_w = template_map.size()
        _, _, s_h, s_w = scene_map.size()

        t_proj_query_c = template_map.view(m_batchsize, C, -1)
        t_proj_key_c = template_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        t_energy_c = torch.bmm(t_proj_query_c, t_proj_key_c)
        t_energy_c_new = torch.max(t_energy_c, -1, keepdim=True)[0].expand_as(t_energy_c) - t_energy_c
        t_attention_c = self.softmax(t_energy_c_new)
        t_proj_value_c = template_map.view(m_batchsize, C, -1)

        t_out_c = torch.bmm(t_attention_c, t_proj_value_c)
        t_out_c = t_out_c.view(m_batchsize, C, t_h, t_w)
        t_out_c = self.alpha * t_out_c + template_map


        s_proj_query_c = scene_map.view(m_batchsize, C, -1)
        s_proj_key_c = scene_map.view(m_batchsize, C, -1).permute(0, 2, 1)
        s_energy_c = torch.bmm(s_proj_query_c, s_proj_key_c)
        s_energy_c_new = torch.max(s_energy_c, -1, keepdim=True)[0].expand_as(s_energy_c) - s_energy_c
        s_attention_c = self.softmax(s_energy_c_new)
        s_proj_value_c = scene_map.view(m_batchsize, C, -1)

        s_out_c = torch.bmm(s_attention_c, s_proj_value_c)
        s_out_c = s_out_c.view(m_batchsize, C, s_h, s_w)
        s_out_c = self.beta * s_out_c + scene_map

        t_out_x = torch.bmm(s_attention_c, t_proj_value_c)
        t_out_x = t_out_x.view(m_batchsize, C, t_h, t_w)
        t_out_x = self.gamma * t_out_x + template_map

        s_out_x = torch.bmm(t_attention_c, s_proj_value_c)
        s_out_x = s_out_x.view(m_batchsize, C, s_h, s_w)
        s_out_x = self.omega * s_out_x + scene_map

        fused_template_map = t_out_c + t_out_x
        fused_scene_map = s_out_c + s_out_x


        return fused_template_map, fused_scene_map