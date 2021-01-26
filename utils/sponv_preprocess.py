#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: sponv_preprocess.py
@Author: Zhuang Yi
@Date: 2020/8/25
"""


import numpy as np
from config import cfg

def shuffle_points(point_cloud):
    shuffle_idx = np.random.permutation(point_cloud.shape[0])
    points = point_cloud[shuffle_idx]
    return points

def sponv_process_pointcloud(point_cloud, template=False, is_testset=False):
    if not is_testset:
        point_cloud = shuffle_points(point_cloud)
        max_number_of_voxel = 16000
    else:
        max_number_of_voxel = 40000
    try:
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    except:
        from spconv.utils import VoxelGenerator
    if template:
        point_cloud_range = [cfg.TEMPLATE_X_MIN, cfg.TEMPLATE_Y_MIN, cfg.TEMPLATE_Z_MIN, cfg.TEMPLATE_X_MAX, cfg.TEMPLATE_Y_MAX, cfg.TEMPLATE_Z_MAX]
    else:
        point_cloud_range = [cfg.SCENE_X_MIN, cfg.SCENE_Y_MIN, cfg.SCENE_Z_MIN, cfg.SCENE_X_MAX, cfg.SCENE_Y_MAX, cfg.SCENE_Z_MAX]
    voxel_size = np.array([cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE], dtype = np.float32)
    max_point_number = cfg.VOXEL_POINT_COUNT


    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_point_number,
        max_voxels=max_number_of_voxel
    )
    voxel_output = voxel_generator.generate(point_cloud)
    if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
    else:
        voxels, coordinates, num_points = voxel_output

    voxel_dict = {'feature_buffer': voxels,
                  'coordinate_buffer': coordinates,
                  'number_buffer': num_points}
    return voxel_dict