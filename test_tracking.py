import math
import time
import os
import logging
import argparse
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mayavi import mlab
from pyquaternion import Quaternion
from tqdm import tqdm

import torch

import utils.kitty_utils as utils
import copy
from datetime import datetime

from model.model import SiamPillar
from utils.draw_utils import draw_lidar, draw_gt_boxes3d
from utils.metrics import AverageMeter, Success, Precision
from utils.metrics import estimateOverlap, estimateAccuracy
from utils.data_classes import PointCloud, Box
from loader.Dataset import SiameseTest

import torch.nn.functional as F
from torch.autograd import Variable

from utils.sponv_preprocess import sponv_process_pointcloud


def test(loader,model,epoch=-1,shape_aggregation="",reference_BB="",max_iter=-1,IoU_Space=3):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    dataset = loader.dataset
    batch_num = 0

    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno)) as t:
        for batch in loader:          
            batch_num = batch_num+1
            # measure data loading time
            data_time.update((time.time() - end))
            for PCs, BBs, list_of_anno in batch: # tracklet
                results_BBs = []
                results_success = []
                results_precision = []
                for i, _ in enumerate(PCs):
                    this_anno = list_of_anno[i]
                    this_BB = BBs[i]
                    this_PC = PCs[i]
                    gt_boxs = []
                    result_boxs = []

                    # INITIAL FRAME
                    if i == 0:
                        box = BBs[i]
                        results_BBs.append(box)
                        first_PC = utils.getModel([this_PC], [this_BB], offset=dataset.offset_BB, scale=dataset.scale_BB)

                    else:
                        previous_BB = BBs[i - 1]
                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB
                            # ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB
                        candidate_PC,candidate_label,candidate_reg, new_ref_box, new_this_box, trans, rot_mat = utils.cropAndCenterPC_test(
                                        this_PC,
                                        ref_BB,this_BB,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB)

                        candidate_PC = utils.regularizePC_scene(candidate_PC, dataset.input_size,istrain=False)
                        scene_voxel_dict = sponv_process_pointcloud(candidate_PC)

                            # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                        if ("firstandprevious".upper() in shape_aggregation.upper()):
                            # tracklet_length = len(PCs)
                            # first_ratio = - (i ** 4  / (tracklet_length * 2) ** 4) + 1
                            # new_pts_idx = np.random.randint(low=0, high=PCs[0].points.shape[1], size=int(first_ratio * PCs[0].points.shape[1]), dtype=np.int64)
                            # # previous_ratio = 1 - first_ratio
                            # PCs[0] = PointCloud(PCs[0].points[:, new_pts_idx])
                            # PCs[i - 1] = PointCloud(PCs[i - 1].points[:, 0:int(PCs[i - 1].points.shape[1] * previous_ratio)])
                            model_PC = utils.getModel([PCs[0], PCs[i-1]], [results_BBs[0],results_BBs[i-1]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                        elif ("first".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0]], [results_BBs[0]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                        elif ("previous".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[i-1]], [results_BBs[i-1]],offset=dataset.offset_BB,scale=dataset.scale_BB)
                        elif ("all".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel(PCs[:i],results_BBs,offset=dataset.offset_BB,scale=dataset.scale_BB)
                        else:
                            model_PC = utils.getModel(PCs[:i],results_BBs,offset=dataset.offset_BB,scale=dataset.scale_BB)
                        model_PC = utils.regularizePC_template(model_PC, dataset.input_size,istrain=False)


                        template_voxel_dict = sponv_process_pointcloud(model_PC, template=True)
                        t_vox_feature = torch.from_numpy(template_voxel_dict['feature_buffer']).float()
                        t_vox_number = torch.from_numpy(template_voxel_dict['number_buffer']).float()
                        t_vox_coordinate = torch.from_numpy(np.pad(template_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode = 'constant', constant_values = 0)).float()
                        s_vox_feature = torch.from_numpy(scene_voxel_dict['feature_buffer']).float()
                        s_vox_number = torch.from_numpy(scene_voxel_dict['number_buffer']).float()
                        s_vox_coordinate = torch.from_numpy(np.pad(scene_voxel_dict['coordinate_buffer'], ((0, 0), (1, 0)), mode = 'constant', constant_values = 0)).float()


                        
                       
                        t_vox_feature = Variable(t_vox_feature, requires_grad=False).cuda()
                        t_vox_number = Variable(t_vox_number, requires_grad=False).cuda()
                        t_vox_coordinate = Variable(t_vox_coordinate, requires_grad=False).cuda()
                        s_vox_feature = Variable(s_vox_feature, requires_grad=False).cuda()
                        s_vox_number = Variable(s_vox_number, requires_grad=False).cuda()
                        s_vox_coordinate = Variable(s_vox_coordinate, requires_grad=False).cuda()

                        model.track_init(t_vox_feature, t_vox_number, t_vox_coordinate)

                        estimation_box = model.track(s_vox_feature, s_vox_number, s_vox_coordinate)
                        estimation_box = estimation_box.cpu().detach().numpy()
                        # box_idx = estimation_boxs_cpu[:,4].argmax()
                        # estimation_box_cpu = estimation_boxs_cpu[box_idx,0:4]
                        #
                        b_center = estimation_box[0:3]
                        b_size = [estimation_box[4], estimation_box[5], estimation_box[3]]
                        b_rot = Quaternion(axis=[0, 0, 1], radians=estimation_box[6])
                        box = Box(center=b_center,
                                  size=b_size,
                                  orientation=b_rot)
                        final_box = copy.deepcopy(box)
                        final_box.rotate(Quaternion(matrix=(np.transpose(rot_mat))))
                        final_box.translate(-trans)
                        results_BBs.append(final_box)

                        # box_2 = utils.getOffsetBB(ref_BB, estimate_offset_cpu)
                        # center_box_2 = utils.getOffsetBB(new_ref_box, estimate_offset_cpu)


                        # print(estimation_boxs_cpu[6])
                        # print(box.orientation.radians)
                        # print(new_this_box.orientation.radians)

                        if args.vis:
                            #p2b_box = np.loadtxt('p2bbox/{}.txt'.format(this_anno['frame']), dtype=np.float32)
                            pic_dir = os.path.join('tracking_results', str(this_anno['scene']), str(this_anno['track_id']))
                            if not os.path.exists(pic_dir):
                                os.makedirs(pic_dir)

                            fig = draw_lidar(candidate_PC, is_grid=False, is_axis=False,
                                         is_top_region=False)
                            draw_gt_boxes3d(gt_boxes3d=new_this_box.corners().T.reshape(1, 8, 3), color=(0, 1, 0), fig=fig)
                            #draw_gt_boxes3d(gt_boxes3d=p2b_box.reshape(1, 8, 3), color=(0, 0, 1), fig=fig)
                            draw_gt_boxes3d(gt_boxes3d=box.corners().T.reshape(1, 8, 3), color=(1, 0, 0), fig=fig)

                            # draw_gt_boxes3d(gt_boxes3d=center_box_2.corners().T.reshape(1, 8, 3), color=(0, 0, 1), fig=fig)
                            pic_path = os.path.join(pic_dir, str(this_anno['frame']) + '.png')
                            # heat_path = os.path.join(pic_dir, str(this_anno['frame']) + '_heat2' + '.png')
                            mlab.savefig(pic_path, figure=fig)
                            mlab.close()

                            # cv2.imwrite(heat_path, score_map)

                    # print(results_BBs[-1])
                    # print(BBs[i])

                    # estimate overlap/accuracy fro current sample

                    this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
                    this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)
                    results_success.append(this_overlap)
                    results_precision.append(this_accuracy)


                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)
                    Success_batch.add_overlap(this_overlap)
                    Precision_batch.add_accuracy(this_accuracy)



                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    t.update(1)

                    if Success_main.count >= max_iter and max_iter >= 0:
                        return Success_main.average, Precision_main.average


                t.set_description('Test {}: '.format(epoch)+
                                  'Time {:.3f}s '.format(batch_time.avg)+
                                  '(it:{:.3f}s) '.format(batch_time.val)+
                                  'Data:{:.3f}s '.format(data_time.avg)+
                                  '(it:{:.3f}s), '.format(data_time.val)+
                                  'Succ/Prec:'+
                                  '{:.1f}/'.format(Success_main.average)+
                                  '{:.1f}'.format(Precision_main.average))
                logging.info('batch {}'.format(batch_num)+'Succ/Prec:'+
                                  '{:.1f}/'.format(Success_batch.average)+
                                  '{:.1f}'.format(Precision_batch.average))
                Success_batch.reset()
                Precision_batch.reset()

    return Success_main.average, Precision_main.average



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--save_root_dir', type=str, default='./results/',  help='output folder')
    parser.add_argument('--data_dir', type=str, default = '/home/zhuangyi/SiamVoxel/kitti/training/',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'model_11.pth',  help='model name for training resume')
    parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="firstandprevious",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--vis', type=bool, default=False, help='set to True if dumping visualization')
    args = parser.parse_args()
    print (args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(args.save_root_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')), level=logging.INFO)
    logging.info('======================================================')

    args.manualSeed = 1
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    model = SiamPillar()
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, range(args.ngpu))
    if args.model != '':
        model.load_state_dict(torch.load(os.path.join(args.save_root_dir, args.model)), strict=False)
    model.cuda()
    print(model)
    torch.cuda.synchronize()
    # Car/Pedestrian/Van/Cyclist
    dataset_Test = SiameseTest(
            input_size=1024,
            path= args.data_dir,
            split='Test_tiny',
            category_name=args.category_name,
            offset_BB=0,
            scale_BB=1.25)

    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        collate_fn=lambda x: x,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    Success_run = AverageMeter()
    Precision_run = AverageMeter()

    if dataset_Test.isTiny():
        max_epoch = 2
    else:
        max_epoch = 1

    for epoch in range(max_epoch):
        Succ, Prec = test(
            test_loader,
            model,
            epoch=epoch + 1,
            shape_aggregation=args.shape_aggregation,
            reference_BB=args.reference_BB,
            IoU_Space=args.IoU_Space)
        Success_run.update(Succ)
        Precision_run.update(Prec)
        logging.info("mean Succ/Prec {}/{}".format(Success_run.avg,Precision_run.avg))
