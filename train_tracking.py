import argparse
import os
import random
import time
import logging
import pdb

from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from model.loss import rpn_cross_entropy_balance, rpn_smoothL1, box_iou3d, focal_loss
from utils.anchors import cal_rpn_target, cal_anchors

from loader.Dataset import SiameseTrain
from model.model import SiamPillar
from config import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = '/home/zhuangyi/SiamVoxel/kitti/training/',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')


opt = parser.parse_args()
print (opt)

#torch.cuda.set_device(opt.main_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = opt.save_root_dir

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data

def tracking_collate(batch):

	t_vox_feature = []
	t_vox_number = []
	t_vox_coordinate = []
	s_vox_feature = []
	s_vox_number = []
	s_vox_coordinate = []
	sample_box = []

	for i, data in enumerate(batch):
		t_vox_feature.append(data[0])
		t_vox_number.append(data[1])
		t_vox_coordinate.append(np.pad(data[2], ((0, 0), (1, 0)), mode = 'constant', constant_values = i))
		s_vox_feature.append(data[3])
		s_vox_number.append(data[4])
		s_vox_coordinate.append(np.pad(data[5], ((0, 0), (1, 0)), mode = 'constant', constant_values = i))
		sample_box.append(data[6])

	return torch.from_numpy(np.concatenate(t_vox_feature, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(t_vox_number, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(t_vox_coordinate, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(s_vox_feature, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(s_vox_number, axis=0)).float(),\
		   torch.from_numpy(np.concatenate(s_vox_coordinate, axis=0)).float(),\
		   np.array(sample_box)



train_data = SiameseTrain(
            input_size=1024,
            path= opt.data_dir,
            split='Train_tiny',
            category_name=opt.category_name,
            offset_BB=0,
            scale_BB=1.25)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
	collate_fn=tracking_collate,
    pin_memory=True)

test_data = SiameseTrain(
    input_size=1024,
    path=opt.data_dir,
    split='Valid_tiny',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=int(opt.batchSize/2),
    shuffle=False,
    num_workers=int(opt.workers),
	collate_fn=tracking_collate,
    pin_memory=True)

										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
model = SiamPillar()
if opt.ngpu > 1:
	model = torch.nn.DataParallel(model, range(opt.ngpu))
if opt.model != '':
	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)), strict=False)
	  
model.cuda()
print(model)


optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas = (0.9, 0.999), eps=1e-08)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)

# 3. Training and testing
for epoch in range(opt.nepoch):
	scheduler.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
#	# 3.1 switch to train mode
	torch.cuda.synchronize()
	model.train()
	train_mse = 0.0
	timer = time.time()

	batch_correct = 0.0
	batch_cla_loss = 0.0
	batch_reg_loss = 0.0
	batch_cla_pos_loss = 0.0
	batch_cla_neg_loss = 0.0
	batch_label_loss = 0.0
	batch_box_loss = 0.0
	batch_num = 0.0
	batch_iou = 0.0
	batch_true_correct = 0.0
	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		t_vox_feature, t_vox_number, t_vox_coordinate, \
		s_vox_feature, s_vox_number, s_vox_coordinate, sample_box = data
		t_vox_feature = Variable(t_vox_feature, requires_grad=False).cuda()
		t_vox_number = Variable(t_vox_number, requires_grad=False).cuda()
		t_vox_coordinate = Variable(t_vox_coordinate, requires_grad=False).cuda()
		s_vox_feature = Variable(s_vox_feature, requires_grad=False).cuda()
		s_vox_number = Variable(s_vox_number, requires_grad=False).cuda()
		s_vox_coordinate = Variable(s_vox_coordinate, requires_grad=False).cuda()


		anchors = cal_anchors()  # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)
		pos_equal_one, targets = cal_rpn_target(sample_box, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT], anchors, coordinate='lidar')
		pos_equal_one = torch.from_numpy(pos_equal_one).float()
		targets = torch.from_numpy(targets).float()
		pos_equal_one = Variable(pos_equal_one, requires_grad=False).cuda()
		targets = Variable(targets, requires_grad=False).cuda()


		# 3.1.2 compute output
		optimizer.zero_grad()
		pred_conf, pred_reg = model(len(sample_box), t_vox_feature, t_vox_number, t_vox_coordinate, \
									s_vox_feature, s_vox_number, s_vox_coordinate)
		cls_loss, pcls_loss, ncls_loss = focal_loss(pred_conf, pos_equal_one)
		#cls_loss, pcls_loss, ncls_loss = rpn_cross_entropy_balance(pred_conf, pos_equal_one)
		reg_loss = rpn_smoothL1(pred_reg, targets, pos_equal_one)
		box_loss = box_iou3d(pred_reg, targets, anchors, pos_equal_one)
		#loss_label = criterion_cla(pred_seed, label_cla)
		#loss_box = criterion_box(pred_offset, label_reg)
		#loss_box = (loss_box.mean(2) * label_cla).sum()/(label_cla.sum()+1e-06)


		loss = cls_loss + 5 * reg_loss + 0.1 * box_loss

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		# estimation_cla_cpu = seed_pediction.sigmoid().detach().cpu().numpy()
		# label_cla_cpu = label_cla.detach().cpu().numpy()
		# correct = float(np.sum((estimation_cla_cpu[0:len(label_point_set),:] > 0.4) == label_cla_cpu[0:len(label_point_set),:])) / 169.0
		# true_correct = float(np.sum((np.float32(estimation_cla_cpu[0:len(label_point_set),:] > 0.4) + label_cla_cpu[0:len(label_point_set),:]) == 2)/(np.sum(label_cla_cpu[0:len(label_point_set),:])))
					
		train_mse = train_mse + loss.data*len(sample_box)
		# batch_correct += correct
		batch_cla_loss += cls_loss.data
		batch_reg_loss += reg_loss.data
		batch_cla_pos_loss += pcls_loss
		batch_cla_neg_loss += ncls_loss
		batch_box_loss += box_loss.data
		# batch_num += len(label_point_set)
		# batch_true_correct += true_correct
		if (i+1)%20 == 0:
			print('\n ---- batch: %03d ----' % (i+1))
			print('cla_loss: %f, reg_loss: %f, cla_pos_loss: %f, cls_neg_loss: %f, box_loss: %f' %
				  (batch_cla_loss/20, batch_reg_loss/20, batch_cla_pos_loss/20, batch_cla_neg_loss/20, batch_box_loss/20))
			# print('accuracy: %f' % (batch_correct / float(batch_num)))
			# print('true accuracy: %f' % (batch_true_correct / 20))
			batch_label_loss = 0.0
			batch_cla_loss = 0.0
			batch_reg_loss = 0.0
			batch_cla_pos_loss = 0.0
			batch_cla_neg_loss = 0.0
			batch_box_loss = 0.0
			batch_num = 0.0
			batch_true_correct = 0.0
           
	# time taken
	train_mse = train_mse/len(train_data)
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	torch.save(model.state_dict(), '%s/model_%d.pth' % (save_dir, epoch))
	torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
	
	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	model.eval()
	test_cla_loss = 0.0
	test_reg_loss = 0.0
	test_cla_pos_loss = 0.0
	test_cla_neg_loss = 0.0
	test_label_loss = 0.0
	test_box_loss = 0.0
	test_correct = 0.0
	test_true_correct = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, 0)):
		torch.cuda.synchronize()
		# 3.2.1 load inputs and targets
		t_vox_feature, t_vox_number, t_vox_coordinate, \
		s_vox_feature, s_vox_number, s_vox_coordinate, sample_box = data
		t_vox_feature = Variable(t_vox_feature, requires_grad=False).cuda()
		t_vox_number = Variable(t_vox_number, requires_grad=False).cuda()
		t_vox_coordinate = Variable(t_vox_coordinate, requires_grad=False).cuda()
		s_vox_feature = Variable(s_vox_feature, requires_grad=False).cuda()
		s_vox_number = Variable(s_vox_number, requires_grad=False).cuda()
		s_vox_coordinate = Variable(s_vox_coordinate, requires_grad=False).cuda()


		anchors = cal_anchors()  # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)
		pos_equal_one, targets = cal_rpn_target(sample_box, [cfg.FEATURE_WIDTH, cfg.FEATURE_HEIGHT], anchors, coordinate='lidar')
		pos_equal_one = torch.from_numpy(pos_equal_one).float()
		targets = torch.from_numpy(targets).float()
		pos_equal_one = Variable(pos_equal_one, requires_grad=False).cuda()
		targets = Variable(targets, requires_grad=False).cuda()

		# 3.2.2 compute output
		pred_conf, pred_reg = model(len(sample_box), t_vox_feature, t_vox_number,
									t_vox_coordinate, \
									s_vox_feature, s_vox_number, s_vox_coordinate)
		cls_loss, pcls_loss, ncls_loss = focal_loss(pred_conf, pos_equal_one)
		#cls_loss, pcls_loss, ncls_loss = rpn_cross_entropy_balance(pred_conf, pos_equal_one)
		reg_loss = rpn_smoothL1(pred_reg, targets, pos_equal_one)
		box_loss = box_iou3d(pred_reg, targets, anchors, pos_equal_one)

		#loss_label = criterion_cla(pred_seed, label_cla)
		#loss_box = criterion_box(pred_offset, label_reg)
		#loss_box = (loss_box.mean(2) * label_cla).sum() / (label_cla.sum() + 1e-06)

		loss = cls_loss + 5 * reg_loss + 0.1 * box_loss

		torch.cuda.synchronize()
		test_cla_loss = test_cla_loss + cls_loss.data*len(sample_box)
		test_reg_loss = test_reg_loss + reg_loss.data*len(sample_box)
		test_cla_pos_loss = test_cla_pos_loss + pcls_loss.data*len(sample_box)
		test_cla_neg_loss = test_cla_neg_loss + ncls_loss.data*len(sample_box)
		test_box_loss = test_box_loss + box_loss.data*len(sample_box)
		# estimation_cla_cpu = seed_pediction.sigmoid().detach().cpu().numpy()
		# label_cla_cpu = label_cla.detach().cpu().numpy()
		# correct = float(np.sum((estimation_cla_cpu[0:len(label_point_set),:] > 0.4) == label_cla_cpu[0:len(label_point_set),:])) / 169.0
		# true_correct = float(np.sum((np.float32(estimation_cla_cpu[0:len(label_point_set),:] > 0.4) + label_cla_cpu[0:len(label_point_set),:]) == 2)/(np.sum(label_cla_cpu[0:len(label_point_set),:])))
		# test_correct += correct
		# test_true_correct += true_correct*len(label_point_set)

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	# print mse
	test_cla_loss = test_cla_loss / len(test_data)
	test_reg_loss = test_reg_loss / len(test_data)
	test_cla_pos_loss = test_cla_pos_loss / len(test_data)
	test_cla_neg_loss = test_cla_neg_loss / len(test_data)
	test_label_loss = test_label_loss / len(test_data)
	test_box_loss = test_box_loss / len(test_data)
	print('cla_loss: %f, reg_loss: %f, box_loss: %f, #test_data = %d' %(test_cla_loss, test_reg_loss, test_box_loss, len(test_data)))
	# test_correct = test_correct / len(test_data)
	# print('mean-correct of 1 sample: %f, #test_data = %d' %(test_correct, len(test_data)))
	# test_true_correct = test_true_correct / len(test_data)
	# print('true correct of 1 sample: %f' %(test_true_correct))
	# log
	logging.info('Epoch#%d: train error=%e, test error=%e, %e, %e, lr = %f' %(epoch, train_mse, test_cla_loss, test_reg_loss, test_box_loss, scheduler.get_lr()[0]))
