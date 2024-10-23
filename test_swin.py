import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.net1_swin import TGLNet
# from Code.lib.net2_swin import TGLNet
from Code.utils.data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str, default='./test_data/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#load the model
model = TGLNet()
model.cuda()

model.load_state_dict(torch.load('./Checkpoint/'))
model.eval()

#test
test_datasets = ['NJUD','NLPR', 'SIP', 'STERE1000', 'SSD', 'LFSD']

for dataset in test_datasets:
    save_path = './finalmap/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image_root  = dataset_path + dataset + '/test_images/'
    gt_root     = dataset_path + dataset + '/test_masks/'
    depth_root  = dataset_path + dataset + '/test_depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        
        gt      = np.asarray(gt, np.float32)
        gt     /= (gt.max() + 1e-8)
        image   = image.cuda()
        depth   = depth.cuda()
        pre_res = model(image,depth)
        res     = pre_res[0]
        res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res     = res.sigmoid().data.cpu().numpy().squeeze()
        res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
