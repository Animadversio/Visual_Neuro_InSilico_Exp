import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
# from cv2 import imread, imwrite
import matplotlib
matplotlib.use('Agg') # if you dont want image show up
import matplotlib.pylab as plt
import sys
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from caffenet import *
from hessian import hessian
#% Set up PerceptualLoss judger
sys.path.append(r"D:\Github\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
model_alex = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=1, gpu_ids=[0])
#%%
result_dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Artiphysiology\Sample_Diversity"
from torch_net_utils import load_generator, load_caffenet, visualize, preprocess
net_torch = load_caffenet()
G_torch = load_generator()
#%%
sigma = 3.0
codes = sigma * np.random.randn(40, 4096)
img_list = [visualize(G_torch, code, "cuda") for code in codes]
#%
dist_mat = np.zeros((len(codes), len(codes)))
for i in range(len(codes)):
    for j in range(len(codes)):
        dist = model_squ.forward(img_list[i].unsqueeze(0).permute(0,3,1,2), img_list[j].unsqueeze(0).permute(0,3,1,2), normalize=True)
        dist_mat[i, j] = dist.squeeze().detach().cpu().numpy()
dist_mat.mean()
#%%
basis = 5 * np.random.randn(1, 4096)
sigma = 3.0
codes = sigma * np.random.randn(40, 4096) + basis
img_list = [visualize(G_torch, code, "cuda") for code in codes]
dist_mat2 = np.zeros((len(codes), len(codes)))
for i in range(len(codes)):
    for j in range(len(codes)):
        dist = model_squ.forward(img_list[i].unsqueeze(0).permute(0,3,1,2), img_list[j].unsqueeze(0).permute(0,3,1,2), normalize=True)
        dist_mat2[i, j] = dist.squeeze().detach().cpu().numpy()
#%
dist_mat2.mean()
#%%
basis = 8 * np.random.randn(1, 4096)
sigma = 3.0
codes3 = sigma * np.random.randn(40, 4096) + basis
img_list = [visualize(G_torch, code, "cuda") for code in codes3]
dist_mat3 = np.zeros((len(codes), len(codes)))
for i in range(len(codes3)):
    for j in range(len(codes3)):
        dist = model_squ.forward(img_list[i].unsqueeze(0).permute(0,3,1,2), img_list[j].unsqueeze(0).permute(0, 3, 1, 2), normalize=True)
        dist_mat3[i, j] = dist.squeeze().detach().cpu().numpy()
#%
dist_mat3.mean()
#%%
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1))
from imageio import imread
img = imread(r"D:\Github\CMAES_optimizer_matlab\fc8_02.jpg")
#%%
img_tsr = torch.from_numpy(img).float()
img_tsr = img_tsr.unsqueeze(0).permute([0, 3, 1, 2])
resz_out_img = F.interpolate(img_tsr[:, [2, 1, 0], :, :] - BGR_mean, (227, 227), mode='bilinear', align_corners=True)
blobs_CNN = Caffenet(resz_out_img)
activ = blobs_CNN['fc8'][0, 1]
print(activ)
# Seems the caffe model and torch model match the score and activation < 1E-3 error. So torch and caffe can be used interchangeably
# seems the same image will generate different number of activation in the network!
#%%
import matplotlib.pylab as plt
plt.figure()
plt.imshow(img)
plt.show()
#%%
img_torch = visualize(G_torch, code)