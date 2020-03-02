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
from torch_net_utils import load_generator, load_caffenet, visualize
Caffenet = load_caffenet()
Generator = load_generator()
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
from insilico_Exp import CNNmodel
net = CNNmodel("caffe-net");net.select_unit(('caffe-net', 'fc8', 1))
net.score(img[np.newaxis,])
#%%
import matplotlib.pylab as plt
plt.figure()
plt.imshow(img)
plt.show()