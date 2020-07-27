import utils
import net_utils
import utils
from utils import load_GAN
from Generator import Generator
from time import time, sleep
import numpy as np
from Optimizer import CholeskyCMAES, Genetic, Optimizer  # Optimizer is the base class for these things
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from os.path import join
from sys import platform
#%% Decide the result storage place based on the computer the code is running
if platform == "linux": # cluster
    recorddir = "/scratch/binxu/CNN_data/"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
        initcodedir = r"D:\Generator_DB_Windows\stimuli\texture006"  # Code & image folder to initialize the Genetic Algorithm
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        recorddir = r"E:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
# Basic properties for Optimizer.
#%%
import torch
from torch_net_utils import load_generator
G = load_generator("fc7")
G.requires_grad_(False)
G.cuda()
#%%
def visualize(G, code, mode="cuda"):
    """Do the De-caffe transform (Validated)
    works for a single code """
    if mode == "cpu":
        blobs = G(code)
    else:
        blobs = G(code.cuda())
    out_img = blobs['deconv0']  # get raw output image from GAN
    if mode == "cpu":
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    else:
        clamp_out_img = torch.clamp(out_img + BGR_mean.cuda(), 0, 255)
    vis_img = clamp_out_img[:, [2, 1, 0], :, :].permute([2, 3, 1, 0]).squeeze() / 255
    return vis_img

BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1))
#%%
code = np.random.randn(4096)
code = code.reshape(-1, 4096)
feat = torch.from_numpy(code).float().requires_grad_(True)
img = visualize(G, feat)
#%%
from imageio import imread
import matplotlib.pylab as plt
from skimage.transform import resize
target_img = imread(r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\Cat.jpg")
tsr_target = target_img.astype(float)/255
rsz_target = resize(tsr_target, (256, 256), anti_aliasing=True)
tsr_target = torch.from_numpy(rsz_target).cuda()
# target_img =
#%%
plt.imshow(tsr_target.cpu())
plt.show()
tsr_target.cuda()
#%%
plt.imshow(img.detach().cpu().numpy())
plt.show()
#%%
code = np.random.randn(4096)
code = code.reshape(-1, 4096)
feat = torch.from_numpy(code).float().requires_grad_(True)
feat.cuda()
optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
for i in range(150):
    optimizer.zero_grad()
    img = visualize(G, feat)
    loss = (img - tsr_target).abs().sum(axis=2).mean()
    loss.backward()
    optimizer.step()
    print("step%d" % i, loss)
#%%
plt.imshow(img.detach().cpu().numpy())
plt.show()