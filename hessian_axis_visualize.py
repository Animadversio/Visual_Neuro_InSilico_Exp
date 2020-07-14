import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
#%%
out_dir = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace"
with np.load(join(out_dir, "Pasu_Space_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]

# go through spectrum in batch, and plot B number of axis in a row
#%%
out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
with np.load(join(out_dir, "Evolution_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]