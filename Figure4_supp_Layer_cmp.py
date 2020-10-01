import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from time import time
from os.path import join
import sys
import lpips
from GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper, upconvGAN
from hessian_analysis_tools import plot_spectra, compute_hess_corr
#%%
G = upconvGAN()
G.G.requires_grad_(False)
layernames = [name for name, _ in G.G.named_children()]
#%%
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\FC6GAN"
eva_col = []
evc_col = []
eva_ctrl_col = []
evc_ctrl_col = []
for Li in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    with np.load(join(datadir, "eig_Layer%d.npz" % (Li))) as data:
        eva_col.append(data["eva"].copy())
        evc_col.append(data["evc"].copy())
    with np.load(join(datadir, "eig_shfl_Layer%d.npz" % (Li))) as data:
        eva_ctrl_col.append(data["eva"].copy())
        evc_ctrl_col.append(data["evc"].copy())

#%%