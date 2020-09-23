""""""
from GAN_utils import loadPGGAN, loadDCGAN, loadStyleGAN2, loadBigGAN, loadBigBiGAN, upconvGAN
from GAN_utils import PGGAN_wrapper, DCGAN_wrapper, StyleGAN2_wrapper, BigGAN_wrapper, BigBiGAN_wrapper

from hessian_analysis_tools import average_H, plot_spectra
from hessian_axis_visualize import vis_eigen_action, vis_distance_curve
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from torchvision.utils import make_grid
#%%

rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"

BGAN = loadBigGAN()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
data = np.load(join(rootdir, 'BigGAN', "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
H_BG = data['H_avg']
#%%
samp_n = 10
classid = np.random.randint(0, 1000, samp_n)
refvec = np.vstack((0.7*np.random.randn(128,samp_n), EmbedMat[:, classid])).T
cutoff = 128
refvec_proj = refvec@evc_BG[:,-cutoff:]@evc_BG[:,-cutoff:].T

orig_img = BG.visualize_batch_np(refvec)
proj_img = BG.visualize_batch_np(refvec_proj)
make_grid(torch.cat((orig_img, proj_img)), size=(2, samp_n))
#%%


