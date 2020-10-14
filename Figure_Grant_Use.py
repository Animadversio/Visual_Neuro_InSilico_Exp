"""Borrow code from GAN_interpretable_axes_cmp.py"""
from GAN_utils import loadPGGAN, loadDCGAN, loadStyleGAN2, loadBigGAN, loadBigBiGAN, upconvGAN
from GAN_utils import PGGAN_wrapper, DCGAN_wrapper, StyleGAN2_wrapper, BigGAN_wrapper, BigBiGAN_wrapper
#%% Due to naming convention conflict, StyleGAN need to be loaded in a seperate session from other GANs.
from GAN_utils import loadStyleGAN, StyleGAN_wrapper
# %%
from hessian_analysis_tools import average_H, plot_spectra
from hessian_axis_visualize import vis_eigen_action, vis_distance_curve, vis_eigen_frame, vis_eigen_explore, \
    vis_eigen_action_row, vis_eigen_explore_row
import numpy as np
import torch
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
import os
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
figsumdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\Figure_Grant_Use"
os.makedirs(figsumdir, exist_ok=True)
"""Note the loading and visualization is mostly deterministic, reproducible."""
#%% PGGAN
PGGAN = loadPGGAN()
PG = PGGAN_wrapper(PGGAN)
figdir = join(figsumdir, "PGGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "PGGAN", "H_avg_%s.npz"%"PGGAN"))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%%
refvecs = np.random.randn(2, 512)
mtg = vis_eigen_action_row(evc_avg[:,-1], refvecs, PG, figdir=figdir, sphere=False, maxdist=2.5, rown=5, transpose=True)
#%% StyleGAN
SGGAN = loadStyleGAN()
SG = StyleGAN_wrapper(SGGAN)
figdir = join(figsumdir, "StyleGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "StyleGAN", "H_avg_%s.npz"%"StyleGAN"))
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%%
refvecs = np.random.randn(2, 512)
mtg = vis_eigen_action_row(evc_avg[:,-1], refvecs, SG, figdir=figdir, namestr="", sphere=False, maxdist=2.5,
                           rown=5, transpose=True)
#%% StyleGAN2
Hessdir = join(rootdir, 'StyleGAN2')
modellist = ["ffhq-256-config-e-003810",
             "ffhq-512-avg-tpurun1",
             "stylegan2-ffhq-config-f",
             "2020-01-11-skylion-stylegan2-animeportraits",]
modelsnms = ["Face256",
             "Face512",
             "Face1024",
             "Anime",]
for modelnm, modelsnm in zip(modellist, modelsnms):
    SGAN = loadStyleGAN2(modelnm+".pt")
    SG = StyleGAN2_wrapper(SGAN)
    figdir = join(figsumdir, "StyleGAN2_" + modelsnm)
    os.makedirs(figdir, exist_ok=True)
    data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
    H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
    refvecs = np.random.randn(2, 512)
    mtg = vis_eigen_action_row(evc_avg[:, -1], refvecs, SG, figdir=figdir, namestr="evc_1", sphere=False, maxdist=2.5,
                               rown=5, transpose=True)
#%%
modelnm, modelsnm = "ffhq-512-avg-tpurun1", "Face512",
SGAN = loadStyleGAN2(modelnm+".pt")
SG = StyleGAN2_wrapper(SGAN)
figdir = join(figsumdir, "StyleGAN2_" + modelsnm)
os.makedirs(figdir, exist_ok=True)
data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%%
refvecs = np.random.randn(2, 512)
mtg = vis_eigen_action_row(evc_avg[:, -3], refvecs, SG, figdir=figdir, namestr="evc_2", sphere=False, maxdist=2.5,
                           rown=5, transpose=True)
