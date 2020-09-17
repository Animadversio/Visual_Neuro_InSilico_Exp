from hessian_axis_visualize import vis_eigen_frame, vis_eigen_action
from GAN_utils import loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, StyleGAN2_wrapper, upconvGAN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
#%%
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
#%% BigGAN
"""Load up BigGAN and create montages of images"""
BGAN = loadBigGAN("biggan-deep-256").cuda()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
figdir = join(rootdir, 'BigGAN')
Hessdir = join(rootdir, 'BigGAN')
data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
evc_nois = data['eigvects_nois_avg']
evc_clas = data['eigvects_clas_avg']
evc_clas_f = np.vstack((np.zeros((128, 128)), evc_clas, ))
evc_nois_f = np.vstack((evc_nois, np.zeros((128, 128)), ))
#%% Class specific visualization
# refvecs = np.vstack((0.5*np.random.randn(128,10)), EmbedMat[:,np.random.randint(0, 1000, 10)]).T
# class_id =  np.random.randint(0, 1000, 1) #[752] #
refvec = np.vstack((0.8*np.random.randn(128,1), EmbedMat[:, class_id])).T
# Teddy bear 850
mtg = vis_eigen_frame(evc_clas_f, eva_BG, BG, ref_code=refvec, figdir=figdir, namestr="class_spect",
                eiglist=[0,2,4,6,8,10,15,30,40], maxdist=0.1, rown=5, transpose=True)
mtg = vis_eigen_frame(evc_nois_f, eva_BG, BG, ref_code=refvec, figdir=figdir, namestr="noise_spect_sph",
                eiglist=[0,2,4,6,8,10,20,40], maxdist=1.5, rown=3, transpose=True, sphere=True)
#%%
eigi = 5
tanvec = np.hstack((evc_nois[:, -eigi], np.zeros(128)))
refvecs = np.vstack((0.7*np.random.randn(128,10), EmbedMat[:,np.random.randint(0, 1000, 10)])).T
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=1, rown=5, transpose=False, namestr="eig_nois%d_sph"%eigi, sphere=True)
# using spherical exploration is much better than linear
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=0.4, rown=5, transpose=False, namestr="eig_nois%d"%eigi)
#%%
refvec = np.vstack((0.7*np.random.randn(128,1), EmbedMat[:,np.random.randint(0, 1000, 1)])).T
mtg, codes_all = vis_eigen_frame(evc_BG, eva_BG, BG, ref_code=refvec, figdir=figdir, namestr="spect_all",
                eiglist=[1,2,4,6,8,10,15,20,30,40,60,80,120], maxdist=0.4, rown=5, transpose=True)
