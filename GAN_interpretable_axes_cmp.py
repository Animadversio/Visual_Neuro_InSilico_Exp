from GAN_utils import loadPGGAN, loadDCGAN, loadStyleGAN2, loadBigGAN, loadBigBiGAN, upconvGAN
from GAN_utils import PGGAN_wrapper, DCGAN_wrapper, StyleGAN2_wrapper, BigGAN_wrapper, BigBiGAN_wrapper

from hessian_analysis_tools import average_H, plot_spectra
from hessian_axis_visualize import vis_eigen_action, vis_distance_curve
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
import os
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
#%% Due to naming convention conflict, StyleGAN need to be loaded in a seperate session from other GANs. 
# from GAN_utils import loadStyleGAN, StyleGAN_wrapper
#%%
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
axesdir = r"E:\OneDrive - Washington University in St. Louis\Hess_Spect_Interpret"
os.makedirs(axesdir, exist_ok=True)
#%%
"""BigGAN"""
figdir = join(rootdir, 'BigGAN')
Hessdir = join(rootdir, 'BigGAN')
BGAN = loadBigGAN()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
H_BG = data['H_avg']
#%%

#%% """StyleGAN2"""
Hessdir = join(rootdir, 'StyleGAN2')
modellist = ["ffhq-256-config-e-003810",
			"ffhq-512-avg-tpurun1",
		    "stylegan2-ffhq-config-f",
			"stylegan2-cat-config-f", 
			"2020-01-11-skylion-stylegan2-animeportraits",
			 "model.ckpt-533504"]
modelsnms = ["Face256",
			 "Face512",
			 "Face1024",
			"Cat256", 
			"Anime",
			"ImageNet512",
			"Car"]
#%% for modelnm, modelsnm in zip(modellist, modelsnms):
modelnm, modelsnm = modellist[1], modelsnms[1]
SGAN = loadStyleGAN2(modelnm+".pt", size=256)
SG = StyleGAN2_wrapper(SGAN)
figdir = join(axesdir, "StyleGAN2", modelsnm)
os.makedirs(figdir, exist_ok=True)
data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg = data["H_avg"]
eva_avg = data["eva_avg"]
evc_avg = data["evc_avg"]
#%%
RND = np.random.randint(1E4)
ref_codes = 0.7*np.random.randn(8, 512)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG2_%s_eig%d_lin"%(modelsnm, eigidx+1),
    			maxdist=2.5, rown=7, sphere=False, transpose=False, RND=RND)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG2_%s_eig%d_sph"%(modelsnm, eigidx+1),
    			maxdist=0.2, rown=7, sphere=True, transpose=False, RND=RND)

#%%

