from GAN_utils import loadPGGAN, loadDCGAN, loadStyleGAN2, loadBigGAN, loadBigBiGAN, upconvGAN
from GAN_utils import PGGAN_wrapper, DCGAN_wrapper, StyleGAN2_wrapper, BigGAN_wrapper, BigBiGAN_wrapper
#%% Due to naming convention conflict, StyleGAN need to be loaded in a seperate session from other GANs.
from GAN_utils import loadStyleGAN, StyleGAN_wrapper
# %%
from hessian_analysis_tools import average_H, plot_spectra
from hessian_axis_visualize import vis_eigen_action, vis_distance_curve, vis_eigen_frame
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
axesdir = r"E:\OneDrive - Washington University in St. Louis\Hess_Spect_Interpret"
os.makedirs(axesdir, exist_ok=True)
#%%

#%% """StyleGAN2"""
Hessdir = join(rootdir, 'StyleGAN2')
modellist = ["ffhq-256-config-e-003810",
			"ffhq-512-avg-tpurun1",
			"stylegan2-cat-config-f", 
			"2020-01-11-skylion-stylegan2-animeportraits",
			 "model.ckpt-533504",
		    "stylegan2-ffhq-config-f",
			 "stylegan2-car-config-f"]
modelsnms = ["Face256",
			 "Face512",
			"Cat256", 
			"Anime",
			"ImageNet512",
			"Face1024",
			"Car512"]
#%%
for modelnm, modelsnm in zip(modellist[6:], modelsnms[6:]):
# modelnm, modelsnm = modellist[1], modelsnms[1]
	SGAN = loadStyleGAN2(modelnm+".pt") # , size=256
	SG = StyleGAN2_wrapper(SGAN)
	figdir = join(axesdir, "StyleGAN2", modelsnm)
	os.makedirs(figdir, exist_ok=True)
	data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
	H_avg = data["H_avg"]
	eva_avg = data["eva_avg"]
	evc_avg = data["evc_avg"]
	#%
	RND = np.random.randint(1E4)
	ref_codes = 0.7*np.random.randn(8, 512)
	for eigidx in range(40):
		vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG2_%s_eig%d_lin"%(modelsnm, eigidx+1),
					maxdist=3.5, rown=7, sphere=False, transpose=False, RND=RND)
	for eigidx in range(40):
		vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG2_%s_eig%d_sph"%(modelsnm, eigidx+1),
					maxdist=0.35, rown=7, sphere=True, transpose=False, RND=RND)
#%%
for modelnm, modelsnm in zip(modellist[0:], modelsnms[0:]):
# modelnm, modelsnm = modellist[1], modelsnms[1]
	SGAN = loadStyleGAN2(modelnm+".pt") # , size=256
	SG = StyleGAN2_wrapper(SGAN)
	figdir = join(axesdir, "StyleGAN2", modelsnm)
	os.makedirs(figdir, exist_ok=True)
	data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
	H_avg = data["H_avg"]
	eva_avg = data["eva_avg"]
	evc_avg = data["evc_avg"]
	vis_eigen_frame(evc_avg, eva_avg, SG, figdir=figdir, namestr="SG2_%s_orig" % (modelsnm, ), page_B=10,
					eig_rng=(0, 60), maxdist=9, rown=5, sphere=False, transpose=False, RND=0)
	# for eigidx in range(40):
	# 	vis_eigen_action(evc_avg[:, -eigidx - 1], ref_codes, SG, figdir=figdir,
	# 					 namestr="SG2_%s_eig%d_orig" % (modelsnm, eigidx + 1),
	# 					 maxdist=9, rown=7, sphere=False, transpose=False, RND=0)

#%% PGGAN
PGGAN = loadPGGAN() # , size=256
PG = PGGAN_wrapper(PGGAN)
figdir = join(axesdir, "PGGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "PGGAN", "H_avg_%s.npz"%"PGGAN"))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg = data["H_avg"]
eva_avg = data["eva_avg"]
evc_avg = data["evc_avg"]
#%%
RND = np.random.randint(1E4)
ref_codes = np.random.randn(8, 512)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, PG, figdir=figdir, namestr="PGG_eig%d_lin"%(eigidx+1),
				maxdist=2.5, rown=7, sphere=False, transpose=False, RND=RND)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, PG, figdir=figdir, namestr="PGG_eig%d_sph"%(eigidx+1),
				maxdist=0.2, rown=7, sphere=True, transpose=False, RND=RND)
#%%
for eigidx in range(40,80):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, PG, figdir=figdir, namestr="PGG_eig%d_lin"%(eigidx+1),
				maxdist=4.5, rown=7, sphere=False, transpose=False, RND=RND)
for eigidx in range(40,80):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, PG, figdir=figdir, namestr="PGG_eig%d_sph"%(eigidx+1),
				maxdist=1, rown=7, sphere=True, transpose=False, RND=RND)
#%%
vis_eigen_frame(evc_avg, eva_avg, PG, figdir=figdir, namestr="PGG_orig", page_B=15,
					eig_rng=(0, 120), maxdist=9, rown=5, sphere=False, transpose=False, RND=0)
#%%
vis_eigen_frame(evc_avg, eva_avg, PG, figdir=figdir, namestr="PGG_orig", page_B=15,
					eig_rng=(0, 120), maxdist=22, rown=7, sphere=False, transpose=False, RND=0)


#%% StyleGAN
SGGAN = loadStyleGAN() # , size=256
SG = StyleGAN_wrapper(SGGAN)
figdir = join(axesdir, "StyleGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "StyleGAN", "H_avg_%s.npz"%"StyleGAN"))
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%%
RND = np.random.randint(1E4)
ref_codes = np.random.randn(8, 512)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG_eig%d_lin"%(eigidx+1),
				maxdist=2.5, rown=7, sphere=False, transpose=False, RND=RND)
for eigidx in range(40):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, SG, figdir=figdir, namestr="SG_eig%d_sph"%(eigidx+1),
				maxdist=0.2, rown=7, sphere=True, transpose=False, RND=RND)
#%%
vis_eigen_frame(evc_avg, eva_avg, SG, figdir=figdir, namestr="SG1_orig", page_B=15,
					eig_rng=(0, 120), maxdist=9, rown=5, sphere=False, transpose=False, RND=0)
vis_eigen_frame(evc_avg, eva_avg, SG, figdir=figdir, namestr="SG1_orig", page_B=15,
					eig_rng=(0, 120), maxdist=22, rown=7, sphere=False, transpose=False, RND=1)

#%% BigBiGAN
BBGAN = loadBigBiGAN()
BBG = BigBiGAN_wrapper(BBGAN)
figdir = join(axesdir, "BigBiGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "BigBiGAN", "H_avg_%s.npz"%"BigBiGAN"))
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%%
vis_eigen_frame(evc_avg, eva_avg, BBG, figdir=figdir, namestr="BBG_orig", page_B=15,
					eig_rng=(0, 60), maxdist=8, rown=5, sphere=False, transpose=False, RND=None)
#%%
RND = np.random.randint(1E4)
ref_codes = np.random.randn(10, 120)
for eigidx in range(60):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, BBG, figdir=figdir, namestr="BBG_eig%d_lin"%(eigidx+1),
				maxdist=2, rown=7, sphere=False, transpose=False, RND=RND)
	# if eigidx==10:break
for eigidx in range(60):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, BBG, figdir=figdir, namestr="BBG_eig%d_sph"%(eigidx+1),
				maxdist=0.4, rown=7, sphere=True, transpose=False, RND=RND)
	# if eigidx==10:break

#%% BigGAN
"""BigGAN"""
BGAN = loadBigGAN()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
figdir = join(axesdir, 'BigGAN')
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, 'BigGAN', "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
H_BG = data['H_avg']
evc_nois = data['eigvects_nois_avg']
evc_clas = data['eigvects_clas_avg']
eva_nois = data['eigvals_nois_avg']
eva_clas = data['eigvals_clas_avg']
evc_clas_f = np.vstack((np.zeros((128, 128)), evc_clas, ))
evc_nois_f = np.vstack((evc_nois, np.zeros((128, 128)), ))
#%%
ref_codes = np.vstack((0.7*np.random.randn(128, 1), EmbedMat[:, [1]])).T
refvec = torch.from_numpy(ref_codes,).float()
refvec.requires_grad_(True)
#%%
img = BG.visualize(refvec.cuda())
#%%
torch.autograd.grad()
#%% Start from origin
vis_eigen_frame(evc_BG, eva_BG, BG, figdir=figdir, namestr="BG_all_orig", page_B=15,
					eig_rng=(0, 120), maxdist=8, rown=5, sphere=False, transpose=False, RND=None)
vis_eigen_frame(evc_nois_f, eva_nois, BG, figdir=figdir, namestr="BG_nois_orig", page_B=15,
					eig_rng=(0, 120), maxdist=8, rown=5, sphere=False, transpose=False, RND=None)
vis_eigen_frame(evc_clas_f, eva_clas, BG, figdir=figdir, namestr="BG_clas_orig", page_B=15,
					eig_rng=(0, 120), maxdist=8, rown=5, sphere=False, transpose=False, RND=None)
#%% Action on random codes
samp_n = 8
classid = np.random.randint(0, 1000, samp_n)
ref_codes = np.vstack((0.7*np.random.randn(128, samp_n), EmbedMat[:, classid])).T
for eigidx in range(40):
	vis_eigen_action(evc_clas_f[:, -eigidx-1], ref_codes, BG, figdir=figdir, namestr="BG_clas_eig%d_lin"%(eigidx+1),
				maxdist=0.5, rown=7, sphere=False, transpose=False, RND=RND)
	# if eigidx==10:break
# for eigidx in range(40):
# 	vis_eigen_action(evc_clas_f[:, -eigidx-1], ref_codes, BG, figdir=figdir, namestr="BG_clas_eig%d_sph"%(eigidx+1),
# 				maxdist=1, rown=7, sphere=True, transpose=False, RND=RND)
# 	if eigidx==10:break
for eigidx in range(40):
	vis_eigen_action(evc_nois_f[:, -eigidx-1], ref_codes, BG, figdir=figdir, namestr="BG_nois_eig%d_lin"%(eigidx+1),
				maxdist=2.5, rown=7, sphere=False, transpose=False, RND=RND)
	# if eigidx==10:break
for eigidx in range(40):
	vis_eigen_action(evc_nois_f[:, -eigidx-1], ref_codes, BG, figdir=figdir, namestr="BG_nois_eig%d_sph"%(eigidx+1),
				maxdist=0.5, rown=7, sphere=True, transpose=False, RND=RND)
	# if eigidx == 10: break
#%% DCGAN
DCGAN = loadDCGAN()
DCG = DCGAN_wrapper(DCGAN)
figdir = join(axesdir, "DCGAN")
os.makedirs(figdir, exist_ok=True)
data = np.load(join(rootdir, "DCGAN", "H_avg_%s.npz"%"DCGAN"))
H_avg, eva_avg, evc_avg = data["H_avg"], data["eva_avg"], data["evc_avg"]
#%% Start from origin
vis_eigen_frame(evc_avg, eva_avg, DCG, figdir=figdir, namestr="DCG_orig", page_B=15,
					eig_rng=(0, 60), maxdist=8, rown=5, sphere=False, transpose=False, RND=None)
#%% Action on random codes
RND = np.random.randint(1E4)
ref_codes = np.random.randn(10, 120)
for eigidx in range(60):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, DCG, figdir=figdir, namestr="DCG_eig%d_lin"%(eigidx+1),
				maxdist=4.5, rown=7, sphere=False, transpose=False, RND=RND)
	# if eigidx==10:break

for eigidx in range(60):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, DCG, figdir=figdir, namestr="DCG_eig%d_sph"%(eigidx+1),
				maxdist=1, rown=7, sphere=True, transpose=False, RND=RND)
	# if eigidx==10:break
