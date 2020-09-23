from GAN_utils import loadPGGAN, loadDCGAN, loadStyleGAN2, loadBigGAN, loadBigBiGAN, upconvGAN
from GAN_utils import PGGAN_wrapper, DCGAN_wrapper, StyleGAN2_wrapper, BigGAN_wrapper, BigBiGAN_wrapper

from hessian_analysis_tools import average_H, plot_spectra
from hessian_axis_visualize import vis_eigen_action, vis_distance_curve
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join

#%% Due to naming convention conflict, StyleGAN need to be loaded in a seperate session from other GANs. 
# from GAN_utils import loadStyleGAN, StyleGAN_wrapper
#%% StyleGAN2 
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
axesdir = r"E:\OneDrive - Washington University in St. Louis\Hess_Spect_Interpret"
figdir = join(axesdir, 'StyleGAN2')
Hessdir = join(rootdir, 'StyleGAN2')
os.makedirs(figdir, exist_ok=True)
modellist = ["ffhq-512-avg-tpurun1", 
			"stylegan2-cat-config-f", 
			"2020-01-11-skylion-stylegan2-animeportraits"]
modelsnms = ["Face512", 
			"Cat256", 
			"Anime"]

#%% for modelnm, modelsnm in zip(modellist, modelsnms):
modelnm, modelsnm = modellist[0], modelsnms[0]
data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg = data["H_avg"]
eva_avg = data["eva_avg"]
evc_avg = data["evc_avg"]

ref_codes = 0.7*np.random.randn(10, 512)
for eigidx in range(20):
	vis_eigen_action(evc_avg[:, -eigidx-1], ref_codes, G, figdir=figdir, namestr="SG2_%s_eig%d"%(modelsnm, eigidx+1), 
    			maxdist=1, rown=7, sphere=False, transpose=False)


#%% BigGAN
figdir = join(rootdir, 'BigGAN')
Hessdir = join(rootdir, 'BigGAN')
data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
H_BG = data['H_avg']