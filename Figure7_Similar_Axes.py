from hessian_axis_visualize import vis_eigen_frame, vis_eigen_action, vis_distance_curve, vis_eigen_explore
from hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, plot_consistentcy_mat, \
    plot_consistency_hist, plot_consistency_example, plot_spectra
from GAN_utils import loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, StyleGAN2_wrapper, \
    loadStyleGAN, StyleGAN_wrapper, upconvGAN, PGGAN_wrapper, loadPGGAN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
figsumdir = r"E:\OneDrive - Washington University in St. Louis\Figure7"
"""Note the loading and visualization is fully deterministic, reproducible."""
#%% StyleGAN2 Face 1024
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "stylegan2-ffhq-config-f"
modelsnm = "Face1024"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, feat_col = data["eva_avg"], data["evc_avg"], data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )


#%% StyleGAN2 Face 512
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "ffhq-512-avg-tpurun1"
modelsnm = "Face512"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, feat_col = data["eva_avg"], data["evc_avg"], data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )


#%% StyleGAN2 Face 256
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "ffhq-256-config-e-003810"
modelsnm = "Face256"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, feat_col = data["eva_avg"], data["evc_avg"], data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )

#%% StyleGAN
Hessdir = join(rootdir, 'StyleGAN')
modelnm = "StyleGAN"
SGAN = loadStyleGAN()
SG = StyleGAN_wrapper(SGAN)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, feat_col = data["eva_avg"], data["evc_avg"], data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
vis_eigen_frame(evc_avg, eva_avg, SG, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN", transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )


#%% ProgGAN
Hessdir = join(rootdir, 'PGGAN')
modelnm = "PGGAN"
PGAN = loadPGGAN()
PG = PGGAN_wrapper(PGAN)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, feat_col = data["eva_avg"], data["evc_avg"], data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = PG.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
vis_eigen_frame(evc_avg, eva_avg, PG, ref_code=refvec, figdir=figsumdir, namestr="PGGAN", transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )

