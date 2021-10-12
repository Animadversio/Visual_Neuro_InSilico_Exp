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
from Hessian.GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper, upconvGAN
from Hessian.hessian_analysis_tools import plot_spectra, compute_hess_corr
#%%
G = upconvGAN()
G.G.requires_grad_(False)
layernames = np.array([name for name, _ in G.G.named_children()])
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
layernames_ = layernames[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\FC6GAN"
from Hessian.hessian_analysis_tools import plot_layer_amplif_curves, plot_layer_amplif_consistency, compute_plot_layer_corr_mat
plot_layer_amplif_curves(eva_col, evc_col, None, layernames=layernames_, savestr="FC6", figdir=datadir)
plot_layer_amplif_curves(eva_col, evc_col, None, layernames=layernames_, savestr="FC6", maxnorm=True, figdir=datadir)
#%%
plot_layer_amplif_consistency(eva_col, evc_col, layernames_, layeridx=[0,2,4], titstr="FC6", figdir=datadir,
                                   savelabel="FC6")
plot_layer_amplif_consistency(eva_col, evc_col, layernames_, layeridx=[1,9,12], titstr="FC6", figdir=datadir,
                                   savelabel="FC6")
#%%
plot_layer_amplif_curves(eva_ctrl_col, evc_ctrl_col, None, layernames=layernames_, savestr="FC6_ctrl", figdir=datadir, use_cuda=True)
plot_layer_amplif_curves(eva_ctrl_col, evc_ctrl_col, None, layernames=layernames_, savestr="FC6_ctrl", maxnorm=True, figdir=datadir, use_cuda=True)

#%%
corr_mat_lin, corr_mat_log, log_reg_slope, log_reg_intcp, _, _, _, _, = compute_plot_layer_corr_mat(eva_col, evc_col, None, layernames_, titstr="FC6", savestr="FC6", figdir=datadir, use_cuda=True)
corr_mat_lin_ctrl, corr_mat_log_ctrl, log_reg_slope_ctrl, log_reg_intcp_ctrl, _, _, _, _, = compute_plot_layer_corr_mat(eva_ctrl_col, evc_ctrl_col, None, layernames_, titstr="FC6_ctrl", savestr="FC6_ctrl", figdir=datadir, use_cuda=True)

#%%

ctrldatadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2\ctrl_Hessians"
realdatadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2\real_Hessians"
eva_col = []
evc_col = []
eva_ctrl_col = []
evc_ctrl_col = []
with np.load(join(realdatadir, "eig_style_trial100.npz")) as data:
    eva_col.append(data["eva"].copy())
    evc_col.append(data["evc"].copy())
with np.load(join(ctrldatadir, "eig_style_trial100.npz")) as data:
    eva_ctrl_col.append(data["eva"].copy())
    evc_ctrl_col.append(data["evc"].copy())
for Li in range(14):
    with np.load(join(realdatadir, "eig_genBlock%02d_trial100.npz" % (Li))) as data:
        eva_col.append(data["eva"].copy())
        evc_col.append(data["evc"].copy())
    with np.load(join(ctrldatadir, "eig_genBlock%02d_trial100.npz" % (Li))) as data:
        eva_ctrl_col.append(data["eva"].copy())
        evc_ctrl_col.append(data["evc"].copy())
with np.load(join(realdatadir, "eig_full_trial100.npz")) as data:
    eva_col.append(data["eva"].copy())
    evc_col.append(data["evc"].copy())
with np.load(join(ctrldatadir, "eig_full_trial100.npz")) as data:
    eva_ctrl_col.append(data["eva"].copy())
    evc_ctrl_col.append(data["evc"].copy())
#%%
layernames = ["MLP"] + [("StyleBlock%02d" % blocki) for blocki in range(14)] + ["Image"]
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
# from hessian_analysis_tools import plot_layer_amplif_curves, plot_layer_amplif_consistency, compute_plot_layer_corr_mat
plot_layer_amplif_curves(eva_col, evc_col, None, layernames=layernames, savestr="StyleGAN2_face512", figdir=figdir)
plot_layer_amplif_curves(eva_col, evc_col, None, layernames=layernames, savestr="StyleGAN2_face512", maxnorm=True, figdir=figdir)
plot_layer_amplif_consistency(eva_col, evc_col, layernames, layeridx=[0,1,2], titstr="StyleGAN2_face512", figdir=figdir, savelabel="StyleGAN2_face512")
plot_layer_amplif_consistency(eva_col, evc_col, layernames, layeridx=[1,9,12], titstr="StyleGAN2_face512", figdir=figdir, savelabel="StyleGAN2_face512")

plot_layer_amplif_curves(eva_ctrl_col, evc_ctrl_col, None, layernames=layernames, savestr="StyleGAN2_face512_ctrl", figdir=figdir)
plot_layer_amplif_curves(eva_ctrl_col, evc_ctrl_col, None, layernames=layernames, savestr="StyleGAN2_face512_ctrl", maxnorm=True, figdir=figdir)
plot_layer_amplif_consistency(eva_ctrl_col, evc_ctrl_col, layernames, layeridx=[0,1,2], titstr="StyleGAN2_face512_ctrl", figdir=figdir, savelabel="StyleGAN2_face512_ctrl")
plot_layer_amplif_consistency(eva_ctrl_col, evc_ctrl_col, layernames, layeridx=[1,9,12], titstr="StyleGAN2_face512_ctrl", figdir=figdir, savelabel="StyleGAN2_face512_ctrl")
#%%
corr_mat_lin, corr_mat_log, log_reg_slope, log_reg_intcp, _, _, _, _, = compute_plot_layer_corr_mat(eva_col, evc_col, None, layernames, titstr="StyleGAN2_face512", savestr="StyleGAN2_face512", figdir=figdir, use_cuda=True)
corr_mat_lin_ctrl, corr_mat_log_ctrl, log_reg_slope_ctrl, log_reg_intcp_ctrl, _, _, _, _, = compute_plot_layer_corr_mat(eva_ctrl_col, evc_ctrl_col, None, layernames, titstr="StyleGAN2_face512_ctrl", savestr="StyleGAN2_face512_ctrl", figdir=figdir, use_cuda=True)
