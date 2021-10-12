import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from time import time
from os.path import join
import os
import sys
from Hessian.GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, BigGAN_wrapper, loadStyleGAN2, StyleGAN2_wrapper, ckpt_root
from Hessian.hessian_analysis_tools import plot_spectra, compute_hess_corr
from Hessian.hessian_analysis_tools import scan_hess_npz, average_H, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example, compute_vector_hess_corr, compute_hess_corr
from lpips import LPIPS

saveroot = r"E:\Cluster_Backup\StyleGAN2"
summaryroot = join(saveroot, "summary")
modelnms = ["ffhq-256-config-e-003810", "ffhq-512-avg-tpurun1", "stylegan2-cat-config-f",]
shortnms = ["Face256", "Face512", "Cat512"]
#%% Comparing Spectrum
modelnm = "ffhq-256-config-e-003810"
modelsnm = "Face256"
# modelnm = "ffhq-512-avg-tpurun1"
# modelsnm = "Face512"
for modelnm, modelsnm in zip(modelnms, shortnms):
    label = modelnm+"_fix"
    eva_col = np.load(join(saveroot, "summary", label, "spectra_col_%s.npz"%label))["eigval_col"]
    label = modelnm+"_fix_ctrl"
    eva_col_ctrl = np.load(join(saveroot, "summary", label, "spectra_col_%s.npz"%label))["eigval_col"]
    label = modelnm+"_W_fix"
    eva_col_W = np.load(join(saveroot, "summary", label, "spectra_col_%s.npz"%label))["eigval_col"]
    label = modelnm+"_W_fix_ctrl"
    eva_col_W_ctrl = np.load(join(saveroot, "summary", label, "spectra_col_%s.npz"%label))["eigval_col"]

    fig0 = plot_spectra(eva_col, label="Fix", titstr="StyleGAN2 %s"%modelsnm,
                        savename="SG2_%s_shfl_spectrum_cmp" % modelsnm, figdir=summaryroot)
    fig0 = plot_spectra(eva_col_ctrl, label="Fix_Shuffled", titstr="StyleGAN2 %s"%modelsnm, fig=fig0,
                        savename="SG2_%s_shfl_spectrum_cmp_Zspace" % modelsnm, figdir=summaryroot)

    fig0 = plot_spectra(eva_col_W, label="Fix Wspace", titstr="StyleGAN2 %s"%modelsnm, fig=fig0,
                        savename="SG2_%s_shfl_spectrum_cmp" % modelsnm, figdir=summaryroot)
    fig0 = plot_spectra(eva_col_W_ctrl, label="Fix Wspace Shuffled", titstr="StyleGAN2 %s"%modelsnm, fig=fig0,
                        savename="SG2_%s_shfl_spectrum_cmp" % modelsnm, figdir=summaryroot)

    fig1 = plot_spectra(eva_col_W, label="Fix Wspace", titstr="StyleGAN2 %s"%modelsnm,
                        savename="SG2_%s_shfl_spectrum_cmp_Wspace" % modelsnm, figdir=summaryroot)
    fig1 = plot_spectra(eva_col_W_ctrl, label="Fix Wspace Shuffled", titstr="StyleGAN2 %s"%modelsnm, fig=fig1,
                        savename="SG2_%s_shfl_spectrum_cmp_Wspace" % modelsnm, figdir=summaryroot)
#%% Comparing Consistency?
modelnm = "ffhq-512-avg-tpurun1"
modelsnm = "Face512"
modelnm = "ffhq-256-config-e-003810"
modelsnm = "Face256"
for modelnm, modelsnm in zip(modelnms, shortnms):
    label = modelnm+"_fix"
    data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
    corr_mat_log, corr_mat_lin = data['corr_mat_log'], data['corr_mat_lin']
    label = modelnm+"_fix_ctrl"
    data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
    corr_mat_log_ctrl, corr_mat_lin_ctrl = data['corr_mat_log'], data['corr_mat_lin']
    fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=summaryroot, label="fix",
                                        savelabel="SG2_%s_shuffle_cmp_Zspace"%modelsnm)
    fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=summaryroot, label="fix_shuffle",
                                         savelabel="SG2_%s_shuffle_cmp_Zspace"%modelsnm, figs=(fig11,fig22))

    label = modelnm+"_W_fix"
    data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
    corr_mat_log, corr_mat_lin = data['corr_mat_log'], data['corr_mat_lin']
    label = modelnm+"_W_fix_ctrl"
    data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
    corr_mat_log_ctrl, corr_mat_lin_ctrl = data['corr_mat_log'], data['corr_mat_lin']
    fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=summaryroot, label="fix_W",
                                        savelabel="SG2_%s_shuffle_cmp_Wspace"%modelsnm)
    fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=summaryroot, label="fix_W_shuffle",
                                         savelabel="SG2_%s_shuffle_cmp_Wspace"%modelsnm, figs=(fig11,fig22))
#%% Plot some examples of Zspace Wspace and etc?
saveroot = r"E:\Cluster_Backup\StyleGAN"
summaryroot = join(saveroot, "summary")
"""StyleGAN W space compare """
modelnm = r"StyleGAN_Face256"
modelsnm = r"Face256"
# for modelnm, modelsnm in zip(modelnms, shortnms):
label = modelnm+"_fix"  # Z space fixed
eva_col = np.load(join(summaryroot, label, "spectra_col_%s.npz"%label))["eigval_col"]
label = modelnm+"_fix_ctrl"  # Z space fixed shuffled
eva_col_ctrl = np.load(join(summaryroot, label, "spectra_col_%s.npz"%label))["eigval_col"]
label = modelnm+"_W_fix"  # W space fixed
eva_col_W = np.load(join(summaryroot, label, "spectra_col_%s.npz"%label))["eigval_col"]
label = modelnm+"_W_fix_ctrl"  # W space fixed shuffled
eva_col_W_ctrl = np.load(join(summaryroot, label, "spectra_col_%s.npz"%label))["eigval_col"]
fig0 = plot_spectra(eva_col, label="Fix", titstr="StyleGAN %s" % modelsnm, fig=None,
                    savename="SG_%s_shfl_spectrum_cmp_Zspace" % modelsnm, figdir=summaryroot)
fig0 = plot_spectra(eva_col_ctrl, label="Fix_Shuffled", titstr="StyleGAN %s" % modelsnm, fig=fig0,
                    savename="SG_%s_shfl_spectrum_cmp_Zspace" % modelsnm, figdir=summaryroot)

fig0 = plot_spectra(eva_col_W, label="Fix Wspace", titstr="StyleGAN %s"%modelsnm, fig=fig0,
                        savename="SG_%s_shfl_spectrum_cmp_All" % modelsnm, figdir=summaryroot)
fig0 = plot_spectra(eva_col_W_ctrl, label="Fix Wspace Shuffled", titstr="StyleGAN %s"%modelsnm, fig=fig0,
                        savename="SG_%s_shfl_spectrum_cmp_All" % modelsnm, figdir=summaryroot)

fig0 = plot_spectra(eva_col_W, label="Fix Wspace", titstr="StyleGAN %s"%modelsnm, fig=None,
                        savename="SG_%s_shfl_spectrum_cmp_Wspace" % modelsnm, figdir=summaryroot)
fig0 = plot_spectra(eva_col_W_ctrl, label="Fix Wspace Shuffled", titstr="StyleGAN %s"%modelsnm, fig=fig0,
                        savename="SG_%s_shfl_spectrum_cmp_Wspace" % modelsnm, figdir=summaryroot)

label = modelnm+"_fix"
data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
corr_mat_log, corr_mat_lin = data['corr_mat_log'], data['corr_mat_lin']
label = modelnm+"_fix_ctrl"
data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
corr_mat_log_ctrl, corr_mat_lin_ctrl = data['corr_mat_log'], data['corr_mat_lin']
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=summaryroot, label="fix",
                                    savelabel="SG_%s_shuffle_cmp_Zspace"%modelsnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=summaryroot, label="fix_shuffle",
                                     savelabel="SG_%s_shuffle_cmp_Zspace"%modelsnm, figs=(fig11,fig22))

label = modelnm+"_W_fix"
data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
corr_mat_log, corr_mat_lin = data['corr_mat_log'], data['corr_mat_lin']
label = modelnm+"_W_fix_ctrl"
data = np.load(join(summaryroot, label, "Hess_%s_corr_mat.npz"%label))
corr_mat_log_ctrl, corr_mat_lin_ctrl = data['corr_mat_log'], data['corr_mat_lin']
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=summaryroot, label="fix_W",
                                    savelabel="SG_%s_shuffle_cmp_Wspace"%modelsnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=summaryroot, label="fix_W_shuffle",
                                     savelabel="SG_%s_shuffle_cmp_Wspace"%modelsnm, figs=(fig11,fig22))