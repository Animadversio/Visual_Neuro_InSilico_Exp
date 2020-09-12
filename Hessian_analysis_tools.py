#%%
"""
This lib curates functions that are useful for Hessian analysis for different GANs
"""
import sys
import re
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from time import time
import os
from os.path import join
def plot_spectra(eigval_col, savename="spectrum_all", figdir="", abs=True,
                 titstr="GAN", label="all", fig=None):
    """A local function to compute these figures for different subspaces. """
    if abs:
        eigval_col = np.abs(eigval_col)
    eigmean = eigval_col.mean(axis=0)
    eiglim = np.percentile(eigval_col, [5, 95], axis=0)
    sortIdx = np.argsort(-np.abs(eigmean))
    eigmean = eigmean[sortIdx]
    eiglim = eiglim[:, sortIdx]
    eigN = len(eigmean)
    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    else:
        plt.figure(fig.number)
        axs = fig.axes
    plt.sca(axs[0])
    plt.plot(range(eigN), eigmean, alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), eiglim[0, :], eiglim[1, :], alpha=0.3, label="all space")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.legend()
    plt.sca(axs[1])
    plt.plot(range(eigN), np.log10(eigmean), alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label=label)
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of %s\n (error bar for [5,95] percentile among all samples)"%(titstr))
    plt.savefig(join(figdir, savename+".png"), bbox_extra_artists=[st]) # this is working.
    plt.savefig(join(figdir, savename+".pdf"), bbox_extra_artists=[st])  # this is working.
    plt.show()
    return fig
#%%
import numpy.ma as ma
import torch
def corr_torch(V1, V2):
    C1 = (V1 - V1.mean())
    C2 = (V2 - V2.mean())
    return torch.dot(C1, C2) / C1.norm() / C2.norm()

def corr_nan_torch(V1, V2):
    Msk = torch.isnan(V1) | torch.isnan(V2)
    return corr_torch(V1[~Msk], V2[~Msk])

def compute_hess_corr(eigval_col, eigvec_col, savelabel="", figdir="", use_cuda=False):
    posN = len(eigval_col)
    T0 = time()
    if use_cuda:
        corr_mat_log = torch.zeros((posN, posN)).cuda()
        corr_mat_lin = torch.zeros((posN, posN)).cuda()
        for eigi in tqdm(range(posN)):
            evc_i, eva_i = torch.from_numpy(eigvec_col[eigi]).cuda(), torch.from_numpy(eigval_col[eigi]).cuda()
            for eigj in range(posN):
                evc_j, eva_j = torch.from_numpy(eigvec_col[eigj]).cuda(), torch.from_numpy(eigval_col[eigj]).cuda()
                inpr = evc_i.T @ evc_j
                vHv_ij = torch.diag((inpr * eva_j.unsqueeze(0)) @ inpr.T)
                corr_mat_log[eigi, eigj] = corr_nan_torch(vHv_ij.log10(), eva_j.log10())
                corr_mat_lin[eigi, eigj] = corr_nan_torch(vHv_ij, eva_j)
        corr_mat_log = corr_mat_log.cpu().numpy()
        corr_mat_lin = corr_mat_lin.cpu().numpy()
    else:
        corr_mat_log = np.zeros((posN, posN))
        corr_mat_lin = np.zeros((posN, posN))
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi]
            for eigj in range(posN):
                eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj]
                inpr = evc_i.T @ evc_j
                vHv_ij = np.diag((inpr * eva_j[np.newaxis, :]) @ inpr.T)
                corr_mat_log[eigi, eigj] = ma.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(eva_j)))[0, 1]
                corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eva_j)[0, 1]

    print("%.1f sec" % (time() - T0)) # 582.2 secs for the 1000 by 1000 mat. not bad!
    np.savez(join(figdir, "Hess_%s_corr_mat.npz" % savelabel), corr_mat_log=corr_mat_log, corr_mat_lin=corr_mat_lin)

    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)  # corr_mat_log_nodiag =
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)  # corr_mat_log_nodiag =
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    print("Log scale non-diag mean corr value %.3f med %.3f" % (log_nodiag_mean_cc, log_nodiag_med_cc))
    print("Lin scale non-diag mean corr value %.3f med %.3f" % (lin_nodiag_mean_cc, lin_nodiag_med_cc))
    return corr_mat_log, corr_mat_lin
#%
def plot_consistentcy_mat(corr_mat_log, corr_mat_lin, savelabel="", posN=100, figdir="", titstr="GAN"):
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    print("Log scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_log_nodiag))
    print("Lin scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_lin_nodiag))
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("%s Hessian at %d vectors\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.pdf"%savelabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("%s Hessian at %d vectors\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.pdf"%savelabel))
    plt.show()
    return fig1, fig2