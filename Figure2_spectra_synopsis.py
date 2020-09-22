import os
import re
from time import time
from os.path import join
from glob import glob
import sys
import pandas as pd
import numpy as np
from hessian_analysis_tools import scan_hess_npz, plot_spectra
import matplotlib.pylab as plt
import matplotlib
summarydir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary"
#%%
FC6figdir = r"E:\OneDrive - Washington University in St. " \
            r"Louis\Hessian_summary\fc6GAN" #r"E:\Cluster_Backup\FC6GAN\summary"
FC6dir = r"E:\Cluster_Backup\FC6GAN"
eva_col, _, feat_arr, meta = scan_hess_npz(FC6dir, "evol_(\d*)_bpfull.npz", featkey='code', evakey='eigvals',
                                                     evckey=None)#'eigvects')

#%%

eigvals_col = np.array(eigvals_col)[:, ::-1]
code_all = np.array(code_all)
np.savez(join(FC6figdir, "spectra_col_evol.npz"), eigval_col=eigvals_col, )
#%%
"""Visualize the spectra of different GANs all in one place"""
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
spaceD = [4096, 120, 256, 120, 512, 512, 512, 512, 512]
GANlist = ["FC6", "DCGAN-fashion", "BigGAN", "BigBiGAN", "PGGAN-face", "StyleGAN-face", "StyleGAN2-face512",
           "StyleGAN2-face256", "StyleGAN2-cat", ]
           # "StyleGAN-face-Forw", "StyleGAN-cat-Forw"]
fnlist = ["FC6GAN\\spectra_col_evol.npz",
          "DCGAN\\spectra_col_BP.npz",
          "BigGAN\\spectra_col.npz",
          "BigBiGAN\\spectra_col.npz",
          "PGGAN\\spectra_col_BP.npz",
          "StyleGAN\\spectra_col_face256_BP.npz",
          "StyleGAN2\\spectra_col_FFHQ512.npz",
          "StyleGAN2\\spectra_col_ffhq-256-config-e-003810_BP.npz",
          "StyleGAN2\\spectra_col_stylegan2-cat-config-f_BP.npz", ]
          # "StyleGAN2\\spectra_col_stylegan2-cat-config-f.npz",
          # "StyleGAN2\\spectra_col_ffhq-512-avg-tpurun1_Forwa.npz",
          # "StyleGAN2\\spectra_col_stylegan2-cat-config-f_Forwa.npz"]
#%%
cutoffR = 1E-9
#%%
def spectra_montage(GANlist, fnlist, ylog=True, xnorm=False, ynorm=True, shade=True, xlim=(-25, 525), \
                                             lw=1, fn="spectra_synopsis_log_rank"):
    fig = plt.figure()
    for i, GAN in enumerate(GANlist):
        with np.load(join(rootdir, fnlist[i])) as data:
            eigval_col = data["eigval_col"]
        if eigval_col[:, -1].mean() > eigval_col[:, 0].mean():
            eigval_col = eigval_col[:, ::-1]
        eva_mean = eigval_col.mean(axis=0)
        eva_lim = np.percentile(eigval_col, [5, 95], axis=0)
        # eva_lim_pos = np.maximum(eva_lim, cutoffR * eva_mean.max())
        eva_lim_pos = eva_lim.copy()
        if ylog:
            negmask = eva_lim_pos[0, :] < 0
            eva_lim_pos[0, negmask] = eva_mean[negmask]
        xnormalizer = len(eva_mean) if xnorm else 1
        ynormalizer = eva_mean.max() if ynorm else 1
        ytfm = np.log10 if ylog else lambda x: x
        plt.plot(np.arange(len(eva_mean))/xnormalizer, ytfm(eva_mean / ynormalizer), alpha=1, lw=lw)  # ,
        # eigval_arr.std(axis=0)
        if shade:
            plt.fill_between(np.arange(len(eva_mean))/xnormalizer, ytfm(eva_lim_pos[0, :] / ynormalizer),
                         ytfm(eva_lim_pos[1, :] / ynormalizer), alpha=0.35, label=GAN)
    plt.ylabel("log10(eig/eigmax)" if ylog else "eig/eigmax")
    plt.xlabel("rank normalized to latent dim" if xnorm else "ranks")
    plt.xlim(xlim)
    plt.title("Spectra Compared Across GANs")
    plt.legend(loc="best")
    plt.savefig(join(rootdir, fn+".png"))
    plt.savefig(join(rootdir, fn+".pdf"))
    plt.show()
    return fig

fig1 = spectra_montage(GANlist, fnlist, xlim=(-25, 525), fn="spectra_synopsis_log_rank2")
fig2 = spectra_montage(GANlist, fnlist, xlim=(-25, 4125), fn="spectra_synopsis_log_rank_full2")
fig3 = spectra_montage(GANlist, fnlist, xlim=(-25, 515), shade=False, fn="spectra_synopsis_log_rank_line")
#%%
GANlist_Conv = [GANlist[i] for i in [1,2,3,4]]
fnlist_Conv = [fnlist[i] for i in [1,2,3,4]]
GANlist_Style = [GANlist[i] for i in [5,6,7,8]]
fnlist_Style = [fnlist[i] for i in [5,6,7,8]]
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 140), lw=2, fn="spectra_synopsis_log_rank_Conv")
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 520), lw=2, fn="spectra_synopsis_log_rank_Conv_Full")
fig5 = spectra_montage(GANlist_Style, fnlist_Style, xlim=(-5, 140), lw=2, fn="spectra_synopsis_log_rank_Style")
