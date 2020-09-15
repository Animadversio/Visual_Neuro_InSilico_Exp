
"""
Spectral Mixing for Random directions make the figures to demonstrate this
"""

import os
import re
from time import time
from os.path import join
from glob import glob
import sys
import pandas as pd
import numpy as np
from Hessian_analysis_tools import scan_hess_npz
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
summarydir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary"
#%%
def plot_spectra_mix_cmp(eigval_m, repN=3000, figdir="", Hlabel=""):
    mixeig_col = []
    for i in range(repN):
        v = np.random.randn(len(eigval_m))
        v /= np.sqrt((v ** 2).sum())
        mixeig = (v ** 2 * eigval_m).sum() #/ sum(v ** 2)
        mixeig_col.append(mixeig)
    fig = plt.figure()
    plt.hist(np.log10(eigval_m), bins=40, alpha=0.5, density=True, label="eigen value")
    plt.hist(np.log10(mixeig_col), bins=40, alpha=0.5, density=True, label="vHv of random unit vector")
    plt.ylabel("Frequency Density")
    plt.xlabel("log10(lambda)")
    plt.title("Comparison of Original Spectrum and Random Mixing of\n%s"%Hlabel)
    plt.savefig(join(figdir, "Spectra_rand_mix_cmp_%s.png"%Hlabel))
    plt.legend()
    plt.show()
    return mixeig_col, fig

#%%
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

eigval_col = np.load(join(rootdir, fnlist[2]))["eigval_col"]
eigval_m = eigval_col.mean(0)
plot_spectra_mix_cmp(eigval_m, Hlabel="BigGAN", figdir=join(summarydir, "BigGAN"))

eigval_col = np.load(join(rootdir, fnlist[3]))["eigval_col"]
eigval_m = eigval_col.mean(0)
plot_spectra_mix_cmp(eigval_m, Hlabel="BigBiGAN", figdir=join(summarydir, "BigBiGAN"))
#%%
for path, label in zip(fnlist, GANlist):
    GANname = path.split("\\")[0]
    eigval_col = np.load(join(rootdir, path))["eigval_col"]
    eigval_m = eigval_col.mean(0)
    figdir = join(rootdir, GANname)
    plot_spectra_mix_cmp(eigval_m, Hlabel=label, figdir=figdir)

#%% The first and second moment of the alpha distribution by derivation.
def theoretical_moment(eigval_m):
    """Compute the theoretical first and 2nd moment for the eigenvalue distribution"""
    dim = len(eigval_m)
    Var = ((eigval_m**2).sum() * dim - eigval_m.sum()**2) * 2 / dim**2 / (dim+2)
    Mean = eigval_m.sum()/dim
    return Mean, Var

