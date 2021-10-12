
import os
import re
from time import time
from os.path import join
from glob import glob
import sys
import pandas as pd
import numpy as np
from Hessian.hessian_analysis_tools import scan_hess_npz
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
summarydir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary"

def theoretical_moment(eigval_m):
    """Compute the theoretical first and 2nd moment for the eigenvalue distribution"""
    dim = len(eigval_m)
    Var = ((eigval_m**2).sum() * dim - eigval_m.sum()**2) * 2 / dim**2 / (dim+2)
    Mean = eigval_m.sum()/dim
    return Mean, Var

def plot_spectra_mix_cmp(eigval_m, repN=3000, figdir="", Hlabel=""):
    mixeig_col = []
    for i in range(repN):
        v = np.random.randn(len(eigval_m))
        v /= np.sqrt((v ** 2).sum())
        mixeig = (v ** 2 * eigval_m).sum() #/ sum(v ** 2)
        mixeig_col.append(mixeig)
    Mean, Var = theoretical_moment(eigval_m)
    Mean_emp, Var_emp = np.mean(mixeig_col), np.var(mixeig_col)
    fig = plt.figure(figsize=[4,3.5])
    plt.hist(np.log10(eigval_m), bins=40, alpha=0.5, density=True, label="eigen value")
    plt.hist(np.log10(mixeig_col), bins=40, alpha=0.5, density=True, label="vHv of random unit vector")
    plt.ylabel("Frequency Density")
    plt.xlabel("log10(lambda)")
    plt.title("Comparison of Spectrum and Random Mixing \n%s\nTheory %.1e(%.1e) Empir %.1e(%.1e)"%(Hlabel, Mean, np.sqrt(Var), Mean_emp, np.sqrt(Var_emp)))
    plt.legend()
    plt.subplots_adjust(top=0.8, bottom=0.2)
    plt.savefig(join(figdir, "Spectra_rand_mix_cmp_%s.png"%Hlabel))
    plt.savefig(join(figdir, "Spectra_rand_mix_cmp_%s.pdf"%Hlabel))
    plt.show()
    return mixeig_col, fig

spectra_npz_dict = {"fc6GAN": "FC6GAN\\spectra_col_evol.npz",
          "DCGAN": "DCGAN\\spectra_col_BP.npz",
          "BigGAN": "BigGAN\\spectra_col.npz",
          "BigGAN_noise": "BigGAN\\spectra_col.npz",
          "BigGAN_class": "BigGAN\\spectra_col.npz",
          "BigBiGAN": "BigBiGAN\\spectra_col.npz",
          "PGGAN": "PGGAN\\spectra_col_BP.npz",
          "StyleGAN-Face_var": "StyleGAN\\spectra_col_face256_BP.npz",
          "StyleGAN2-Face512_var": "StyleGAN2\\spectra_col_FFHQ512.npz",
          "StyleGAN2-Face256_var": "StyleGAN2\\spectra_col_ffhq-256-config-e-003810_BP.npz",
          "StyleGAN2-Cat256_var": "StyleGAN2\\spectra_col_stylegan2-cat-config-f_BP.npz",
          "StyleGAN-Face_Z": "StyleGAN_Fix\\StyleGAN_Face256_fix\\spectra_col_StyleGAN_Face256_fix.npz",
          "StyleGAN2-Face512_Z": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix\\spectra_col_ffhq-512-avg-tpurun1_fix.npz",
          "StyleGAN2-Face256_Z": "StyleGAN2_Fix\\ffhq-256-config-e-003810_fix\\spectra_col_ffhq-256-config-e-003810_fix.npz",
          "StyleGAN2-Cat256_Z": "StyleGAN2_Fix\\stylegan2-cat-config-f_fix\\spectra_col_stylegan2-cat-config-f_fix.npz",
          "StyleGAN-Face_W": "StyleGAN_Fix\\StyleGAN_Face256_W_fix\\spectra_col_StyleGAN_Face256_W_fix.npz",
          "StyleGAN2-Face512_W": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix\\spectra_col_ffhq-512-avg-tpurun1_W_fix.npz",
          "StyleGAN2-Face256_W": "StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix\\spectra_col_ffhq-256-config-e-003810_W_fix.npz",
          "StyleGAN2-Cat256_W": "StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix\\spectra_col_stylegan2-cat-config-f_W_fix.npz",
          }

summarydir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
figdir = join(summarydir, "Figure2_Supp")
for label, path,  in spectra_npz_dict.items():#zip(fnlist, GANlist):
    GANname = path.split("\\")[0]
    eigval_col = np.load(join(summarydir, path))["eigval_col"]
    eigval_m = eigval_col.mean(0)
    # figdir = join(rootdir, GANname)
    plot_spectra_mix_cmp(eigval_m, Hlabel=label, figdir=figdir)