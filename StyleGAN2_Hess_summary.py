"""
This script summarize and the Hessian computation for StyleGAN2
Analyze the geometry of the BigGAN manifold. How the metric tensor relates to the coordinate.
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
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
SGdir = r"E:\Cluster_Backup\StyleGAN2"
Hdir = "E:\Cluster_Backup\StyleGAN2\stylegan2-cat-config-f"
figdir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
#%%
npzpaths = glob(join(Hdir, "*.npz"))
npzfns = [path.split("\\")[-1] for path in npzpaths]
eigval_col = []
for fn, path in zip(npzfns, npzpaths):
    data = np.load(path)
    evas = data["eigvals"]
    eigval_col.append(evas)
eigval_col = np.array(eigval_col)
plt.plot(eigval_col.mean(axis=0))
plt.show()
#%%
def scan_hess_npz(Hdir):
    npzpaths = glob(join(Hdir, "*.npz"))
    npzfns = [path.split("\\")[-1] for path in npzpaths]
    npzpattern = re.compile("Hess_trunc([\d\.]*)_(\d*).npz")
    eigval_col = []
    eigvec_col = []
    meta = []
    for fn, path in zip(npzfns, npzpaths):
        data = np.load(path)
        trunc, RND = npzpattern.findall(fn)[0]
        evas = data["eigvals"]
        evcs = data["eigvects"]
        eigval_col.append(evas)
        eigvec_col.append(evcs)
        meta.append((float(trunc), int(RND)))
    eigval_col = np.array(eigval_col)
    return eigval_col, eigvec_col, meta
#%%
#%%
def plot_spectra(eigval_col, savename="spectrum_stat_all3.jpg", ):
    """A local function to compute these figures for different subspaces. """
    eigmean = eigval_col.mean(axis=0)
    eiglim = np.percentile(eigval_col, [5, 95], axis=0)
    eigN = len(eigmean)
    fig = plt.figure(figsize=[10, 5])
    plt.subplot(1,2,1)
    plt.plot(range(eigN), eigmean, alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), eiglim[0, :], eiglim[1, :], alpha=0.3, label="all space")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(eigN), np.log10(eigmean), alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label="all space")
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of StyleGAN2\n (error bar for [5,95] percentile among all samples)")
    plt.savefig(join(figdir, savename), bbox_extra_artists=[st]) # this is working.
    plt.show()
    return fig

plot_spectra(eigval_col, savename="Cat-f_spectra.jpg", )
#%% Go through all the models see how their thing is doing
subpath = [f.path for f in os.scandir(SGdir) if f.is_dir()]
subfdnm = [f.name for f in os.scandir(SGdir) if f.is_dir()]
failnms = []
for fdnm in subfdnm:
    try:
        eigval_col, _, meta = scan_hess_npz(join(SGdir, fdnm))
        plot_spectra(eigval_col, savename="%s-spectra.jpg"%fdnm)
        print(fdnm, "folder finished")
    except:
        print(fdnm, "folder failed, please check")
        failnms.append(fdnm)
#%%
import numpy.ma as ma
def compute_hess_corr(eigval_col, eigvec_col, fdnm=""):
    posN = len(eigval_col)
    T0 = time()
    corr_mat_log = np.zeros((posN, posN))
    corr_mat_lin = np.zeros((posN, posN))
    for eigi in tqdm(range(posN)):
        for eigj in range(posN):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi] # torch.from_numpy(eigvect_j).cuda()
            eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj] # torch.from_numpy(eigval_j).cuda()
            inpr = evc_i.T @ evc_j
            vHv_ij = np.diag((inpr * eva_j[np.newaxis, :]) @ inpr.T)
            corr_mat_log[eigi, eigj] = ma.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(
                eva_j)))[0, 1]
            corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eva_j)[0, 1]
            # corr_mat_log[eigi, eigj] = corr_nan_torch(vHv_ij.log10(), eva_j.log10())
            # corr_mat_lin[eigi, eigj] = corr_nan_torch(vHv_ij, eva_j)
            # vHv_ij = np.diag(eigvec_col[eigi].T @ H_col[eigj] @ eigvec_col[eigi])

    print("%.1f sec" % (time() - T0)) # 582.2 secs for the 1000 by 1000 mat. not bad!
    np.savez(join(figdir, "Hess_%s_corr_mat.npz" % fdnm), corr_mat_log=corr_mat_log, corr_mat_lin=corr_mat_lin)
    return corr_mat_log, corr_mat_lin
#%%
def plot_consistentcy_mat(corr_mat_log, corr_mat_lin, Hlabel="", posN=100):
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan) # corr_mat_log_nodiag =
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan) # corr_mat_log_nodiag =
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    print("Log scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_log_nodiag))  # 0.934
    print("Lin scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_lin_nodiag))  # 0.934
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("StyleGAN2 %s Hessian at %d vectors\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(Hlabel, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.jpg"%Hlabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.pdf"%Hlabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("StyleGAN2 %s Hessian at %d vectors\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(Hlabel, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.jpg"%Hlabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.pdf"%Hlabel))
    plt.show()
    return fig1, fig2
#%%
subpath = [f.path for f in os.scandir(SGdir) if f.is_dir()]
subfdnm = [f.name for f in os.scandir(SGdir) if f.is_dir()]
failnms2 = []
for fdnm in subfdnm:#["FFHQ512"]:
    try:
        eigval_col, eigvec_col, meta = scan_hess_npz(join(SGdir, fdnm))
        posN = len(eigval_col)
        corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, fdnm=fdnm)
        plot_consistentcy_mat(corr_mat_log, corr_mat_lin, Hlabel=fdnm, posN=posN)
        print(fdnm, "folder finished")
    except:
        print(fdnm, "folder failed, please check")
        failnms2.append(fdnm)
#%%

