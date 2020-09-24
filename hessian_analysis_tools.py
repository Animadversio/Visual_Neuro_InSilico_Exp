#%%
"""
This lib curates functions that are useful for Hessian analysis for different GANs
- load computed hess npz
- Visualize spectra

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
def scan_hess_npz(Hdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP', evckey='evc_BP', featkey=None):
    """ Function to load in npz and collect the spectra.
    Set evckey=None to avoid loading eigenvectors.

    Note for newer experiments use evakey='eva_BP', evckey='evc_BP'
    For older experiments use evakey='eigvals', evckey='eigvects'"""
    npzpaths = glob(join(Hdir, "*.npz"))
    npzfns = [path.split("\\")[-1] for path in npzpaths]
    npzpattern = re.compile(npzpat)
    eigval_col = []
    eigvec_col = []
    feat_col = []
    meta = []
    for fn, path in tqdm(zip(npzfns, npzpaths)):
        match = npzpattern.findall(fn)
        if len(match) == 0:
            continue
        parts = match[0]  # trunc, RND
        data = np.load(path)
        try:
            evas = data[evakey]
            eigval_col.append(evas)
            if evckey is not None:
                evcs = data[evckey]
                eigvec_col.append(evcs)
            if featkey is not None:
                feat = data[featkey]
                feat_col.append(feat)
            meta.append(parts)
        except KeyError:
            print("KeyError, keys in the archive : ", list(data))
            return
    eigval_col = np.array(eigval_col)
    print("Load %d npz files of Hessian info" % len(meta))
    if featkey is None:
        return eigval_col, eigvec_col, meta
    else:
        feat_col = np.array(tuple(feat_col)).squeeze()
        return eigval_col, eigvec_col, feat_col, meta
#%%
def average_H(eigval_col, eigvec_col):
    """Compute the average Hessian over a bunch of positions"""
    nH = len(eigvec_col)
    dimen = eigval_col.shape[1]
    H_avg = np.zeros((dimen, dimen))
    for iH in range(nH):
        H = (eigvec_col[iH] * eigval_col[iH][np.newaxis, :]) @ eigvec_col[iH].T
        H_avg += H
    H_avg /= nH
    eva_avg, evc_avg = np.linalg.eigh(H_avg)
    return H_avg, eva_avg, evc_avg
#%%
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
        plt.figure(num=fig.number)
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
    # plt.show()
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
    """cuda should be used for large mat mul like 512 1024 4096.
    small matmul should stay with cpu numpy computation. cuda will add the IO overhead."""
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
    print("Compute results saved to %s" % join(figdir, "Hess_%s_corr_mat.npz" % savelabel))
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

def compute_vector_hess_corr(eigval_col, eigvec_col, savelabel="", figdir="", use_cuda=False):
    posN = len(eigval_col)
    T0 = time()
    if use_cuda:
        corr_mat_vec = torch.zeros((posN, posN)).cuda()
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = torch.from_numpy(eigval_col[eigi]).cuda(), torch.from_numpy(eigvec_col[eigi]).cuda()
            H_i = (evc_i * eva_i.unsqueeze(0)) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = torch.from_numpy(eigval_col[eigj]).cuda(), torch.from_numpy(eigvec_col[eigj]).cuda()
                H_j = (evc_j * eva_j.unsqueeze(0)) @ evc_j.T
                corr_mat_vec[eigi, eigj] = corr_torch(H_i.flatten(), H_j.flatten())
        corr_mat_vec = corr_mat_vec.cpu().numpy()
    else:
        corr_mat_vec = np.zeros((posN, posN))
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi]
            H_i = (evc_i * eva_i[np.newaxis, :]) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj]
                H_j = (evc_j * eva_j[np.newaxis, :]) @ evc_j.T
                # corr_mat_log[eigi, eigj] = \
                # np.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(eva_j)))[0, 1]
                corr_mat_vec[eigi, eigj] = np.corrcoef(H_i.flatten(), H_j.flatten())[0, 1]
    print("%.1f sec" % (time() - T0))  #
    np.savez(join(figdir, "Hess_%s_corr_mat_vec.npz" % savelabel), corr_mat_vec=corr_mat_vec, )
    print("Compute results saved to %s" % join(figdir, "Hess_%s_corr_mat_vec.npz" % savelabel))
    return corr_mat_vec
#%
def plot_consistentcy_mat(corr_mat_log, corr_mat_lin, savelabel="", figdir="", titstr="GAN"):
    posN = corr_mat_log.shape[0]
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
#%%
def histogram_corrmat(corr_mat_lin, log=True, GAN="GAN", fig=None):
    if fig is None:
        fig = plt.figure(figsize=[4, 3])
    else:
        plt.figure(num=fig.number)
    plt.hist(corr_mat_lin.flatten()[~np.isnan(corr_mat_lin.flatten())], 60, density=True, alpha=0.7)
    corr_mean = np.nanmean(corr_mat_lin)
    corr_medi = np.nanmedian(corr_mat_lin)
    _, YMAX = plt.ylim()
    plt.vlines(corr_mean, 0, YMAX, linestyles="dashed", color="black")
    plt.vlines(corr_medi, 0, YMAX, linestyles="dashed", color="red")
    plt.xlabel("corr(log(V_iH_jV_i), log(Lambda_j))" if log else "corr(V_iH_jV_i, Lambda_j)")
    plt.ylabel("density")
    plt.title("Histogram of Non-Diag Correlation\n %s on %s scale\n mean %.3f median %.3f" %
              (GAN, "log" if log else "lin", corr_mean, corr_medi))
    # plt.show()
    return fig

def plot_consistency_hist(corr_mat_log, corr_mat_lin, savelabel="", figdir="", titstr="GAN", figs=(None, None)):
    """Histogram way to represent correlation instead of corr matrix, same interface as plot_consistentcy_mat"""
    posN = corr_mat_log.shape[0]
    np.fill_diagonal(corr_mat_lin, np.nan)
    np.fill_diagonal(corr_mat_log, np.nan)
    if figs is not None: fig1, fig2 = figs
    fig1 = histogram_corrmat(corr_mat_log, log=True, GAN=titstr, fig=fig1)
    fig1.savefig(join(figdir, "Hess_%s_corr_mat_log_hist.jpg"%savelabel))
    fig1.savefig(join(figdir, "Hess_%s_corr_mat_log_hist.pdf"%savelabel))
    # fig1.show()
    fig2 = histogram_corrmat(corr_mat_lin, log=False, GAN=titstr, fig=fig2)
    fig2.savefig(join(figdir, "Hess_%s_corr_mat_lin_hist.jpg"%savelabel))
    fig2.savefig(join(figdir, "Hess_%s_corr_mat_lin_hist.pdf"%savelabel))
    # fig2.show()
    return fig1, fig2
#%% Derive from BigBiGAN
def plot_consistency_example(eigval_col, eigvec_col, nsamp=5, titstr="GAN", figdir="", savelabel=""):
    """
    Note for scatter plot the aspect ratio is set fixed to one.
    :param eigval_col:
    :param eigvec_col:
    :param nsamp:
    :param titstr:
    :param figdir:
    :return:
    """
    Hnums = len(eigval_col)
    eiglist = sorted(np.random.choice(Hnums, nsamp, replace=False))  # range(5)
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, eigi in enumerate(eiglist):
        eigval_i, eigvect_i = eigval_col[eigi], eigvec_col[eigi]
        for axj, eigj in enumerate(eiglist):
            eigval_j, eigvect_j = eigval_col[eigj], eigvec_col[eigj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_j), 20)
            else:
                ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp-1:
                ax.set_xlabel("eigvals @ pos %d" % eigj)
            if axj == 0:
                ax.set_ylabel("vHv eigvec @ pos %d" % eigi)
    ST = plt.suptitle("Consistency of %s Hessian Across Vectors\n"
                      "Cross scatter of EigenValues and vHv values for Hessian at %d Random Vectors"%(titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Hess_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Hess_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig
#%%
def plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="", figdir="", titstr="GAN", layernames=None):
    """How Hessian matrix in different layers correspond to each other. """
    posN = corr_mat_log.shape[0]
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    corr_mat_vec_nodiag = corr_mat_vec.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
    np.fill_diagonal(corr_mat_vec_nodiag, np.nan)
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    vec_nodiag_mean_cc = np.nanmean(corr_mat_vec_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    vec_nodiag_med_cc = np.nanmedian(corr_mat_vec_nodiag)
    print("Log scale corr non-diag mean value %.3f"%log_nodiag_mean_cc)
    print("Lin scale corr non-diag mean value %.3f"%lin_nodiag_mean_cc)
    print("Vec Hessian corr non-diag mean value %.3f"%vec_nodiag_mean_cc)
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    fig1.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN-0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.pdf"%savelabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    fig2.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.pdf"%savelabel))
    plt.show()

    fig3 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_vec, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vectorized Hessian Mat"
              "\nNon-Diagonal mean %.3f median %.3f" % (titstr, posN, vec_nodiag_mean_cc, vec_nodiag_med_cc),
              fontsize=15)
    plt.colorbar()
    fig3.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.jpg" % savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.pdf" % savelabel))
    plt.show()
    return fig1, fig2, fig3

#%%
def plot_layer_consistency_example(eigval_col, eigvec_col, layernames, layeridx=[0,1,-1], titstr="GAN", figdir="",
                                   savelabel=""):
    """
    Note for scatter plot the aspect ratio is set fixed to one.
    :param eigval_col:
    :param eigvec_col:
    :param nsamp:
    :param titstr:
    :param figdir:
    :return:
    """
    nsamp = len(layeridx)
    # Hnums = len(eigval_col)
    # eiglist = sorted(np.random.choice(Hnums, nsamp, replace=False))  # range(5)
    print("Plot hessian of layers : ", [layernames[idx] for idx in layeridx])
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, Li in enumerate(layeridx):
        eigval_i, eigvect_i = eigval_col[Li], eigvec_col[Li]
        for axj, Lj in enumerate(layeridx):
            eigval_j, eigvect_j = eigval_col[Lj], eigvec_col[Lj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_j), 20)
            else:
                ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp-1:
                ax.set_xlabel("eigvals @ %s" % layernames[Lj])
            if axj == 0:
                ax.set_ylabel("vHv eigvec @ %s" % layernames[Li])
    ST = plt.suptitle("Consistency of %s Hessian Across Layers\n"
                      "Cross scatter of EigenValues and vHv values for Hessian at %d Layers"%(titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Hess_layer_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Hess_layer_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig