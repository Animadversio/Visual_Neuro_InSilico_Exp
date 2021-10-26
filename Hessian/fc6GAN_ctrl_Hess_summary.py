"""
This script summarize and the Hessian computation for FC6 GAN.
at the inverted codes of Pasupathy patches or Evolved codes from monkey experiments.
Analyze the geometry of the FC6 manifold. How the metric tensor relates to the coordinate.
"""
#%%
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
import torch
import os
#%%
savedir = r"E:\Cluster_Backup\fc6_shflGAN"
figdir = r"E:\Cluster_Backup\fc6_shflGAN\summary"
os.makedirs(figdir,exist_ok=True)
labeldict = {"BP": "bpfull", "BackwardIter": "bkwlancz", "ForwardIter": "frwlancz"}
method = "BackwardIter"
space = "evol"
labstr = labeldict[method]
idx = 0
data = np.load(join(savedir, "%s_%03d_%s.npz" % (space, idx, labstr)))
#%%
savedir = r"E:\Cluster_Backup\fc6_shflGAN"
space = "evol"
method = "BackwardIter"
#%%
savedir = r"E:\Cluster_Backup\fc6_shfl_fixGAN"
figdir = r"E:\Cluster_Backup\fc6_shfl_fixGAN\summary"
os.makedirs(figdir,exist_ok=True)
labeldict = {"BP": "bpfull", "BackwardIter": "bkwlancz", "ForwardIter": "frwlancz"}
space = "evol"
method = "BP"
labstr = labeldict[method]
#%%
H_col = []
eigvals_col = []
eigvects_col = []
code_all = []
for idx in range(284): # Note load it altogether is very slow, not recommended
    fn = "%s_%03d_%s.npz" % (space, idx, labstr)
    data = np.load(join(savedir, fn))
    eigvals = data['eigvals']
    eigvals_col.append(eigvals.copy())
    # eigvects = data['eigvects']
#%
eigval_arr_ctrl = np.array(eigvals_col)
eigmean_ctrl = eigval_arr_ctrl[:, ::-1].mean(axis=0)
eigstd_ctrl = eigval_arr_ctrl[:, ::-1].std(axis=0)
eiglim_ctrl = np.percentile(eigval_arr_ctrl[:, ::-1], [5, 95], axis=0)

#%
def plot_spectra(savename="spectrum_stat_4096.jpg", ):
    """A local function to compute these figures for different subspaces. """
    fig = plt.figure(figsize=[10, 5])
    cutoff = len(eigmean_ctrl)
    plt.subplot(1,2,1)
    plt.plot(range(cutoff), eigmean_ctrl, alpha=0.7)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(cutoff), eiglim_ctrl[0, :], eiglim_ctrl[1, :], alpha=0.5, color="orange", label="5-95 percentile")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.xlim([-50, 4100])
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(cutoff), np.log10(eigmean_ctrl), alpha=0.7)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(cutoff), np.log10(eiglim_ctrl[0, :]), np.log10(eiglim_ctrl[1, :]), alpha=0.5, color="orange", label="5-95 percentile")
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.xlim([-50, 4100])
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of Weight shuffled FC6GAN \n (error bar for [5,"
                      "95] percentile among 284 positions)")
    plt.savefig(join(figdir, savename), bbox_extra_artists=[st]) # this is working.
    plt.show()
plot_spectra(savename="spectrum_stat_org_shuffleG.jpg", )
#%%
# plt.figure()
# plt.plot(range(4096), eigmean, alpha=0.7)  #, eigval_arr.std(axis=0)
# plt.fill_between(range(4096), eiglim[0, :], eiglim[1, :], alpha=0.5, color="orange", label="all space")
# plt.ylabel("eigenvalue")
# plt.xlabel("eig id")
# plt.legend()
# plt.show()
#%%
def load_eig(space, idx, labstr):
    fn = "%s_%03d_%s.npz" % (space, idx, labstr)
    data = np.load(join(savedir, fn))
    eigvals = data['eigvals']
    eigvects = data['eigvects']
    return eigvals, eigvects
#%% Consistency of the Hessians
fig = plt.figure(figsize=[10, 10], constrained_layout=False)
spec = fig.add_gridspec(ncols=5, nrows=5, left=0.05, right=0.95, top=0.9, bottom=0.05)
for eigi in range(5):
    eigval_i, eigvect_i = load_eig(space, eigi, labstr)
    for eigj in range(5):
        eigval_j, eigvect_j = load_eig(space, eigj, labstr)

        vHv_ij = np.diag(eigvect_i.T @ eigvect_j @ np.diag(eigval_j) @ eigvect_j.T @ eigvect_i)
        ax = fig.add_subplot(spec[eigi, eigj])
        if eigi == eigj:
            ax.hist(np.log10(eigval_j), 20)
        else:
            ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
            ax.set_aspect(1, adjustable='datalim')
        if eigi == 4:
            ax.set_xlabel("eigvals %d" % eigj)
        if eigj == 0:
            ax.set_ylabel("vHv eigvects %d" % eigj)
ST = plt.suptitle("Consistency of Hessian Across Vectors\n"
                  "Cross scatter of EigenValues and vHv values for Hessian of 5 evolutions",
                  fontsize=18)
plt.savefig(join(figdir, "Hess_consistency_example.jpg"), bbox_extra_artists=[ST]) # this is working.
plt.show()

#%%
%%time
# eigval_i, eigvect_i = load_eig(space, 1, labstr)
# eigval_j, eigvect_j = load_eig(space, 2, labstr)
innerprod = eigvect_i.T @ eigvect_j
vHv_ij = np.diag((innerprod * eigval_j[np.newaxis, :]) @ innerprod.T) # 664 ms on cpu
# vHv_ij2 = np.diag(eigvect_i.T @ eigvect_j @ np.diag(eigval_j) @ eigvect_j.T @ eigvect_i)
#%%
%%time
eigval_i, eigvect_i = load_eig(space, 1, labstr)
eigval_j, eigvect_j = load_eig(space, 2, labstr)
ev_i = torch.from_numpy(eigvect_i).cuda()
eva_i = torch.from_numpy(eigval_i).cuda()
ev_j = torch.from_numpy(eigvect_j).cuda()
eva_j = torch.from_numpy(eigval_j).cuda()
inpr = ev_i.T @ ev_j
vHv_ij = torch.diag((inpr * eva_j.unsqueeze(0)) @ inpr.T) # 3.99ms on cuda per se
#%%
def corr_torch(V1, V2):
    C1 = (V1 - V1.mean())
    C2 = (V2 - V2.mean())
    return torch.dot(C1, C2) / C1.norm() / C2.norm()

def corr_nan_torch(V1, V2):
    Msk = torch.isnan(V1) | torch.isnan(V2)
    return corr_torch(V1[~Msk], V2[~Msk])

corr_nan_torch(eva_i.log10(), vHv_ij.log10())
#%%
T0 = time()
corr_mat_log_ctrl = torch.zeros((284,284)).cuda()
corr_mat_lin_ctrl = torch.zeros((284,284)).cuda()
for eigi in range(284):
    eigval_i, eigvect_i = load_eig(space, eigi, labstr)
    evc_i = torch.from_numpy(eigvect_i).cuda()
    eva_i = torch.from_numpy(eigval_i).cuda()
    for eigj in range(284):
        eigval_j, eigvect_j = load_eig(space, eigj, labstr)
        evc_j = torch.from_numpy(eigvect_j).cuda()
        eva_j = torch.from_numpy(eigval_j).cuda()
        inpr = evc_i.T @ evc_j
        vHv_ij = torch.diag((inpr * eva_j.unsqueeze(0)) @ inpr.T)
        corr_mat_log_ctrl[eigi, eigj] = corr_nan_torch(vHv_ij.log10(), eva_j.log10())
        corr_mat_lin_ctrl[eigi, eigj] = corr_nan_torch(vHv_ij, eva_j)
        # vHv_ij = np.diag(eigvect_i.T @ eigvect_j @ np.diag(eigval_j) @ eigvect_j.T @ eigvect_i)
        # corr_mat_log[eigi, eigj] = np.corrcoef(np.log10(vHv_ij), np.log10(eigvect_j))[0, 1]
        # corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eigvect_j)[0, 1]
    print(time() - T0)
print(time() - T0) # 3239.401490688324
corr_mat_log_ctrl = corr_mat_log_ctrl.cpu().numpy()
corr_mat_lin_ctrl = corr_mat_lin_ctrl.cpu().numpy()
#%%
np.savez(join(figdir, "evol_hess_ctrl_corr_mat.npz"), corr_mat_log=corr_mat_log_ctrl,
         corr_mat_lin=corr_mat_lin_ctrl,code_all=code_all)
#%%
figdir = r"E:\Cluster_Backup\fc6_shfl_fixGAN\summary"
data = np.load(join(figdir, "evol_hess_ctrl_corr_mat.npz"))
corr_mat_log_ctrl = data["corr_mat_log"]
corr_mat_lin_ctrl = data["corr_mat_lin"]
#%%
corr_mat_log_ctrl_nodiag = corr_mat_log_ctrl
corr_mat_lin_ctrl_nodiag = corr_mat_lin_ctrl
np.fill_diagonal(corr_mat_log_ctrl_nodiag, np.nan)
np.fill_diagonal(corr_mat_lin_ctrl_nodiag, np.nan)
log_nodiag_mean = np.nanmean(corr_mat_log_ctrl_nodiag)
lin_nodiag_mean = np.nanmean(corr_mat_lin_ctrl_nodiag)
print("Log scale mean corr value %.3f"%log_nodiag_mean)  # 0.984
print("Linear scale mean corr value %.3f"%lin_nodiag_mean)  # 0.600

Hlabel = "evol"
plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_log_ctrl, fignum=0)
plt.title("FC6GAN Hessian at evolved codes\nCorrelation Mat of log of vHv and eigenvalues"
          "\nNon-Diagonal mean %.3f"%log_nodiag_mean, fontsize=15)
plt.colorbar()
plt.subplots_adjust(top=0.85)
plt.savefig(join(figdir, "%s_Hess_ctrl_corrmat_log.jpg"%Hlabel))
plt.show()
#%
fig = plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_lin_ctrl, fignum=0)
plt.title("FC6GAN Hessian at evolved codes\nCorrelation Mat of vHv and eigenvalues"
          "\nNon-Diagonal mean %.3f"%lin_nodiag_mean, fontsize=15)
plt.colorbar()
plt.subplots_adjust(top=0.85)
plt.savefig(join(figdir, "%s_Hess_ctrl_corrmat_lin.jpg"%Hlabel))
plt.show()
#%
#%%
code_corr = np.corrcoef(code_all)
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr, fignum=0)
plt.title("Correlation Mat of code vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_corrmat.jpg"))
plt.show()
#%%
avg_data = np.load(join(r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace",
                        "Evolution_Avg_Hess.npz"))
eigvect_avg = avg_data["eigvect_avg"]
eigval_avg = avg_data["eigv_avg"]
#%%
cutoff = 400
code_corr = np.corrcoef(code_all @ eigvect_avg[:,-cutoff:])
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr, fignum=0)
plt.title("Correlation Mat of code vectors (in top %d eigenspace)"%cutoff, fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_corrmat_topeig%d.jpg"%cutoff))
plt.show()
#%%
cutoff = 400
code_corr = np.corrcoef(code_all @ eigvect_avg[:,-cutoff:])
np.fill_diagonal(code_corr,np.nan)
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr, fignum=0)
plt.title("Correlation Mat of code vectors (in top %d eigenspace)"%cutoff, fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_corrmat_topeig%d.jpg"%cutoff))
plt.show()
#%%
cutoff = 400
code_corr_rnd = np.corrcoef(np.random.randn(284,4096)*300 @ eigvect_avg[:,-cutoff:])
#%%
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr_rnd, fignum=0)
plt.title("Correlation Mat of random vectors (in top %d eigenspace)"%cutoff, fontsize=16)
plt.colorbar()
plt.show()
#%%
from scipy.stats.stats import pearsonr
np.fill_diagonal(code_corr, 1)
cc_log_raw = np.corrcoef(code_corr.reshape(-1), corr_mat_log.cpu().numpy().reshape(-1))[0,1]  # 0.300 (diagonals can
# inflate the correlation)  0.1514754571939899
cc_lin_raw = np.corrcoef(code_corr.reshape(-1), corr_mat_lin.cpu().numpy().reshape(-1))[0,1]  # 0.300 (diagonals can inflate
# the correlation)  0.3159495935200964
print("Correlation between code correlation  and  Hessian similarity ", cc_lin_raw)
print("Correlation between code correlation  and  log Hessian similarity ", cc_log_raw)
code_corr_nodiag = code_corr.copy()
corr_mat_lin_nodiag = corr_mat_lin.cpu().numpy().copy()
corr_mat_log_nodiag = corr_mat_log.cpu().numpy().copy()
np.fill_diagonal(code_corr_nodiag, np.nan)
np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
np.fill_diagonal(corr_mat_log_nodiag, np.nan)
cc_lin, p_lin = pearsonr(code_corr_nodiag[~np.isnan(code_corr_nodiag)], corr_mat_lin_nodiag[~np.isnan(code_corr_nodiag)])  # 0.0813
cc_log, p_log = pearsonr(code_corr_nodiag[~np.isnan(code_corr_nodiag)], corr_mat_log_nodiag[~np.isnan(
    corr_mat_log_nodiag)]) # 0.292
print("Correlation between code correlation  and  Hessian similarity (non-diagonal) ", cc_lin)
print("Correlation between code correlation  and  log Hessian similarity (non-diagonal) ", cc_log)
# Correlation between code correlation  and  Hessian similarity (non-diagonal)  0.020121576393265176
# Correlation between code correlation  and  log Hessian similarity (non-diagonal)  0.022462684881444483

#%% Newer version API
from Hessian.hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, compute_vector_hess_corr, \
    plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example, plot_spectra
savedir = r"E:\Cluster_Backup\fc6_shfl_fixGAN"
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\FC6GAN"
realfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\FC6GAN"

modelnm = "FC6GAN_shuffle_evol"
# Load the Hessian NPZ
eva_ctrl, evc_ctrl, feat_ctrl, meta = scan_hess_npz(savedir, "evol_(\d*)_bpfull.npz", evakey='eigvals', evckey='eigvects',
                                                    featkey='code')
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_ctrl, evc_ctrl)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_ctrl, )
# compute and plot the correlation between hessian at different points
corr_mat_log_ctrl, corr_mat_lin_ctrl = compute_hess_corr(eva_ctrl, evc_ctrl, figdir=figdir, use_cuda=True, savelabel=modelnm)
corr_mat_vec_ctrl = compute_vector_hess_corr(eva_ctrl, evc_ctrl, figdir=figdir, use_cuda=True, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_ctrl, evc_ctrl, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
#%%
with np.load(join(figdir, "spectra_col_%s.npz"%modelnm)) as data:
    eva_ctrl = data["eigval_col"]
with np.load(join(realfigdir, "spectra_col_evol.npz")) as data:
    eva_real = data["eigval_col"]
fig0 = plot_spectra(eva_real, savename="FC6GAN_shuffle_spectrum_cmp", figdir=figdir, abs=True,
            titstr="FC6GAN cmp", label="real", fig=None)
fig0 = plot_spectra(eva_ctrl, savename="FC6GAN_shuffle_spectrum_cmp", figdir=figdir, abs=True,
            titstr="FC6GAN cmp", label="shuffled", fig=fig0)
#%%
with np.load(join(realfigdir, "evol_hess_corr_mat.npz")) as data:
    corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%"real",
                                    savelabel="FC6GAN_shuffle_cmp")
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%"shuffle",
                                    savelabel="FC6GAN_shuffle_cmp", figs=(fig11, fig22))
