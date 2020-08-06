"""
This script summarize and the Hessian computation for FC6 GAN.
at the inverted codes of Pasupathy patches or Evolved codes from monkey experiments.

This is deployed on cluster for painless computation of lots of matrix multiplication
"""
#%%
import sys
sys.path.append("/home/binxu/Visual_Neuro_InSilico_Exp/")
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
import torch
import os
import tqdm
#%% Specify the path to Matrices
if sys.platform == "linux":
    savedir = r"/scratch/binxu/GAN_hessian/fc6_shfl_fixGAN"
else:
    savedir = r"E:\Cluster_Backup\fc6_shfl_fixGAN"
space = "evol"
method = "BP"

figdir = join(savedir, "summary")
os.makedirs(figdir, exist_ok=True)
labeldict = {"BP": "bpfull", "BackwardIter": "bkwlancz", "ForwardIter": "frwlancz"}
labstr = labeldict[method]
#%%
def load_eig(space, idx, labstr):
    """3 var in the npz file 'eigvals', 'eigvects', 'code'"""
    fn = "%s_%03d_%s.npz" % (space, idx, labstr)
    data = np.load(join(savedir, fn))
    eigvals = data['eigvals']
    eigvects = data['eigvects']
    return eigvals, eigvects

def corr_torch(V1, V2):
    C1 = (V1 - V1.mean())
    C2 = (V2 - V2.mean())
    return torch.dot(C1, C2) / C1.norm() / C2.norm()

def corr_nan_torch(V1, V2):
    Msk = torch.isnan(V1) | torch.isnan(V2)
    return corr_torch(V1[~Msk], V2[~Msk])
#%%
eigvals_col = []
code_all = []
for idx in range(284): # Note load it altogether is very slow, not recommended
    fn = "%s_%03d_%s.npz" % (space, idx, labstr)
    data = np.load(join(savedir, fn))
    eigvals = data['eigvals']
    code = data['code']
    eigvals_col.append(eigvals.copy())
    code_all.append(code.copy())
#%
code_all = np.array(code_all)
eigval_arr_ctrl = np.array(eigvals_col)
eigmean_ctrl = eigval_arr_ctrl[:, ::-1].mean(axis=0)
eigstd_ctrl = eigval_arr_ctrl[:, ::-1].std(axis=0)
eiglim_ctrl = np.percentile(eigval_arr_ctrl[:, ::-1], [5, 95], axis=0)

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
T0 = time()
corr_mat_log_ctrl = torch.zeros((284, 284)).cuda()
corr_mat_lin_ctrl = torch.zeros((284, 284)).cuda()
for eigi in tqdm.trange(284):
    eigval_i, eigvect_i = load_eig(space, eigi, labstr)
    evc_i = torch.from_numpy(eigvect_i).cuda()
    eva_i = torch.from_numpy(eigval_i).cuda()
    for eigj in tqdm.trange(284):
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
print(time() - T0)  # 3239.401490688324
corr_mat_log_ctrl = corr_mat_log_ctrl.cpu().numpy()
corr_mat_lin_ctrl = corr_mat_lin_ctrl.cpu().numpy()
#%
np.savez(join(figdir, "evol_hess_ctrl_corr_mat.npz"), corr_mat_log=corr_mat_log_ctrl,
         corr_mat_lin=corr_mat_lin_ctrl, code_all=code_all)
#%%
Hlabel = "evol"
plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_log_ctrl, fignum=0)
plt.title("FC6GAN Hessian at evolved codes\nCorrelation Mat of log of vHv and eigenvalues", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "%s_Hess_ctrl_corrmat_log.jpg"%Hlabel))
plt.show()
#%
fig = plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_lin_ctrl, fignum=0)
plt.title("FC6GAN Hessian at evolved codes\nCorrelation Mat of vHv and eigenvalues", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "%s_Hess_ctrl_corrmat_lin.jpg"%Hlabel))
plt.show()
#%
corr_mat_log_ctrl_nodiag = corr_mat_log_ctrl
corr_mat_lin_ctrl_nodiag = corr_mat_lin_ctrl
np.fill_diagonal(corr_mat_log_ctrl_nodiag, np.nan)
np.fill_diagonal(corr_mat_lin_ctrl_nodiag, np.nan)
print("Log scale mean corr value %.3f"%np.nanmean(corr_mat_log_ctrl_nodiag))  # 0.984
print("Linear scale mean corr value %.3f"%np.nanmean(corr_mat_lin_ctrl_nodiag))  # 0.600

code_corr = np.corrcoef(code_all)
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr, fignum=0)
plt.title("Correlation Mat of code vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_corrmat.jpg"))
plt.show()
