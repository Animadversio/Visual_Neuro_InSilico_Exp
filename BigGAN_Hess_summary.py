"""
This script summarize and the Hessian computation for BigGAN
Analyze the geometry of the BigGAN manifold. How the metric tensor relates to the coordinate.
"""
#%%
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages

BGdatadir = r"E:\Cluster_Backup\BigGANH"
figdir = r"E:\Cluster_Backup\BigGANH\summary"
#%%
# np.savez(join(savedir, "Hess_cls%d.npz" % class_id), H=H, H_nois=H_nois, H_clas=H_clas, eigvals=eigvals,
#              eigvects=eigvects, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, eigvals_nois=eigvals_nois,
#              eigvects_nois=eigvects_nois, vect=ref_vect.cpu().numpy(),
#              noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())

eigval_col = []
eigvals_clas_col = []
eigvals_nois_col = []
for class_id in range(1000):
    fn = "Hess_cls%d.npz"%class_id
    data = np.load(join(BGdatadir, fn))
    eigvals = data["eigvals"]
    eigval_col.append(eigvals.copy())
    eigvals_clas = data["eigvals_clas"]
    eigvals_clas_col.append(eigvals_clas.copy())
    eigvals_nois = data["eigvals_nois"]
    eigvals_nois_col.append(eigvals_nois.copy())
    # H_nois = data["H_nois"]
    # H_clas = data["H_clas"]
#%%
eigval_arr = np.array(eigval_col)
eigval_nois_arr = np.array(eigvals_nois_col)
eigval_clas_arr = np.array(eigvals_clas_col)
eigmean = eigval_arr[:, ::-1].mean(axis=0)
eigstd = eigval_arr[:, ::-1].std(axis=0)
eiglim = np.percentile(eigval_arr[:,::-1], [5, 95], axis=0)
eigmean_cls = eigval_clas_arr[:, ::-1].mean(axis=0)
eigstd_cls = eigval_clas_arr[:, ::-1].std(axis=0) # std can generate negative errorbar. bad
eiglim_cls = np.percentile(eigval_clas_arr[:, ::-1], [5, 95], axis=0) # use percentile is more reasonable in this
# non-Gaussian sense
eigmean_nos = eigval_nois_arr[:, ::-1].mean(axis=0)
eigstd_nos = eigval_nois_arr[:, ::-1].std(axis=0)
eiglim_nos = np.percentile(eigval_nois_arr[:, ::-1], [5, 95], axis=0)
#%%
plt.subplot(1,2,1)
plt.plot(range(256), eigmean)  #, eigval_arr.std(axis=0)
plt.fill_between(range(256), eiglim[0, :], eiglim[1, :], alpha=0.3)
plt.subplot(1,2,2)
plt.plot(range(256), np.log10(eigmean))  #, eigval_arr.std(axis=0)
plt.fill_between(range(256), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3)
plt.show()
#%%
fig = plt.figure(figsize=[10, 5])
plt.subplot(1,2,1)
plt.plot(range(256), eigmean, alpha=0.6)  #, eigval_arr.std(axis=0)
plt.fill_between(range(256), eiglim[0, :], eiglim[1, :], alpha=0.3, label="all space")
plt.plot(range(128), eigmean_cls, alpha=0.6)
# plt.fill_between(range(128), eigmean_cls - eigstd_cls, eigmean_cls + eigstd_cls, alpha=0.3)
plt.fill_between(range(128), eiglim_cls[0, :], eiglim_cls[1, :], alpha=0.3, label="class space")
plt.plot(range(128), eigmean_nos, alpha=0.6)
plt.fill_between(range(128), eiglim_nos[0, :], eiglim_nos[1, :], alpha=0.3, label="noise space")
plt.ylabel("eigenvalue")
plt.xlabel("eig id")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(256), np.log10(eigmean), alpha=0.6)  #, eigval_arr.std(axis=0)
plt.fill_between(range(256), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label="all space")
plt.plot(range(128), np.log10(eigmean_cls), alpha=0.6)
plt.fill_between(range(128), np.log10(eiglim_cls[0, :]), np.log10(eiglim_cls[1, :]), alpha=0.3,
                 label="class space")
plt.plot(range(128), np.log10(eigmean_nos), alpha=0.6)
plt.fill_between(range(128), np.log10(eiglim_nos[0, :]), np.log10(eiglim_nos[1, :]), alpha=0.3,
                 label="noise space")
plt.ylabel("eigenvalue(log)")
plt.xlabel("eig id")
plt.legend()
# fig.subplots_adjust(top=0.85)
st = plt.suptitle("Hessian Spectrum of BigGAN in Different Spaces\n (error bar for [5,95] percentile in 1000 classes)")
plt.savefig(join(figdir, "spectrum_stat_all3.jpg"), bbox_extra_artists=[st]) # this is working.
plt.show()
#%%
def plot_spectra(wholespace=True, subspace=True, savename="spectrum_stat_all3.jpg", ):
    """A local function to compute these figures for different subspaces. """
    fig = plt.figure(figsize=[10, 5])
    plt.subplot(1,2,1)
    if wholespace:
        plt.plot(range(256), eigmean, alpha=0.6)  #, eigval_arr.std(axis=0)
        plt.fill_between(range(256), eiglim[0, :], eiglim[1, :], alpha=0.3, label="all space")
    if subspace:
        plt.plot(range(128), eigmean_cls, alpha=0.6)
        plt.fill_between(range(128), eiglim_cls[0, :], eiglim_cls[1, :], alpha=0.3, label="class space")
        plt.plot(range(128), eigmean_nos, alpha=0.6)
        plt.fill_between(range(128), eiglim_nos[0, :], eiglim_nos[1, :], alpha=0.3, label="noise space")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.legend()
    plt.subplot(1,2,2)
    if wholespace:
        plt.plot(range(256), np.log10(eigmean), alpha=0.6)  #, eigval_arr.std(axis=0)
        plt.fill_between(range(256), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label="all space")
    if subspace:
        plt.plot(range(128), np.log10(eigmean_cls), alpha=0.6)
        plt.fill_between(range(128), np.log10(eiglim_cls[0, :]), np.log10(eiglim_cls[1, :]), alpha=0.3,
                         label="class space")
        plt.plot(range(128), np.log10(eigmean_nos), alpha=0.6)
        plt.fill_between(range(128), np.log10(eiglim_nos[0, :]), np.log10(eiglim_nos[1, :]), alpha=0.3,
                         label="noise space")
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of BigGAN in Different Spaces\n (error bar for [5,95] percentile in 1000 classes)")
    plt.savefig(join(figdir, savename), bbox_extra_artists=[st]) # this is working.
    plt.show()

plot_spectra(wholespace=True, subspace=True, savename="spectrum_stat_all3.jpg", )
plot_spectra(wholespace=True, subspace=False, savename="spectrum_stat_all256.jpg", )
plot_spectra(wholespace=False, subspace=True, savename="spectrum_stat_clas_nois.jpg", )
# def shaded_error_bar(x, y, err):
#     fig = plt.figure()
#     plt.plot(x, y)  #, eigval_arr.std(axis=0)
#     plt.fill_between(x, y - err, y + err, alpha=0.3)
#     return fig

#%% Load all the class space Hessian and examine their effect on the eigenvalues of each other.
H_col = []
eigvals_col = []
eigvects_col = []
for class_id in range(1000):
    fn = "Hess_cls%d.npz" % class_id
    data = np.load(join(BGdatadir, fn))
    H_clas = data["H_clas"]
    eigvals_clas = data["eigvals_clas"]
    eigvects_clas = data["eigvects_clas"]
    H_col.append(H_clas)
    eigvals_col.append(eigvals_clas)
    eigvects_col.append(eigvects_clas)
#%%
fig = plt.figure(figsize=[10, 10], constrained_layout=False)
spec = fig.add_gridspec(ncols=5, nrows=5, left=0.05, right=0.95, top=0.9, bottom=0.05)
for eigi in range(5):
    for eigj in range(5):
        vHv_ij = np.diag(eigvects_col[eigi].T @ H_col[eigj] @ eigvects_col[eigi])
        ax = fig.add_subplot(spec[eigi, eigj])
        if eigi == eigj:
            ax.hist(np.log10(eigvals_col[eigj]), 20)
        else:
            ax.scatter(np.log10(eigvals_col[eigj]), np.log10(vHv_ij), s=15, alpha=0.6)
            ax.set_aspect(1, adjustable='datalim')
        if eigi == 4:
            ax.set_xlabel("eigvals %d" % eigj)
        if eigj == 0:
            ax.set_ylabel("vHv eigvects %d" % eigj)
ST = plt.suptitle("Consistency of Hessian Across Class\n"
                  "Cross scatter of EigenValues and vHv values for Hessian of 5 classes",
                  fontsize=18)
plt.savefig(join(figdir, "Hess_consistency.jpg"), bbox_extra_artists=[ST]) # this is working.
plt.show()
#%%
# import pandas as pd
# df = pd.DataFrame({'a': np.random.randint(0, 50, 1000)})
# df['b'] = df['a'] + np.random.normal(0, 10, 1000) # positively correlated with 'a'
# df['c'] = 100 - df['a'] + np.random.normal(0, 5, 1000) # negatively correlated with 'a'
# df['d'] = np.random.randint(0, 50, 1000) # not correlated with 'a'
# df.corr()
#%%
Hlabel = "noise"
H_col = []
eigvals_col = []
eigvects_col = []
for class_id in range(1000):
    fn = "Hess_cls%d.npz" % class_id
    data = np.load(join(BGdatadir, fn))
    H_clas = data["H_clas"]
    eigvals_clas = data["eigvals_clas"]
    eigvects_clas = data["eigvects_clas"]
    # H_clas = data["H_nois"]
    # eigvals_clas = data["eigvals_nois"]
    # eigvects_clas = data["eigvects_nois"]
    # H_clas = data["H"]
    # eigvals_clas = data["eigvals"]
    # eigvects_clas = data["eigvects"]
    H_col.append(H_clas)
    eigvals_col.append(eigvals_clas)
    eigvects_col.append(eigvects_clas)
#%
T0 = time()
corr_mat_log = np.zeros((1000, 1000))
corr_mat_lin = np.zeros((1000, 1000))
for eigi in range(1000):
    for eigj in range(1000):
        vHv_ij = np.diag(eigvects_col[eigi].T @ H_col[eigj] @ eigvects_col[eigi])
        corr_mat_log[eigi, eigj] = np.corrcoef(np.log10(vHv_ij), np.log10(eigvals_col[eigj]))[0, 1]
        corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eigvals_col[eigj])[0, 1]

print("%.1f sec" % (time() - T0)) # 582.2 secs for the 1000 by 1000 mat. not bad!
#%
np.savez(join(figdir, "Hess_%s_consistency_corr_mat.npz"%Hlabel), corr_mat_log=corr_mat_log, corr_mat_lin=corr_mat_lin)
#%
plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_log, fignum=0)
plt.title("Correlation Mat of log of vHv and eigenvalues", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "Hess_%s_corrmat_log.jpg"%Hlabel))
plt.show()
#%
fig = plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_lin, fignum=0)
plt.title("Correlation Mat of vHv and eigenvalues", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "Hess_%s_corrmat_lin.jpg"%Hlabel))
plt.show()
#%
corr_mat_log_nodiag = corr_mat_log.copy()
np.fill_diagonal(corr_mat_log_nodiag, np.nan) # corr_mat_log_nodiag =
print("mean corr value %.3f"%np.nanmean(corr_mat_log_nodiag))
#%%
from pytorch_pretrained_biggan import BigGAN
G = BigGAN.from_pretrained("biggan-deep-256")
embed_mat = G.embeddings.weight.detach().numpy()
del G
#%%
embed_corr = np.corrcoef(embed_mat.T)
fig = plt.figure(figsize=[10, 8])
plt.matshow(embed_corr, fignum=0)
plt.title("Correlation Mat of 128d embedding vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "embed_vec_corrmat.jpg"))
plt.show()
#%%
normvec = np.linalg.norm(embed_mat, axis=0, keepdims=1)**2
embed_distmat = normvec + normvec.T - 2 * embed_mat.T @ embed_mat
embed_corr = np.corrcoef(embed_mat.T)
fig = plt.figure(figsize=[10, 8])
plt.matshow(-embed_distmat, fignum=0)
plt.title("-L2 Distance Mat of 128d embedding vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "embed_vec_L2distmat.jpg"))
plt.show()
#%%
from scipy.stats.stats import pearsonr
np.corrcoef(embed_corr.reshape(-1), corr_mat_log.reshape(-1))  # 0.300 (diagonals can inflate the correlation)
embed_corr_nodiag = embed_corr.copy()
corr_mat_lin_nodiag = corr_mat_lin.copy()
corr_mat_log_nodiag = corr_mat_log.copy()
np.fill_diagonal(embed_corr_nodiag, np.nan)
np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
np.fill_diagonal(corr_mat_log_nodiag, np.nan)
# np.corrcoef(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)], corr_mat_lin_nodiag[~np.isnan(corr_mat_lin_nodiag)]) # 0.081
cc_lin, p_lin = pearsonr(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)], corr_mat_lin_nodiag[~np.isnan(corr_mat_lin_nodiag)])  # 0.0813
# np.corrcoef(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)], corr_mat_log_nodiag[~np.isnan(
#     corr_mat_log_nodiag)])
cc_log, p_log = pearsonr(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)], corr_mat_log_nodiag[~np.isnan(
    corr_mat_log_nodiag)]) # 0.292
#%%
plt.figure(figsize=[8, 8 ])
plt.scatter(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)], corr_mat_log_nodiag[~np.isnan(corr_mat_log_nodiag)],
            s=10, alpha=0.4)
plt.xlabel("corr of embed vector")
plt.ylabel("corr of Hessian (vHv and eigenvalue)")
plt.savefig(join(figdir, "corr_embedvect_vs_corr_H.jpg"))
plt.show()
#%%
# plt.figure(figsize=[8, 8 ])
# plt.scatter(np.abs(embed_corr_nodiag[~np.isnan(embed_corr_nodiag)]), corr_mat_log_nodiag[~np.isnan(
#     corr_mat_log_nodiag)],
#             s=10, alpha=0.4)
# plt.xlabel("abs(corr) of embed vector")
# plt.ylabel("corr of Hessian (vHv and eigenvalue)")
# plt.show()

#%% Compute the averaged Hessian tensor in the space
#   (Naive way of averaging. No cut off on spectrum.)
H_clas_avg = None
H_nois_avg = None
H_avg = None
for class_id in range(1000):
    fn = "Hess_cls%d.npz" % class_id
    data = np.load(join(BGdatadir, fn))
    H_clas = data["H_clas"]
    H_nois = data["H_nois"]
    H = data["H"]
    H_clas_avg = H_clas_avg + H_clas if H_clas_avg is not None else H_clas
    H_nois_avg = H_nois_avg + H_nois if H_nois_avg is not None else H_nois
    H_avg = H_avg + H if H_avg is not None else H
#%%
H_clas_avg = H_clas_avg / 1000
H_nois_avg = H_nois_avg / 1000
H_avg = H_avg / 1000
eigvals_clas_avg, eigvects_clas_avg = np.linalg.eigh(H_clas_avg)
eigvals_nois_avg, eigvects_nois_avg = np.linalg.eigh(H_nois_avg)
eigvals_avg, eigvects_avg = np.linalg.eigh(H_avg)
np.savez(join(figdir, "H_avg_1000cls.npz"),
eigvals_clas_avg=eigvals_clas_avg, eigvects_clas_avg=eigvects_clas_avg, H_clas_avg=H_clas_avg, 
eigvals_nois_avg=eigvals_nois_avg, eigvects_nois_avg=eigvects_nois_avg, H_nois_avg=H_nois_avg, 
eigvals_avg=eigvals_avg,  eigvects_avg=eigvects_avg, H_avg=H_avg,)
#%%
wholespace, subspace = True, True
fig = plt.figure(figsize=[10, 5])
plt.subplot(1,2,1)
if wholespace:
    plt.plot(range(256), eigvals_avg[::-1], alpha=0.6, label="all space")
if subspace:
    plt.plot(range(128), eigvals_clas_avg[::-1], alpha=0.6, label="class space")
    plt.plot(range(128), eigvals_nois_avg[::-1], alpha=0.6, label="noise space")
plt.ylabel("eigenvalue")
plt.xlabel("eig id")
plt.legend()
plt.subplot(1,2,2)
if wholespace:
    plt.plot(range(256), np.log10(eigvals_avg[::-1]), alpha=0.6, label="all space") 
if subspace:
    plt.plot(range(128), np.log10(eigvals_clas_avg[::-1]), alpha=0.6, label="class space")
    plt.plot(range(128), np.log10(eigvals_nois_avg[::-1]), alpha=0.6, label="noise space")
plt.ylabel("eigenvalue(log)")
plt.xlabel("eig id")
plt.legend()
st = plt.suptitle("Spectrum of Average Hessian across 1000 classes, BigGAN in Different Spaces")
plt.savefig(join(figdir, "H_avg_spectrum.jpg"), bbox_extra_artists=[st]) # this is working.
plt.show()