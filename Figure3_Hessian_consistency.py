#%%
import os
import re
from time import time
from os.path import join
from glob import glob
import sys
import pandas as pd
import numpy as np
from hessian_analysis_tools import scan_hess_npz, plot_spectra, average_H, compute_hess_corr, plot_consistency_example
from hessian_axis_visualize import vis_eigen_explore, vis_eigen_action, vis_eigen_action_row, vis_eigen_explore_row
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper, loadBigGAN, BigGAN_wrapper
import matplotlib.pylab as plt
import matplotlib
summarydir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
#%%
dataroot = "E:\\Cluster_Backup\\StyleGAN2"
modelnm = "stylegan2-cat-config-f"
modelsnm = "Cat256"
SGAN = loadStyleGAN2(modelnm+".pt", size=256,)
SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
# H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
# np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
#%%
vis_eigen_action_row(eigvec_col[0], feat_col[0, :], maxdist=120, rown=7, )
#%%
veci = 76
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\Figure3"
vis_eigen_explore_row(feat_col[veci, :], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir, indivimg=True,
              namestr="Cat256_vec%d"%veci, eiglist=[2,3,7,15,31], maxdist=2.5, rown=5, )
#%%
data = np.load(join(summarydir, "BigGAN", "Hess_all_consistency_corr_mat.npz"))
corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]
np.fill_diagonal(corr_mat_lin, np.nan)
np.fill_diagonal(corr_mat_log, np.nan)
plt.hist(corr_mat_log.flatten()[~np.isnan(corr_mat_log.flatten())], 60, density=False)
corr_mean = np.nanmean(corr_mat_log)
corr_medi = np.nanmedian(corr_mat_log)
plt.vlines(corr_mean, *plt.ylim(), linestyles="dashed", color="red")
plt.vlines(corr_medi, *plt.ylim(), linestyles="dashed", color="orange")
plt.show()
#%%
plt.hist(corr_mat_lin.flatten()[~np.isnan(corr_mat_lin.flatten())], 60, density=False)
corr_mean = np.nanmean(corr_mat_lin)
corr_medi = np.nanmedian(corr_mat_lin)
plt.vlines(corr_mean, *plt.ylim(), linestyles="dashed", color="red")
plt.vlines(corr_medi, *plt.ylim(), linestyles="dashed", color="black")
plt.show()

#%%
def histogram_corrmat(corr_mat_lin, log=True, GAN="GAN"):
    fig = plt.figure(figsize=[4, 3])
    plt.hist(corr_mat_lin.flatten()[~np.isnan(corr_mat_lin.flatten())], 60, density=True)
    corr_mean = np.nanmean(corr_mat_lin)
    corr_medi = np.nanmedian(corr_mat_lin)
    _, YMAX = plt.ylim()
    plt.vlines(corr_mean, 0, YMAX, linestyles="dashed", color="black")
    plt.vlines(corr_medi, 0, YMAX, linestyles="dashed", color="red")
    plt.xlabel("corr(log(V_iH_jV_i), log(Lambda_j))" if log else "corr(V_iH_jV_i, Lambda_j)")
    plt.ylabel("density")
    plt.title("Histogram of Non-Diag Correlation\n %s on %s scale\n mean %.3f median %.3f" %
              (GAN, "log" if log else "lin", corr_mean, corr_medi))
    plt.show()
    return fig
#%%
GANsnm = "StyleGAN2-Face512"
modeldir = join(summarydir, "StyleGAN2")
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(r"E:\Cluster_Backup\StyleGAN2\ffhq-512-avg-tpurun1",
                                                      featkey="feat")
corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, savelabel="ffhq-512-avg-tpurun1",
                                              figdir=modeldir, )
plot_consistency_example(eigval_col, eigvec_col, nsamp=5, titstr=GANsnm, figdir=modeldir, savelabel="ffhq-512-avg-tpurun1")


#%%
corrmat_npz_dict = {"fc6GAN": "fc6GAN\\evol_hess_corr_mat.npz",
     "DCGAN": "DCGAN\\Hess__corr_mat.npz",
     "BigGAN": "BigGAN\\Hess_all_consistency_corr_mat.npz",
     "BigBiGAN": "BigBiGAN\\evol_hess_corr_mat.npz",
     "PGGAN": "PGGAN\\Hess__corr_mat.npz",
     "StyleGAN-Face": "StyleGAN\\Hess__corr_mat.npz",
     "StyleGAN2-Face512": "StyleGAN2\\Hess_ffhq-512-avg-tpurun1_corr_mat.npz",
     "StyleGAN2-Face256": "StyleGAN2\\Hess_ffhq-256-config-e-003810_corr_mat.npz",
     "StyleGAN2-Cat256": "StyleGAN2\\Hess_stylegan2-cat-config-f_corr_mat.npz", }

spectra_npz_dict = {"fc6GAN": "FC6GAN\\spectra_col_evol.npz",
          "DCGAN": "DCGAN\\spectra_col_BP.npz",
          "BigGAN": "BigGAN\\spectra_col.npz",
          "BigBiGAN": "BigBiGAN\\spectra_col.npz",
          "PGGAN": "PGGAN\\spectra_col_BP.npz",
          "StyleGAN-Face": "StyleGAN\\spectra_col_face256_BP.npz",
          "StyleGAN2-Face512": "StyleGAN2\\spectra_col_FFHQ512.npz",
          "StyleGAN2-Face256": "StyleGAN2\\spectra_col_ffhq-256-config-e-003810_BP.npz",
          "StyleGAN2-Cat256": "StyleGAN2\\spectra_col_stylegan2-cat-config-f_BP.npz", }

consist_tab = {}
for GAN, path in corrmat_npz_dict.items():
    folder, npzfn = path.split("\\")
    npzname, _ = npzfn.split(".")
    data = np.load(join(summarydir, path))
    corr_mat_log, corr_mat_lin = data["corr_mat_log"].copy(), data["corr_mat_lin"].copy()
    np.fill_diagonal(corr_mat_lin, np.nan)
    np.fill_diagonal(corr_mat_log, np.nan)
    consist_tab[GAN] = (np.nanmean(corr_mat_log), np.nanstd(corr_mat_log), np.nanmean(corr_mat_lin), np.nanstd(
        corr_mat_lin))
    fig1 = histogram_corrmat(corr_mat_log, log=True, GAN=GAN)
    fig1.savefig(join(summarydir, folder, npzname+"_log_hist.pdf"))
    fig1.savefig(join(summarydir, folder, npzname+"_log_hist.png"))
    fig1.savefig(join(summarydir, folder + npzname+"_log_hist.pdf"))

    fig2 = histogram_corrmat(corr_mat_lin, log=False, GAN=GAN)
    fig2.savefig(join(summarydir, folder, npzname + "_lin_hist.pdf"))
    fig2.savefig(join(summarydir, folder, npzname + "_lin_hist.png"))
    fig2.savefig(join(summarydir, folder + npzname+"_lin_hist.pdf"))
#%% Print some statistics for anisotropy
cutoff_tab = {}
for GAN, npzpath in spectra_npz_dict.items():
    folder, npzfn = npzpath.split("\\")
    npzname, _ = npzfn.split(".")
    data = np.load(join(summarydir, npzpath))
    eig_mean = np.mean(data['eigval_col'], axis=0)
    eigabssort = np.sort(np.abs(eig_mean))[::-1]
    expvar = np.cumsum(eigabssort)/eigabssort.sum()
    dimen = len(eig_mean)
    cutoff_nums = (sum(expvar<0.99), sum(expvar<0.999), sum(expvar<0.9999), sum(expvar<0.99999), dimen)
    cutoff_tab[GAN] = cutoff_nums
    print("%s %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var, total %d"%(GAN,
                        *cutoff_nums))
#%% Summarize the Hessian info in CSV files
GAN_geom_summary = pd.DataFrame.from_dict(consist_tab, orient="index", columns=["log_corr_mean", "log_corr_std",
                                          "lin_corr_mean",  "lin_corr_std"])
GAN_cutoff_summary = pd.DataFrame.from_dict(cutoff_tab, orient="index", columns=["dim99", "dim999",
                                          "dim9999",  "dim9999", "dimen"])
GAN_geom_summary.to_csv(join(summarydir, "Hess_consistency_table.csv"))
GAN_summary = pd.concat((GAN_geom_summary, GAN_cutoff_summary, ), axis=1)
GAN_summary.to_csv(join(summarydir, "Hess_summary_table.csv"))
#%% Synopsis of GAN hessian consistency
fig, ax = plt.subplots()
plt.errorbar(GAN_geom_summary.lin_corr_mean, GAN_geom_summary.log_corr_mean, yerr=GAN_geom_summary.log_corr_std,
             xerr=GAN_geom_summary.lin_corr_std, fmt='o')
plt.xlim([0,1])
for i, GAN in enumerate(GAN_geom_summary.index):
    ax.annotate(GAN, (GAN_geom_summary.lin_corr_mean[i], GAN_geom_summary.log_corr_mean[i]), fontsize=12)
plt.ylabel("log scale corr")
plt.xlabel("lin scale corr")
plt.box(True)
plt.savefig(join(summarydir, "Hess_corrmat_synopsis.png"))
plt.savefig(join(summarydir, "Hess_corrmat_synopsis.pdf"))
plt.show()

#%% Plot an Example of Hessian consistency
BGHpath = "E:\Cluster_Backup\BigGANH"
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(BGHpath, "Hess_cls(\d*).npz", evakey='eigvals',
                                                     evckey='eigvects', featkey='vect')
#%%
figdir = join(summarydir, "BigGAN")
plot_consistency_example(eigval_col, eigvec_col, nsamp=3, titstr="BigGAN", figdir=figdir, savelabel="BigGAN_all")