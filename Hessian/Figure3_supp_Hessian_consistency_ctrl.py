"""Compute the Hessian consistency and how that compare their control! """
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
summaryroot = r"E:\OneDrive - Washington University in St. Louis"
#%%
"""Newer version of the plot with noise free StyleGAN2s and W space spectra"""
corrmat_npz_dict = {"fc6GAN": "fc6GAN\\evol_hess_corr_mat.npz",
     "DCGAN": "DCGAN\\Hess__corr_mat.npz",
     "BigGAN": "BigGAN\\Hess_all_consistency_corr_mat.npz",
    "BigGAN_noise": "BigGAN\\Hess_noise_consistency_corr_mat.npz",
    "BigGAN_class": "BigGAN\\Hess_class_consistency_corr_mat.npz",
     "BigBiGAN": "BigBiGAN\\evol_hess_corr_mat.npz",
     "PGGAN": "PGGAN\\Hess__corr_mat.npz",
     "StyleGAN-Face*": "StyleGAN\\Hess__corr_mat.npz",
     "StyleGAN2-Face512*": "StyleGAN2\\Hess_ffhq-512-avg-tpurun1_corr_mat.npz",
     "StyleGAN2-Face256*": "StyleGAN2\\Hess_ffhq-256-config-e-003810_corr_mat.npz",
     "StyleGAN2-Cat256*": "StyleGAN2\\Hess_stylegan2-cat-config-f_corr_mat.npz",
      "StyleGAN-Face_Z": "StyleGAN_Fix\\StyleGAN_Face256_fix\\Hess_StyleGAN_Face256_fix_corr_mat.npz",
      "StyleGAN2-Face512_Z": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix\\Hess_ffhq-512-avg-tpurun1_fix_corr_mat.npz",
      "StyleGAN2-Face256_Z": "StyleGAN2_Fix\\ffhq-256-config-e-003810_fix\\Hess_ffhq-256-config-e-003810_fix_corr_mat.npz",
      "StyleGAN2-Cat256_Z": "StyleGAN2_Fix\\stylegan2-cat-config-f_fix\\Hess_stylegan2-cat-config-f_fix_corr_mat.npz",
      "StyleGAN-Face_W": "StyleGAN_Fix\\StyleGAN_Face256_W_fix\\Hess_StyleGAN_Face256_W_fix_corr_mat.npz",
      "StyleGAN2-Face512_W": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix\\Hess_ffhq-512-avg-tpurun1_W_fix_corr_mat.npz",
      "StyleGAN2-Face256_W": "StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix\\Hess_ffhq-256-config-e-003810_W_fix_corr_mat.npz",
      "StyleGAN2-Cat256_W": "StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix\\Hess_stylegan2-cat-config-f_W_fix_corr_mat"
                            ".npz",
                    }

Havg_npz_dict = {"fc6GAN": "fc6GAN\\Evolution_Avg_Hess.npz",
     "DCGAN": "DCGAN\\H_avg_DCGAN.npz",
     "BigGAN": "BigGAN\\H_avg_1000cls.npz",
    "BigGAN_noise": "BigGAN\\H_avg_1000cls.npz",
    "BigGAN_class": "BigGAN\\H_avg_1000cls.npz",
     "BigBiGAN": "BigBiGAN\\H_avg_BigBiGAN.npz",
     "PGGAN": "PGGAN\\H_avg_PGGAN.npz",
     "StyleGAN-Face*": "StyleGAN\\H_avg_StyleGAN.npz",
     "StyleGAN2-Face512*": "StyleGAN2\\H_avg_ffhq-512-avg-tpurun1.npz",
     "StyleGAN2-Face256*": "StyleGAN2\\H_avg_ffhq-256-config-e-003810.npz",
     "StyleGAN2-Cat256*": "StyleGAN2\\H_avg_stylegan2-cat-config-f.npz",
      "StyleGAN-Face_Z": "StyleGAN_Fix\\StyleGAN_Face256_fix\\H_avg_StyleGAN_Face256_fix.npz",
      "StyleGAN2-Face512_Z": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix\\H_avg_ffhq-512-avg-tpurun1_fix.npz",
      "StyleGAN2-Face256_Z": "StyleGAN2_Fix\\ffhq-256-config-e-003810_fix\\H_avg_ffhq-256-config-e-003810_fix.npz",
      "StyleGAN2-Cat256_Z": "StyleGAN2_Fix\\stylegan2-cat-config-f_fix\\H_avg_stylegan2-cat-config-f_fix.npz",
      "StyleGAN-Face_W": "StyleGAN_Fix\\StyleGAN_Face256_W_fix\\H_avg_StyleGAN_Face256_W_fix.npz",
      "StyleGAN2-Face512_W": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix\\H_avg_ffhq-512-avg-tpurun1_W_fix.npz",
      "StyleGAN2-Face256_W": "StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix\\H_avg_ffhq-256-config-e-003810_W_fix.npz",
      "StyleGAN2-Cat256_W": "StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix\\H_avg_stylegan2-cat-config-f_W_fix"
                            ".npz",
                    }

spectra_npz_dict = {"fc6GAN": "FC6GAN\\spectra_col_evol.npz",
          "DCGAN": "DCGAN\\spectra_col_BP.npz",
          "BigGAN": "BigGAN\\spectra_col.npz",
          "BigGAN_noise": "BigGAN\\spectra_col.npz",
          "BigGAN_class": "BigGAN\\spectra_col.npz",
          "BigBiGAN": "BigBiGAN\\spectra_col.npz",
          "PGGAN": "PGGAN\\spectra_col_BP.npz",
          "StyleGAN-Face*": "StyleGAN\\spectra_col_face256_BP.npz",
          "StyleGAN2-Face512*": "StyleGAN2\\spectra_col_FFHQ512.npz",
          "StyleGAN2-Face256*": "StyleGAN2\\spectra_col_ffhq-256-config-e-003810_BP.npz",
          "StyleGAN2-Cat256*": "StyleGAN2\\spectra_col_stylegan2-cat-config-f_BP.npz",
          "StyleGAN-Face_Z": "StyleGAN_Fix\\StyleGAN_Face256_fix\\spectra_col_StyleGAN_Face256_fix.npz",
          "StyleGAN2-Face512_Z": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix\\spectra_col_ffhq-512-avg-tpurun1_fix.npz",
          "StyleGAN2-Face256_Z": "StyleGAN2_Fix\\ffhq-256-config-e-003810_fix\\spectra_col_ffhq-256-config-e-003810_fix.npz",
          "StyleGAN2-Cat256_Z": "StyleGAN2_Fix\\stylegan2-cat-config-f_fix\\spectra_col_stylegan2-cat-config-f_fix.npz",
          "StyleGAN-Face_W": "StyleGAN_Fix\\StyleGAN_Face256_W_fix\\spectra_col_StyleGAN_Face256_W_fix.npz",
          "StyleGAN2-Face512_W": "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix\\spectra_col_ffhq-512-avg-tpurun1_W_fix.npz",
          "StyleGAN2-Face256_W": "StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix\\spectra_col_ffhq-256-config-e-003810_W_fix.npz",
          "StyleGAN2-Cat256_W": "StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix\\spectra_col_stylegan2-cat-config-f_W_fix.npz",
          }

ctrl_corrmat_npz_dict = {"fc6GAN": "HessNetArchit\\FC6GAN\\Hess_FC6GAN_shuffle_evol_corr_mat.npz",
     "DCGAN": "HessNetArchit\\DCGAN\\Hess_DCGAN_shuffle_corr_mat.npz",
     "BigGAN": "HessNetArchit\\BigGAN\\Hess_BigGAN_shuffle_corr_mat.npz",
    # "BigGAN_noise": "HessNetArchit\\BigGAN\\Hess_noise_consistency_corr_mat.npz",
    # "BigGAN_class": "HessNetArchit\\BigGAN\\Hess_class_consistency_corr_mat.npz",
     "BigBiGAN": None, #"HessNetArchit\\BigBiGAN\\Hess_BigBiGAN_shuffle_corr_mat.npz",
     "PGGAN": "HessNetArchit\\PGGAN\\Hess_PGGAN_shuffle_corr_mat.npz",
     "StyleGAN-Face*": "HessNetArchit\\StyleGAN\\Hess_StyleGAN_shuffle_corr_mat.npz",
     "StyleGAN2-Face512*": "HessNetArchit\\StyleGAN2\\Hess_StyleGAN2_Face512_shuffle_corr_mat.npz",
     # "StyleGAN2-Face256*": "StyleGAN2\\Hess_ffhq-256-config-e-003810_corr_mat.npz",
     # "StyleGAN2-Cat256*": "StyleGAN2\\Hess_stylegan2-cat-config-f_corr_mat.npz",
      "StyleGAN-Face_Z": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_fix_ctrl\\Hess_StyleGAN_Face256_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face512_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix_ctrl\\Hess_ffhq-512-avg-tpurun1_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face256_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_fix_ctrl\\Hess_ffhq-256-config-e-003810_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Cat256_Z": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_fix_ctrl\\Hess_stylegan2-cat-config-f_fix_ctrl_corr_mat.npz",
      "StyleGAN-Face_W": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_W_fix_ctrl\\Hess_StyleGAN_Face256_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face512_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix_ctrl\\Hess_ffhq-512-avg-tpurun1_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face256_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix_ctrl\\Hess_ffhq-256-config-e-003810_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Cat256_W": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix_ctrl\\Hess_stylegan2-cat-config-f_W_fix_ctrl_corr_mat"
                            ".npz",
                    }

ctrl_Havg_npz_dict = {"fc6GAN": "HessNetArchit\\fc6GAN\\H_avg_FC6GAN_shuffle_evol.npz",
     "DCGAN": "HessNetArchit\\DCGAN\\H_avg_DCGAN_shuffle.npz",
     "BigGAN": "HessNetArchit\\BigGAN\\H_avg_BigGAN_shuffle.npz",
    # "BigGAN_noise": "BigGAN\\H_avg_1000cls.npz",
    # "BigGAN_class": "BigGAN\\H_avg_1000cls.npz",
     "BigBiGAN": "HessNetArchit\\BigBiGAN\\H_avg_BigBiGAN_shuffle.npz",
     "PGGAN": "HessNetArchit\\PGGAN\\H_avg_PGGAN_shuffle.npz",
     "StyleGAN-Face*": "HessNetArchit\\StyleGAN\\H_avg_StyleGAN_shuffle.npz",
     # "HessNetArchit\\StyleGAN\\H_avg_StyleGAN_wspace_shuffle.npz"
     "StyleGAN2-Face512*": "HessNetArchit\\StyleGAN2\\H_avg_StyleGAN2_Face512_shuffle.npz",
     # "StyleGAN2-Face256*": "StyleGAN2\\H_avg_ffhq-256-config-e-003810.npz",
     # "StyleGAN2-Cat256*": "StyleGAN2\\H_avg_stylegan2-cat-config-f.npz",
      "StyleGAN-Face_Z": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_fix_ctrl\\H_avg_StyleGAN_Face256_fix_ctrl.npz",
      "StyleGAN2-Face512_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix_ctrl\\H_avg_ffhq-512-avg-tpurun1_fix_ctrl.npz",
      "StyleGAN2-Face256_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_fix_ctrl\\H_avg_ffhq-256-config-e-003810_fix_ctrl.npz",
      "StyleGAN2-Cat256_Z": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_fix_ctrl\\H_avg_stylegan2-cat-config-f_fix_ctrl.npz",
      "StyleGAN-Face_W": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_W_fix_ctrl\\H_avg_StyleGAN_Face256_W_fix_ctrl.npz",
      "StyleGAN2-Face512_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix_ctrl\\H_avg_ffhq-512-avg-tpurun1_W_fix_ctrl.npz",
      "StyleGAN2-Face256_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix_ctrl\\H_avg_ffhq-256-config-e-003810_W_fix_ctrl.npz",
      "StyleGAN2-Cat256_W": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix_ctrl\\H_avg_stylegan2-cat-config-f_W_fix_ctrl"
                            ".npz",
                    }

ctrl_spectra_npz_dict = {"fc6GAN": "HessNetArchit\\FC6GAN\\spectra_col_FC6GAN_shuffle_evol.npz",
          "DCGAN": "HessNetArchit\\DCGAN\\spectra_col_DCGAN_shuffle.npz",
          "BigGAN": "HessNetArchit\\BigGAN\\spectra_col_BigGAN_shuffle.npz",
          # "BigGAN_noise": "HessNetArchit\\BigGAN\\spectra_col_BigGAN_shuffle.npz",
          # "BigGAN_class": "HessNetArchit\\BigGAN\\spectra_col_BigGAN_shuffle.npz",
          "BigBiGAN": "HessNetArchit\\BigBiGAN\\spectra_col.npz",
          "PGGAN": "HessNetArchit\\PGGAN\\spectra_col_PGGAN_shuffle.npz",
          "StyleGAN-Face*": "HessNetArchit\\StyleGAN\\spectra_col_StyleGAN_shuffle.npz",
          "StyleGAN2-Face512*": "HessNetArchit\\StyleGAN2\\spectra_col_StyleGAN2_Face512_shuffle.npz",
          # "StyleGAN2-Face256*": "StyleGAN2\\spectra_col_ffhq-256-config-e-003810_BP.npz",
          # "StyleGAN2-Cat256*": "StyleGAN2\\spectra_col_stylegan2-cat-config-f_BP.npz",
          "StyleGAN-Face_Z": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_fix_ctrl\\spectra_col_StyleGAN_Face256_fix_ctrl.npz",
          "StyleGAN2-Face512_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix_ctrl\\spectra_col_ffhq-512-avg-tpurun1_fix_ctrl.npz",
          "StyleGAN2-Face256_Z": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_fix_ctrl\\spectra_col_ffhq-256-config-e-003810_fix_ctrl.npz",
          "StyleGAN2-Cat256_Z": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_fix_ctrl\\spectra_col_stylegan2-cat-config-f_fix_ctrl.npz",
          "StyleGAN-Face_W": "Hessian_summary\\StyleGAN_Fix\\StyleGAN_Face256_W_fix_ctrl\\spectra_col_StyleGAN_Face256_W_fix_ctrl.npz",
          "StyleGAN2-Face512_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix_ctrl\\spectra_col_ffhq-512-avg-tpurun1_W_fix_ctrl.npz",
          "StyleGAN2-Face256_W": "Hessian_summary\\StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix_ctrl\\spectra_col_ffhq-256-config-e-003810_W_fix_ctrl.npz",
          "StyleGAN2-Cat256_W": "Hessian_summary\\StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix_ctrl\\spectra_col_stylegan2-cat-config-f_W_fix_ctrl.npz",
          }
#%%
consist_tab = {}
for GAN, path in corrmat_npz_dict.items():
    # parts = path.split("\\")
    # folder, npzfn = join(*parts[:-1]), parts[-1]
    # npzname, _ = npzfn.split(".")
    data = np.load(join(summarydir, path))
    corr_mat_log, corr_mat_lin = data["corr_mat_log"].copy(), data["corr_mat_lin"].copy()
    np.fill_diagonal(corr_mat_lin, np.nan)
    np.fill_diagonal(corr_mat_log, np.nan)
    consist_tab[GAN] = (np.nanmean(corr_mat_log), np.nanstd(corr_mat_log), np.nanmean(corr_mat_lin), np.nanstd(corr_mat_lin))

consist_ctrl_tab = {}
for GAN, path in ctrl_corrmat_npz_dict.items():
    if path is None:
        print("%s GAN doesn't have a valid control"%GAN)
        continue
    data = np.load(join(summaryroot, path))
    corr_mat_log, corr_mat_lin = data["corr_mat_log"].copy(), data["corr_mat_lin"].copy()
    np.fill_diagonal(corr_mat_lin, np.nan)
    np.fill_diagonal(corr_mat_log, np.nan)
    consist_ctrl_tab[GAN] = (
    np.nanmean(corr_mat_log), np.nanstd(corr_mat_log), np.nanmean(corr_mat_lin), np.nanstd(corr_mat_lin))
#%%
consist_tab_cmb = {}
for GAN, ctrlpath in ctrl_corrmat_npz_dict.items():
    if GAN not in corrmat_npz_dict:
        print("%s GAN doesn't have a valid real data" % GAN)
        real_data = (None, None, None, None)
    else:
        realpath = corrmat_npz_dict[GAN]
        # Real GAN
        data = np.load(join(summarydir, realpath))
        corr_mat_log, corr_mat_lin = data["corr_mat_log"].copy(), data["corr_mat_lin"].copy()
        np.fill_diagonal(corr_mat_lin, np.nan)
        np.fill_diagonal(corr_mat_log, np.nan)
        real_data = (
            np.nanmean(corr_mat_log), np.nanstd(corr_mat_log), np.nanmean(corr_mat_lin), np.nanstd(corr_mat_lin))
    if ctrlpath is None:
        print("%s GAN doesn't have a valid control"%GAN)
        ctrl_data = (None, None, None, None)
    else:
        # Shuffled control GAN
        data = np.load(join(summaryroot, ctrlpath))
        corr_mat_log, corr_mat_lin = data["corr_mat_log"].copy(), data["corr_mat_lin"].copy()
        np.fill_diagonal(corr_mat_lin, np.nan)
        np.fill_diagonal(corr_mat_log, np.nan)
        ctrl_data = (
        np.nanmean(corr_mat_log), np.nanstd(corr_mat_log), np.nanmean(corr_mat_lin), np.nanstd(corr_mat_lin))
    consist_tab_cmb[GAN] = real_data + ctrl_data

GAN_consist_cmp_summary = pd.DataFrame.from_dict(consist_tab_cmb, orient="index", columns=["log_corr_mean", "log_corr_std", "lin_corr_mean",  "lin_corr_std", "log_corr_mean_ctrl", "log_corr_std_ctrl", "lin_corr_mean_ctrl",  "lin_corr_std_ctrl"])
#%%
tab = GAN_consist_cmp_summary
msk = ["*" not in name for name in tab.index]
fig, ax = plt.subplots(figsize=[9, 7])
plt.plot(tab[["lin_corr_mean","lin_corr_mean_ctrl"]][msk].T, tab[["log_corr_mean", "log_corr_mean_ctrl"]][msk].T, color="gray")
plt.scatter(tab[["lin_corr_mean"]][msk], tab[["log_corr_mean"]][msk], color="red")
plt.scatter(tab[["lin_corr_mean_ctrl"]][msk], tab[["log_corr_mean_ctrl"]][msk], color="blue")

plt.errorbar(tab["lin_corr_mean"][msk], tab["log_corr_mean"][msk], xerr=tab.lin_corr_std[msk], yerr=tab.log_corr_std[msk], color="red", fmt='none' )
plt.errorbar(tab["lin_corr_mean_ctrl"][msk], tab["log_corr_mean_ctrl"][msk], xerr=tab.lin_corr_std_ctrl[msk], yerr=tab.log_corr_std_ctrl[msk], color="blue", fmt='none')
for i, GAN in enumerate(tab[msk].index):
    ax.annotate(GAN, (tab[msk].lin_corr_mean[i], tab[msk].log_corr_mean[i]), fontsize=14)
plt.ylabel("log scale corr", fontsize=14)
plt.xlabel("lin scale corr", fontsize=14)
ax.set_aspect("equal")
plt.savefig(join(summarydir,"Hess_consisitency_shuffle_synopsis_cmp.png"))
plt.savefig(join(summarydir, "Hess_consisitency_shuffle_synopsis_cmp.pdf"))
plt.show()
tab.to_csv(join(summarydir, "Hess_consistency_shuffle_table.csv"))