#%%
from os.path import join
import pandas as pd
import numpy as np
from Hessian.hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_consistency_example
from Hessian.hessian_axis_visualize import vis_eigen_action_row, vis_eigen_explore_row
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper
import matplotlib.pylab as plt

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
    print("%s %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var, total %d"%
          (GAN, *cutoff_nums))
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
    print("%s logscale corr %.3f(%.3f) linscale corr %.3f(%.3f)" %
          (GAN, *consist_tab[GAN]))

cutoff_tab = {}
for GAN, npzpath in spectra_npz_dict.items():
    # parts = path.split("\\")
    # folder, npzfn = join(*parts[:-1]), parts[-1]
    data = np.load(join(summarydir, npzpath))
    evakey = "eigval_col"
    if GAN == "BigGAN_class": evakey = "eigvals_clas_col"
    if GAN == "BigGAN_noise": evakey = "eigvals_nois_col"
    eig_mean = np.mean(data[evakey], axis=0)
    eigabssort = np.sort(np.abs(eig_mean))[::-1]
    expvar = np.cumsum(eigabssort)/eigabssort.sum()
    dimen = len(eig_mean)
    cutoff_nums = (sum(expvar<0.99), sum(expvar<0.999), sum(expvar<0.9999), sum(expvar<0.99999), dimen)
    cutoff_tab[GAN] = cutoff_nums
    print("%s %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var, total %d"%
          (GAN, *cutoff_nums))

GAN_geom_summary = pd.DataFrame.from_dict(consist_tab, orient="index", columns=["log_corr_mean", "log_corr_std",
                                          "lin_corr_mean",  "lin_corr_std"])
GAN_geom_summary.to_csv(join(summarydir, "Hess_consistency_table.csv"))
GAN_cutoff_summary = pd.DataFrame.from_dict(cutoff_tab, orient="index", columns=["dim99", "dim999",
                                          "dim9999",  "dim9999", "dimen"])
GAN_summary = pd.concat((GAN_geom_summary, GAN_cutoff_summary, ), axis=1)
GAN_summary.to_csv(join(summarydir, "Hess_summary_table.csv"))
#%% Redo the Synopsis of Hessian consistency plot
msk =[("*" not in GAN) and (GAN!='BigGAN_noise') and (GAN!='BigGAN_class') for GAN in GAN_geom_summary.index]
fig, ax = plt.subplots(figsize=[5,4])
plt.errorbar(GAN_geom_summary[msk].lin_corr_mean, GAN_geom_summary[msk].log_corr_mean, yerr=GAN_geom_summary[msk].log_corr_std,
             xerr=GAN_geom_summary[msk].lin_corr_std, fmt='o')
# plt.xlim([0,1])
for i, GAN in enumerate(GAN_geom_summary[msk].index):
    ax.annotate(GAN, (GAN_geom_summary[msk].lin_corr_mean[i], GAN_geom_summary[msk].log_corr_mean[i]), fontsize=13)
plt.ylabel("log scale corr", fontsize=14)
plt.xlabel("lin scale corr", fontsize=14)
plt.box(True)
plt.xlim([0.2, 1])
plt.savefig(join(summarydir, "Hess_corrmat_synopsis_v2.png"))
plt.savefig(join(summarydir, "Hess_corrmat_synopsis_v2.pdf"))
plt.show()
#%%