from os.path import join
import pandas as pd
import numpy as np
from Hessian.hessian_analysis_tools import scan_hess_npz
import matplotlib.pylab as plt

summarydir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary"
def spectra_montage(GANlist, fnlist, ylog=True, xnorm=False, ynorm=True, shade=True, xlim=(-25, 525), ylim=None,\
                                             lw=1, fn="spectra_synopsis_log_rank"):
    fig = plt.figure(figsize=[5,4.5])
    for i, GAN in enumerate(GANlist):
        with np.load(join(rootdir, fnlist[i])) as data:
            eigval_col = data["eigval_col"]
        if eigval_col[:, -1].mean() > eigval_col[:, 0].mean():
            eigval_col = eigval_col[:, ::-1]
        eva_mean = eigval_col.mean(axis=0)
        eva_lim = np.percentile(eigval_col, [5, 95], axis=0)
        # eva_lim_pos = np.maximum(eva_lim, cutoffR * eva_mean.max())
        eva_lim_pos = eva_lim.copy()
        if ylog:
            negmask = eva_lim_pos[0, :] < 0
            eva_lim_pos[0, negmask] = eva_mean[negmask]
        xnormalizer = len(eva_mean) if xnorm else 1
        ynormalizer = eva_mean.max() if ynorm else 1
        ytfm = np.log10 if ylog else lambda x: x
        plt.plot(np.arange(len(eva_mean))/xnormalizer, ytfm(eva_mean / ynormalizer), alpha=.8, lw=lw, label=GAN)  # ,
        # eigval_arr.std(axis=0)
        if shade:
            plt.fill_between(np.arange(len(eva_mean))/xnormalizer, ytfm(eva_lim_pos[0, :] / ynormalizer),
                         ytfm(eva_lim_pos[1, :] / ynormalizer), alpha=0.1)
    plt.ylabel("log10(eig/eigmax)" if ylog else "eig/eigmax")
    plt.xlabel("rank normalized to latent dim" if xnorm else "ranks")
    plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.title("Spectra Compared Across GANs")
    plt.legend(loc="best")
    plt.savefig(join(rootdir, fn+".png"))
    plt.savefig(join(rootdir, fn+".pdf"))
    plt.show()
    return fig

#%%
FC6figdir = r"E:\OneDrive - Washington University in St. " \
            r"Louis\Hessian_summary\fc6GAN" #r"E:\Cluster_Backup\FC6GAN\summary"
FC6dir = r"E:\Cluster_Backup\FC6GAN"
eva_col, _, feat_arr, meta = scan_hess_npz(FC6dir, "evol_(\d*)_bpfull.npz", featkey='code', evakey='eigvals',
                                                     evckey=None)#'eigvects')

#%%

eigvals_col = np.array(eigvals_col)[:, ::-1]
code_all = np.array(code_all)
np.savez(join(FC6figdir, "spectra_col_evol.npz"), eigval_col=eigvals_col, )
#%%
"""Visualize the spectra of different GANs all in one place"""
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
          # "StyleGAN2\\spectra_col_stylegan2-cat-config-f.npz",
          # "StyleGAN2\\spectra_col_ffhq-512-avg-tpurun1_Forwa.npz",
          # "StyleGAN2\\spectra_col_stylegan2-cat-config-f_Forwa.npz"]

#%%
fig1 = spectra_montage(GANlist, fnlist, xlim=(-25, 525), fn="spectra_synopsis_log_rank2")
fig2 = spectra_montage(GANlist, fnlist, xlim=(-25, 4125), fn="spectra_synopsis_log_rank_full2")
fig3 = spectra_montage(GANlist, fnlist, xlim=(-25, 515), shade=False, fn="spectra_synopsis_log_rank_line")
#%%
GANlist_Conv = [GANlist[i] for i in [1,2,3,4]]
fnlist_Conv = [fnlist[i] for i in [1,2,3,4]]
GANlist_Style = [GANlist[i] for i in [5,6,7,8]]
fnlist_Style = [fnlist[i] for i in [5,6,7,8]]
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 140), lw=2, fn="spectra_synopsis_log_rank_Conv")
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 520), lw=2, fn="spectra_synopsis_log_rank_Conv_Full")
fig5 = spectra_montage(GANlist_Style, fnlist_Style, xlim=(-5, 140), lw=2, fn="spectra_synopsis_log_rank_Style")
#%%
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
spaceD = [4096, 120, 256, 120, 512, 512, 512, 512, 512, 512, 512, 512, 512]
GANlist = np.array(["FC6", "DCGAN-fashion", "BigGAN", "BigBiGAN", "PGGAN-face", "StyleGAN-face_Z",
           "StyleGAN2-face512_Z", "StyleGAN2-face256_Z", "StyleGAN2-cat_Z", "StyleGAN-face_W",
           "StyleGAN2-face512_W", "StyleGAN2-face256_W", "StyleGAN2-cat_W", ])
           # "StyleGAN-face-Forw", "StyleGAN-cat-Forw"]
fnlist = np.array(["FC6GAN\\spectra_col_evol.npz",
          "DCGAN\\spectra_col_BP.npz",
          "BigGAN\\spectra_col.npz",
          "BigBiGAN\\spectra_col.npz",
          "PGGAN\\spectra_col_BP.npz",
          # "StyleGAN\\spectra_col_face256_BP.npz",  # need to update!!!!
          # "StyleGAN_wspace\\spectra_col_StyleGAN_Wspace.npz",
          "StyleGAN_Fix\\StyleGAN_Face256_fix\\spectra_col_StyleGAN_Face256_fix.npz",
          "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_fix\\spectra_col_ffhq-512-avg-tpurun1_fix.npz",
          "StyleGAN2_Fix\\ffhq-256-config-e-003810_fix\\spectra_col_ffhq-256-config-e-003810_fix.npz",
          "StyleGAN2_Fix\\stylegan2-cat-config-f_fix\\spectra_col_stylegan2-cat-config-f_fix.npz",
          "StyleGAN_Fix\\StyleGAN_Face256_W_fix\\spectra_col_StyleGAN_Face256_W_fix.npz",
          "StyleGAN2_Fix\\ffhq-512-avg-tpurun1_W_fix\\spectra_col_ffhq-512-avg-tpurun1_W_fix.npz",
          "StyleGAN2_Fix\\ffhq-256-config-e-003810_W_fix\\spectra_col_ffhq-256-config-e-003810_W_fix.npz",
          "StyleGAN2_Fix\\stylegan2-cat-config-f_W_fix\\spectra_col_stylegan2-cat-config-f_W_fix.npz", ])
#%% ffhq-256-config-e-003810_BP.npz",
fig1 = spectra_montage(GANlist[:], fnlist[:], xlim=(0, 500), lw=2.5, fn="spectra_synopsis_SGfix_all")
fig2 = spectra_montage(GANlist[:], fnlist[:], xlim=(0, 4100), lw=2.5, fn="spectra_synopsis_full_SGfix_all")
fig3 = spectra_montage(GANlist[:], fnlist[:], xlim=(0, 500), lw=2.5, shade=False,
                       fn="spectra_synopsis_line_SGfix_all")
fig1 = spectra_montage(GANlist[:9], fnlist[:9], xlim=(-25, 525), lw=2.5, fn="spectra_synopsis_SGfix_Z")
fig2 = spectra_montage(GANlist[:9], fnlist[:9], xlim=(-25, 4125), lw=2.5, fn="spectra_synopsis_full_SGfix_Z")
fig3 = spectra_montage(GANlist[:9], fnlist[:9], xlim=(-25, 515), lw=2.5, shade=False,
                       fn="spectra_synopsis_line_SGfix_Z")
#%%
idxs = [0,1,2,3,4,5,6,9,10]
fig1 = spectra_montage(GANlist[idxs], fnlist[idxs], xlim=(0, 500), lw=2.5,
                       fn="spectra_synopsis_SGfix_sel")
fig2 = spectra_montage(GANlist[idxs], fnlist[idxs], xlim=(0, 4100), lw=2.5, fn="spectra_synopsis_full_SGfix_sel")
fig3 = spectra_montage(GANlist[idxs], fnlist[idxs], xlim=(0, 500), lw=2.5, shade=False,
                       fn="spectra_synopsis_line_SGfix_sel")
#%%
fig5 = spectra_montage(GANlist[5:13], fnlist[5:13], xlim=(-5, 140), ylim=(-9,0.5),shade=True, lw=2.5,
                       fn="spectra_synopsis_Style_SGfix")
fig6 = spectra_montage(GANlist[1:5], fnlist[1:5], xlim=(-5, 140), shade=True, lw=2.5,
                       fn="spectra_synopsis_DCGAN_SGfix")
#%%
GANlist_Conv = [GANlist[i] for i in [1,2,3,4]]
fnlist_Conv = [fnlist[i] for i in [1,2,3,4]]
GANlist_Style = [GANlist[i] for i in range(5,13)]
fnlist_Style = [fnlist[i] for i in range(5,13)]
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 140), shade=True, lw=2.5,
                       fn="spectra_synopsis_log_rank_Conv_SGfix")
fig4 = spectra_montage(GANlist_Conv, fnlist_Conv, xlim=(-5, 520), shade=True, lw=2.5,
                       fn="spectra_synopsis_log_rank_Conv_Full_SGfix")
fig5 = spectra_montage(GANlist_Style, fnlist_Style, xlim=(-5, 140), shade=True, lw=2.5,
                       fn="spectra_synopsis_log_rank_Style_SGfix")
#%%
cutoff_tab = {}
for GAN, npzpath in zip(GANlist, fnlist):
    # folder, npzfn = npzpath.split("\\")
    # npzname, _ = npzfn.split(".")
    data = np.load(join(summarydir, npzpath))
    eig_mean = np.mean(data['eigval_col'], axis=0)
    eigabssort = np.sort(np.abs(eig_mean))[::-1]
    expvar = np.cumsum(eigabssort)/eigabssort.sum()
    dimen = len(eig_mean)
    cutoff_nums = (dimen, sum(expvar<0.99), sum(expvar<0.999), sum(expvar<0.9999), sum(expvar<0.99999), )
    cutoff_tab[GAN] = cutoff_nums
    print("%s (%d D) %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var"%
          (GAN, *cutoff_nums))
    if GAN == "BigGAN":
        eig_mean = np.mean(data["eigvals_nois_col"], axis=0)
        eigabssort = np.sort(np.abs(eig_mean))[::-1]
        expvar = np.cumsum(eigabssort) / eigabssort.sum()
        dimen = len(eig_mean)
        cutoff_nums = (dimen, sum(expvar < 0.99), sum(expvar < 0.999), sum(expvar < 0.9999), sum(expvar < 0.99999),)
        cutoff_tab[GAN+"_noise"] = cutoff_nums
        print("%s (%d D) %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var" %
              (GAN+"_noise", *cutoff_nums))
        eig_mean = np.mean(data["eigvals_clas_col"], axis=0)
        eigabssort = np.sort(np.abs(eig_mean))[::-1]
        expvar = np.cumsum(eigabssort) / eigabssort.sum()
        dimen = len(eig_mean)
        cutoff_nums = (dimen, sum(expvar < 0.99), sum(expvar < 0.999), sum(expvar < 0.9999), sum(expvar < 0.99999),)
        cutoff_tab[GAN + "_class"] = cutoff_nums
        print("%s (%d D) %d dim for 0.99 var, %d dim for 0.999 var, %d dim for 0.9999 var, %d dim for 0.99999 var" %
              (GAN + "_class", *cutoff_nums))
GAN_cutoff_summary = pd.DataFrame.from_dict(cutoff_tab, orient="index", columns=["dimen", "dim99", "dim999",
                                          "dim9999",  "dim9999"])
GAN_cutoff_summary.to_csv(join(summarydir, "Hess_anistropy_table.csv"))
# GAN_geom_summary.to_csv(join(summarydir, "Hess_consistency_table.csv"))
# GAN_summary = pd.concat((GAN_geom_summary, GAN_cutoff_summary, ), axis=1)
#%%
# data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\spectra_col.npz")
#$$



