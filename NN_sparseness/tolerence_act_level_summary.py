import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.visualize_sparse_inv_example import *
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
evoldir = join(rootdir, r"actlevel_tolerence_evol")
INetdir = join(rootdir, r"actlevel_tolerence")
sumdir  = join(rootdir, "summary")
figdir  = join(rootdir, "summary_figs")
netname = "resnet50_linf8"
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
#%%
from easydict import EasyDict as edict
# df = pd.read_csv(join(evoldir, "resnet50_linf8_layer4.B2_unit0245_evol_tolerance.png"))
def pearsonr_na(series1, series2):
    valmsk = ~(np.isnan(series1) | np.isnan(series2))
    N = valmsk.sum()
    if N <= 1:
        return np.nan, np.nan, N
    else:
        corr, pval = pearsonr(series1[valmsk], series2[valmsk])
        return corr, pval, N
#%%
df_col = []
for layer_long in feattsrs.keys():
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    for unit_id in range(0, 250, 5,):
        respvect = feattsrs[layer_long][:, unit_id]
        INet99, INet999, INetmax = torch.quantile(respvect, torch.tensor([0.99,0.999,1.0]))
        INet99, INet999, INetmax = INet99.item(), INet999.item(), INetmax.item()
        df_INet = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_INet_toler.csv"))
        df_evol = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"))
        # INet
        INetmsk = df_INet.centact < INet99
        corr_INet_Lw, pval_INet_Lw, N_INet_Lw = pearsonr_na(df_INet.obj_toler[INetmsk], df_INet.centact[INetmsk])
        corr_INet_Up, pval_INet_Up, N_INet_Up = pearsonr_na(df_INet.obj_toler[~INetmsk], df_INet.centact[~INetmsk])
        # Evol
        Evolmsk = df_evol.centact < INet99
        corr_Evol_Lw, pval_Evol_Lw, N_Evol_Lw = pearsonr_na(df_evol.obj_toler[Evolmsk], df_evol.centact[Evolmsk])
        corr_Evol_Up, pval_Evol_Up, N_Evol_Up = pearsonr_na(df_evol.obj_toler[~Evolmsk], df_evol.centact[~Evolmsk])
        stat = edict(layer_long=layer_long, layer_short=layer_short, unit_id=unit_id,
                     corr_INet_Lw=corr_INet_Lw, pval_INet_Lw=pval_INet_Lw, N_INet_Lw=N_INet_Lw,
                     corr_INet_Up=corr_INet_Up, pval_INet_Up=pval_INet_Up, N_INet_Up=N_INet_Up,
                     corr_Evol_Lw=corr_Evol_Lw, pval_Evol_Lw=pval_Evol_Lw, N_Evol_Lw=N_Evol_Lw,
                     corr_Evol_Up=corr_Evol_Up, pval_Evol_Up=pval_Evol_Up, N_Evol_Up=N_Evol_Up,
                     INet999=INet999, INetmax=INetmax, INet99=INet99,)
        df_col.append(stat)
df_all = pd.DataFrame(df_col)
#%%
# df_all.to_csv(join(sumdir, f"{netname}_actlevel_tolerance_summary_thr999.csv"), index=False)
df_all.to_csv(join(sumdir, f"{netname}_actlevel_tolerance_summary_thr99.csv"), index=False)
#%%
sfx = "_thr99"
df_all = pd.read_csv(join(sumdir, f"{netname}_actlevel_tolerance_summary{sfx}.csv"))
df_long = df_all.melt(id_vars=["layer_short", "unit_id"],
            value_vars=["corr_INet_Lw", "corr_INet_Up", "corr_Evol_Lw", "corr_Evol_Up",])
#%%

#%%
figh = plt.figure(figsize=(11, 6))
ax = sns.stripplot(x="layer_short", y="value", data=df_long, dodge=True, alpha=0.6,
       hue="variable", hue_order=["corr_Evol_Lw", "corr_INet_Lw", "corr_Evol_Up", "corr_INet_Up", ],)
ax = sns.boxplot(x="layer_short", y="value", data=df_long, dodge=True, saturation=.9,
       hue="variable", hue_order=["corr_Evol_Lw", "corr_INet_Lw", "corr_Evol_Up", "corr_INet_Up", ],)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.set(ylim=(-1.05, 1.05), ylabel="Correlation", xlabel="Layer",
       title="Correlation of Tolerance and Activation Level to Object")
ax.legend(loc='upper center', bbox_to_anchor=(1.20, 0.2),
          ncol=2, fancybox=False, shadow=False)
plt.tight_layout()
saveallforms(figdir, f"actlevel-toler_corr_w_thresh_Evol_INet{sfx}", figh, )
plt.show()
#%%
figh = plt.figure(figsize=(9.5, 6))
ax = sns.stripplot(x="layer_short", y="value", data=df_long, dodge=True, alpha=0.6,
       hue="variable", hue_order=["corr_Evol_Lw", "corr_INet_Lw", ],)
ax = sns.boxplot(x="layer_short", y="value", data=df_long, dodge=True, saturation=.9,
       hue="variable", hue_order=["corr_Evol_Lw", "corr_INet_Lw", ],)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.set(ylim=(-1.05, 1.05), ylabel="Correlation", xlabel="Layer",
       title="Correlation of Tolerance and Activation Level to Object")
ax.legend(loc='upper center', bbox_to_anchor=(1.20, 0.2),
          ncol=2, fancybox=False, shadow=False)
plt.tight_layout()
saveallforms(figdir, f"actlevel-toler_corr_w_thresh_Evol_INet{sfx}_lw", figh, )
plt.show()
#%%
figh = plt.figure(figsize=(9.5, 6))
ax = sns.stripplot(x="layer_short", y="value", data=df_long, dodge=True, alpha=0.6,
       hue="variable", hue_order=["corr_Evol_Up", "corr_INet_Up", ],)
ax = sns.boxplot(x="layer_short", y="value", data=df_long, dodge=True, saturation=.9,
       hue="variable", hue_order=["corr_Evol_Up", "corr_INet_Up", ],)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.set(ylim=(-1.05, 1.05), ylabel="Correlation", xlabel="Layer",
       title="Correlation of Tolerance and Activation Level to Object")
ax.legend(loc='upper center', bbox_to_anchor=(1.20, 0.2),
          ncol=2, fancybox=False, shadow=False)
plt.tight_layout()
saveallforms(figdir, f"actlevel-toler_corr_w_thresh_Evol_INet{sfx}_Up", figh, )
plt.show()
#%%
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
def fit_gpr(xseries, yseries, kernel=None):
    valmsk = ~np.isnan(xseries) & ~np.isnan(yseries)
    gspr = GaussianProcessRegressor(kernel=kernel)
    gspr.fit(np.array(xseries[valmsk]).reshape([-1, 1]),
             np.array(yseries[valmsk]).reshape([-1, 1]))
    xbins = np.linspace(0, np.nanmax(xseries), 100)
    mean_pred_gpr, std_pred_gpr = gspr.predict(xbins[:, None], return_std=True, )
    return gspr, xbins, mean_pred_gpr, std_pred_gpr

sfx = "_thr999"
for layer_long in feattsrs.keys():
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    fig_merg, ax_merg = plt.subplots(figsize=(8, 8))
    for unit_id in range(0, 250, 5,):
        respvect = feattsrs[layer_long][:, unit_id]
        INet99, INet999, INetmax = torch.quantile(respvect, torch.tensor([0.99,0.999,1.0]))
        INet99, INet999, INetmax = INet99.item(), INet999.item(), INetmax.item()
        df_INet = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_INet_toler.csv"))
        df_evol = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"))
        df_cmb = pd.concat((df_evol, df_INet))
        gspr_INet, xbins_INet, mean_pred_INet, std_pred_INet = fit_gpr(df_INet.centact, df_INet.obj_toler, )
        gspr_evol, xbins_evol, mean_pred_evol, std_pred_evol = fit_gpr(df_evol.centact, df_evol.obj_toler, )
        gspr_cmb, xbins_cmb, mean_pred_cmb, std_pred_cmb = fit_gpr(df_cmb.centact, df_cmb.obj_toler, )
        #
        # plt.figure(figsize=(6,6))
        # plt.scatter(df_INet.centact, df_INet.obj_toler,  label="ImageNet", alpha=0.5)
        # plt.scatter(df_evol.centact, df_evol.obj_toler,  label="GAN generated", alpha=0.5)
        # plt.plot(xbins, mean_pred_gpr, label="GPR fit ImageNet")
        # plt.plot(xbins, mean_pred_gpr_evol, label="GPR fit GAN generated")
        # plt.plot(xbins_cmb, mean_pred_gpr_cmb, label="GPR fit ImageNet+GAN", color="k", linestyle="--")
        # # plt.fill_between(xbins, mean_pred_gpr[:,0] - std_pred_gpr,
        # #                  mean_pred_gpr[:,0] + std_pred_gpr, alpha=0.4)
        # plt.vlines(INet99, 0, 1, label="99%", color="blue")
        # plt.vlines(INet999, 0, 1, label="99.9%", color="green")
        # plt.vlines(INetmax, 0, 1, label="max", color="red")
        # plt.legend()
        # plt.show()

        ax_merg.plot(xbins_INet / INet999, mean_pred_INet, color="red", linestyle="-")
        ax_merg.plot(xbins_evol / INet999, mean_pred_evol, color="blue", linestyle="-")
        # ax_merg.plot(xbins_cmb / INet999, mean_pred_cmb, color="k", linestyle="-", alpha=0.5)
    ax_merg.set_ylim(-0.05, 1.05)
    fig_merg.savefig(join(figdir, f"{netname}_{layer_short}_toler_corr_w_thresh_Evol_INet{sfx}_norm.png"))
    fig_merg.show()
    # raise Exception("stop")
    # plt.show()
#         # INet
#         INetmsk = df_INet.centact < INet99
#         corr_INet_Lw, pval_INet_Lw, N_INet_Lw = pearsonr_na(df_INet.obj_toler[INetmsk], df_INet.centact[INetmsk])
#         corr_INet_Up, pval_INet_Up, N_INet_Up = pearsonr_na(df_INet.obj_toler[~INetmsk], df_INet.centact[~INetmsk])
#         # Evol
#         Evolmsk = df_evol.centact < INet99
#         corr_Evol_Lw, pval_Evol_Lw, N_Evol_Lw = pearsonr_na(df_evol.obj_toler[Evolmsk], df_evol.centact[Evolmsk])
#         corr_Evol_Up, pval_Evol_Up, N_Evol_Up = pearsonr_na(df_evol.obj_toler[~Evolmsk], df_evol.centact[~Evolmsk])
#         stat = edict(layer_long=layer_long, layer_short=layer_short, unit_id=unit_id,
#                      corr_INet_Lw=corr_INet_Lw, pval_INet_Lw=pval_INet_Lw, N_INet_Lw=N_INet_Lw,
#                      corr_INet_Up=corr_INet_Up, pval_INet_Up=pval_INet_Up, N_INet_Up=N_INet_Up,
#                      corr_Evol_Lw=corr_Evol_Lw, pval_Evol_Lw=pval_Evol_Lw, N_Evol_Lw=N_Evol_Lw,
#                      corr_Evol_Up=corr_Evol_Up, pval_Evol_Up=pval_Evol_Up, N_Evol_Up=N_Evol_Up,
#                      INet999=INet999, INetmax=INetmax, INet99=INet99,)
#         df_col.append(stat)
# df_all = pd.DataFrame(df_col)