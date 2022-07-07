import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.visualize_sparse_inv_example import *

def pearsonr_na(series1, series2):
    valmsk = ~(np.isnan(series1) | np.isnan(series2))
    N = valmsk.sum()
    if N <= 1:
        return np.nan, np.nan, N
    else:
        corr, pval = pearsonr(series1[valmsk], series2[valmsk])
        return corr, pval, N


rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
evoldir = join(rootdir, r"actlevel_tolerence_evol")
INetdir = join(rootdir, r"actlevel_tolerence")
sumdir  = join(rootdir, "summary")
figdir  = join(rootdir, "summary_figs")
# matplotlib get default color cycle
cycClrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
netname = "resnet50_linf8"
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
#%%
# df = pd.read_csv(join(evoldir, "resnet50_linf8_layer4.B2_unit0245_evol_tolerance.png"))

#%% Coompute the correlation data as summary
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

#%% Reload and plot the summary
sfx = "_thr99"
df_all = pd.read_csv(join(sumdir, f"{netname}_actlevel_tolerance_summary{sfx}.csv"))
df_long = df_all.melt(id_vars=["layer_short", "unit_id"],
            value_vars=["corr_INet_Lw", "corr_INet_Up", "corr_Evol_Lw", "corr_Evol_Up",])
df_long["imagespace"] = df_long.variable.apply(lambda x: "ImageNet" if "INet" in x else "Evol")
df_long["range"] = df_long.variable.apply(lambda x: "Up" if "Up" in x else "Low")
#%% pooled plot
figh = plt.figure(figsize=(4.5, 6))
# ax = sns.boxplot(x="imagespace", y="value", data=df_long, dodge=True, saturation=.5,
#        hue="range", order=["ImageNet", "Evol", ],)
ax = sns.stripplot(x="imagespace", y="value", data=df_long, dodge=True, alpha=0.35,
       hue="range", order=["ImageNet", "Evol", ],)
ax = sns.violinplot(x="imagespace", y="value", data=df_long, dodge=True, saturation=.5,
       hue="range", order=["ImageNet", "Evol", ], inner="box", cut=0)
plt.legend()
ax.set(ylabel="Pearson correlation", title=f"Correlation of activation and tolerance\nat different range with threshold {sfx}")
plt.tight_layout()
saveallforms(figdir, f"actlevel-toler_corr_w_thresh_Evol_INet_layerpool{sfx}", figh, )
plt.show()

#%%
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
tval, pval = ttest_1samp(df_all['corr_INet_Lw'], 0)
print("corr ImageNet Lower (<99%%ile) %.3f t=%.3f (P=%.1e)" % (df_all['corr_INet_Lw'].mean(), tval, pval))
tval, pval = ttest_1samp(df_all['corr_INet_Up'], 0)
print("corr ImageNet Upper (>99%%ile) %.3f t=%.3f (P=%.1e)" % (df_all['corr_INet_Up'].mean(), tval, pval))
tval, pval = ttest_1samp(df_all['corr_Evol_Lw'].dropna(), 0)
print("corr Evol Lower (<99%%ile) %.3f t=%.3f (P=%.1e)" % (df_all['corr_Evol_Lw'].mean(), tval, pval))
tval, pval = ttest_1samp(df_all['corr_Evol_Up'].dropna(), 0)
print("corr Evol Upper (>99%%ile) %.3f t=%.3f (P=%.1e)" % (df_all['corr_Evol_Up'].mean(), tval, pval))

#%%
tval, pval = ttest_rel(df_all['corr_INet_Lw'], df_all['corr_INet_Up'])
print("corr ImageNet Lower (<99%%ile) vs. Upper (>99%%ile) %.3f ~ %.3f t=%.3f (P=%.1e)" %
      (df_all['corr_INet_Lw'].mean(), df_all['corr_INet_Up'].mean(), tval, pval))
msk = ~df_all['corr_Evol_Up'].isna()
tval, pval = ttest_rel(df_all['corr_Evol_Lw'][msk], df_all['corr_Evol_Up'][msk])
print("corr Evol Lower (<99%%ile) vs. Upper (>99%%ile) %.3f ~ %.3f t=%.3f (P=%.1e)" %
      (df_all['corr_Evol_Lw'].mean(), df_all['corr_Evol_Up'].mean(), tval, pval))

#%%
df_long.groupby("variable").agg({"value": ["mean", "sem", "count"]})
#%%
figh = plt.figure(figsize=(11, 6))
ax = sns.stripplot(x="layer_short", y="value", data=df_long, dodge=True, alpha=0.6,
       hue="variable", hue_order=["corr_Evol_Lw", "corr_INet_Lw", "corr_Evol_Up", "corr_INet_Up", ],)
ax = sns.boxplot(x="layer_short", y="value", data=df_long, dodge=True, saturation=.6,
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
       title=f"Correlation of Tolerance and Activation Level to Object {sfx}")
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
       title=f"Correlation of Tolerance and Activation Level to Object {sfx}")
ax.legend(loc='upper center', bbox_to_anchor=(1.20, 0.2),
          ncol=2, fancybox=False, shadow=False)
plt.tight_layout()
saveallforms(figdir, f"actlevel-toler_corr_w_thresh_Evol_INet{sfx}_Up", figh, )
plt.show()

#%%
"""
Use Gaussian Process regressor
plot and show smooth curves underlying the noisy relationship
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def fit_gpr(xseries, yseries, kernel=None):
    valmsk = ~np.isnan(xseries) & ~np.isnan(yseries)
    gspr = GaussianProcessRegressor(kernel=kernel)
    gspr.fit(np.array(xseries[valmsk]).reshape([-1, 1]),
             np.array(yseries[valmsk]).reshape([-1, 1]))
    xbins = np.linspace(0, np.nanmax(xseries), 100)
    mean_pred_gpr, std_pred_gpr = gspr.predict(xbins[:, None], return_std=True, )
    return gspr, xbins, mean_pred_gpr, std_pred_gpr
#%%
"""
Main plot montaging the traces from all layers and all units
"""
# gaussker = RBF(length_scale=1, length_scale_bounds=(1e-2, 10)) + \
#            WhiteKernel(noise_level=5E-2, noise_level_bounds=(1e-2, 1))  # noise level, old version
gaussker = RBF(length_scale=.5, length_scale_bounds=(1e-2, 1)) + \
           WhiteKernel(noise_level=5E-2, noise_level_bounds=(1e-2, .5))  # new version
sfx = "_thr99"
figh, axs = plt.subplots(1, len(feattsrs)-1, figsize=(24, 4.5))
for li, layer_long in enumerate(feattsrs.keys()):
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    fig_merg, ax_merg = plt.subplots(figsize=(6, 6))
    for unit_id in range(0, 250, 5,):
        respvect = feattsrs[layer_long][:, unit_id]
        INet99, INet999, INetmax = torch.quantile(respvect, torch.tensor([0.99,0.999,1.0]))
        INet99, INet999, INetmax = INet99.item(), INet999.item(), INetmax.item()
        df_INet = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_INet_toler.csv"))
        df_evol = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"))
        df_cmb = pd.concat((df_evol, df_INet))
        # July 6th edited, normalize the activation before fitting, so the GPR length scale is more constant.
        thresh = INet99 if sfx == "_thr99" else INet999
        gspr_INet, xbins_INet, mean_pred_INet, std_pred_INet = fit_gpr(df_INet.centact / thresh, df_INet.obj_toler, kernel=gaussker)
        gspr_evol, xbins_evol, mean_pred_evol, std_pred_evol = fit_gpr(df_evol.centact / thresh, df_evol.obj_toler, kernel=gaussker)
        gspr_cmb, xbins_cmb, mean_pred_cmb, std_pred_cmb = fit_gpr(df_cmb.centact / thresh, df_cmb.obj_toler, kernel=gaussker)
        #
        ax_merg.plot(xbins_evol, mean_pred_evol, color=cycClrs[0], linestyle="-", alpha=0.35)
        ax_merg.plot(xbins_INet, mean_pred_INet, color=cycClrs[1], linestyle="-", alpha=0.35)

        axs[li].plot(xbins_evol, mean_pred_evol, color=cycClrs[0], linestyle="-", alpha=0.35)
        axs[li].plot(xbins_INet, mean_pred_INet, color=cycClrs[1], linestyle="-", alpha=0.35)
        # ax_merg.plot(xbins_cmb / INet999, mean_pred_cmb, color="k", linestyle="-", alpha=0.5)
    ax_merg.set_ylim(-0.05, 1.05)
    ax_merg.set(xlabel=f"Activation (normed by {sfx}ile)", ylabel="Object Tolerance", title=f"{netname} {layer_short}")
    # fig_merg.savefig(join(figdir, f"{netname}_{layer_short}_toler_corr_w_thresh_Evol_INet{sfx}_norm.png"))
    saveallforms(figdir, f"{netname}_{layer_short}_actlevel-object_tolerance_w_thresh_Evol_INet{sfx}_norm_kermod", fig_merg, )
    fig_merg.show()
    # raise Exception("stop")
    # plt.show()
#%%
for li, layer_long in enumerate(feattsrs.keys()):
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    axs[li].set_ylim(-0.05, 1.05)
    axs[li].set_title(f"{layer_short}")

axs[3].set_xlabel(f"Activation (normed by {sfx}ile)")
axs[0].set_ylabel("Object Tolerance")
figh.suptitle(f"{netname} all layers activation ~ object tolerance relationship", fontsize=14)
figh.tight_layout()
saveallforms(figdir, f"{netname}_all_layers_actlevel-object_tolerance_w_thresh_Evol_INet{sfx}_norm_kermod", figh, )
#%%
for li, layer_long in enumerate(feattsrs.keys()):
    if "fc" in layer_long: continue
    axs[li].set_xlim(-0.2, 5.0)
    axs[li].vlines(1.0, -0.05, 1.05, linestyles="-.", color="k")
saveallforms(figdir, f"{netname}_all_layers_actlevel-object_tolerance_w_thresh_Evol_INet{sfx}_norm_xlim_kermod", figh, )
figh.show()


#%%
"""Plot single channel / unit as demo"""

gaussker = RBF(length_scale=.5, length_scale_bounds=(1e-2, 1)) + \
           WhiteKernel(noise_level=5E-2, noise_level_bounds=(1e-2, .5))  # noise level
# netname
layer_long = '.layer3.Bottleneck4'
unit_id =  35
sfx = "_thr99"
outdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Figure7Tolerance\source"

layer_short = shorten_layername(layer_long)
respvect = feattsrs[layer_long][:, unit_id]
INet99, INet999, INetmax = torch.quantile(respvect, torch.tensor([0.99,0.999,1.0]))
INet99, INet999, INetmax = INet99.item(), INet999.item(), INetmax.item()
df_INet = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_INet_toler.csv"))
df_evol = pd.read_csv(join(evoldir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"))
df_cmb = pd.concat((df_evol, df_INet))
gspr_INet, xbins_INet, mean_pred_INet, std_pred_INet = fit_gpr(df_INet.centact, df_INet.obj_toler, kernel=gaussker)
gspr_evol, xbins_evol, mean_pred_evol, std_pred_evol = fit_gpr(df_evol.centact, df_evol.obj_toler, kernel=gaussker)
gspr_cmb, xbins_cmb, mean_pred_cmb, std_pred_cmb = fit_gpr(df_cmb.centact, df_cmb.obj_toler, kernel=gaussker)
thresh = 1.0

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(xbins_evol / thresh, mean_pred_evol, color=cycClrs[0], linestyle="-", alpha=0.8, lw=1.5)
ax.plot(xbins_INet / thresh, mean_pred_INet, color=cycClrs[1], linestyle="-", alpha=0.8, lw=1.5)
ax.scatter(df_evol.centact, df_evol.obj_toler, color=cycClrs[0],  alpha=0.35) # linestyle="-",
ax.scatter(df_INet.centact, df_INet.obj_toler, color=cycClrs[1],  alpha=0.35) # linestyle="-",
YLIM = (-0.05, 1.05)
ax.set_ylim(*YLIM)
ax.vlines(INetmax, YLIM[0], YLIM[1], color="r", label="ImageNet max")
ax.vlines(INet99, YLIM[0], YLIM[1], color="b", label="ImageNet 99%ile")
ax.vlines(INet999, YLIM[0], YLIM[1], color="g", label="ImageNet 99.9%ile")
plt.legend()
ax.set(xlabel=f"Activation (normed by {sfx}ile)", ylabel="Object Tolerance",
       title=f"{netname} {layer_short} unit {unit_id}")
# fig_merg.savefig(join(figdir, f"{netname}_{layer_short}_toler_corr_w_thresh_Evol_INet{sfx}_norm.png"))
saveallforms(outdir, f"{netname}_{layer_short}_unit{unit_id:04d}_actlevel-toler{sfx}", fig, )
fig.show()