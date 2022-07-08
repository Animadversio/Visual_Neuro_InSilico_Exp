import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from glob import glob
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.sparse_invariance_lib import *
# Invariance_dataset, shorten_layername,\
#         calculate_sparseness, calculate_invariance, \
#         corrcoef_batch, mask_diagonal
#%%
outdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"

#%% Robust
"""
Experiment on ResNet50-robust
Record feature distribution and compute their sparseness
"""
# Load network
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
netname = "resnet50_linf8"
reclayers = [".layer1.Bottleneck1",
             ".layer2.Bottleneck0",
             ".layer2.Bottleneck3",
             ".layer3.Bottleneck2",
             ".layer3.Bottleneck4",
             ".layer4.Bottleneck0",
             ".layer4.Bottleneck2",
             ".Linearfc"]
#%% Process images and record feature vectors
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
#%% record full imagenet valid
dataset = create_imagenet_valid_dataset()
feattsrs = record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=125, num_workers=8)
torch.save(feattsrs, join(outdir, f"{netname}_INvalid_feattsrs.pt"))
#%% Invariance measure
Inv_data = Invariance_dataset()
Invfeatdata = record_dataset(model, reclayers, Inv_data, return_input=False,
                   batch_size=125, num_workers=8)
torch.save(Invfeatdata, join(outdir, f"{netname}_invariance_feattsrs.pt"))
#%% Reload feattsrs
Invfeatdata = torch.load(join(outdir, f"{netname}_invariance_feattsrs.pt"))
feattsrs = torch.load(join(outdir, f"{netname}_INvalid_feattsrs.pt"))
#%% Calculate all statistics
df_inv_all, df_inv_all_pop = calculate_invariance(Invfeatdata, layeralias=shorten_layername)
df_sprs_all, _ = calculate_sparseness(feattsrs, layeralias=shorten_layername)
df_prct_all = calculate_percentile(feattsrs, Invfeatdata, layeralias=shorten_layername)
#%% Merge into one dataframe
df_merge_all = pd.merge(df_inv_all, df_sprs_all,
                        on=['layer', 'layer_s', 'unitid',])
df_merge_all = pd.merge(df_merge_all, df_prct_all,
                        left_on=['layer', 'layer_s', 'unitid',],
                        right_on=['layer_x', 'layer_s', 'unitid',])
#%%
df_merge_all["layer_depth"] = df_merge_all.layer_x.apply(lambda x: reclayers.index(x))
df_merge_all.to_csv(join(sumdir, f"{netname}_sparse_invar_prctl_merge.csv"))
#%%
"""Print the correlation across layers"""
df_merge_all[["sparseness","unit_inv",
               "inv_resp_norm_mean",
               "inv_resp_norm_std",
               "inv_resp_norm_max",
               "top100_resp"]].corr(method="spearman")
#%%
df_merge_all.groupby("layer_s",sort=False)[["unit_inv",
               "inv_resp_norm_mean",
               "inv_resp_norm_max",
               "top100_resp"]].corr(method="spearman")
#%%
df_merge_all.groupby("layer_s",sort=False)[["sparseness", "unit_inv",
               "prct_mean",
               "prct_max",
               "top100_resp"]].corr(method="spearman")
#%%
from NN_sparseness.sparse_plot_utils import scatter_density_grid
"""plot the correlation / scatter across layers"""
for layer_s in df_merge_all.layer_s.unique():
    df_layer = df_merge_all[df_merge_all.layer_s == layer_s]
    g = scatter_density_grid(df_layer, ["sparseness","unit_inv",
                                        "inv_resp_norm_mean",
                                        "inv_resp_norm_max",
                                        "prct_max",])
    g.fig.suptitle(f"Layer {layer_s} (Spearman correlation)")
    g.fig.tight_layout()
    g.fig.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.png"))
    g.fig.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.pdf"))

#%%
df_merge_all[["layer_depth", "sparseness", "unit_inv",
              "inv_resp_norm_mean", "inv_resp_norm_max",
              "prct_max"]].corr(method="spearman")
#%%
df_all = pd.DataFrame()
sparseness_coef_D = {}
for layer in reclayers:
    featmat = feattsrs[layer]
    s_coef = (1 - featmat.mean(dim=0)**2 / featmat.pow(2).mean(dim=0)) / (1 - 1 / featmat.shape[0])
    sparseness_coef_D[layer] = s_coef
    print(f"{layer} Sparseness {torch.nanmean(s_coef):.3f}+-{np.nanstd(s_coef):.3f}")
    df = pd.DataFrame({"sparseness":sparseness_coef_D[layer]})
    df["layer"] = layer
    df["unitid"] = np.arange(len(df))
    df_all = pd.concat((df_all, df), axis=0)

df_all["layer_s"] = df_all.layer.apply(shorten_layername)
#%%
print("nan values",df_all.sparseness.isna().sum())
df_valid = df_all[~df_all.sparseness.isna()]
df_all.to_csv(join(figdir, "resnet50_linf8_unit_sparseness.csv"))
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.violinplot(x="layer_s", y="sparseness", data=df_valid, cut=0.0, ax=ax)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_sparseness_violin.png"))
plt.savefig(join(figdir, "resnet50_linf8_sparseness_violin.pdf"))
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.stripplot(x="layer_s", y="sparseness", data=df_valid, jitter=0.35, alpha=0.2, ax=ax)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_sparseness_strip.png"))
plt.savefig(join(figdir, "resnet50_linf8_sparseness_strip.pdf"))
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.lineplot(x="layer_s", y="sparseness", data=df_valid, ax=ax)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_sparseness_line.png"))
plt.savefig(join(figdir, "resnet50_linf8_sparseness_line.pdf"))
plt.show()
#%%

#%%
unit_inv_cc_dict = {}
pop_inv_cc_dict = {}
df_inv_all = pd.DataFrame()
for layer in reclayers:
    featmat = Invfeatdata[layer]  # 60 by Channel N
    feattsr = featmat.reshape(10, 6, -1).permute(2,1,0)  # 6 by 10 by Channel N
    torch.corrcoef(feattsr[0, :, :]).mean()
    # correlation of  responses to 10 objects across 6 transformations
    unit_cctsr = corrcoef_batch(feattsr)  # Chan, 6, 6
    unit_cctsr_msk = mask_diagonal(unit_cctsr)  # Chan, 6, 6
    invar_cc = unit_cctsr_msk.nanmean(dim=(1,2))  # Chan,
    # correlation of population representations across 6 transformations
    pop_cctsr = corrcoef_batch(feattsr.permute([2, 1, 0]))  # 10, 6, 6
    pop_cctsr_msk = mask_diagonal(pop_cctsr)  # 10, 6, 6
    pop_invar_cc = pop_cctsr_msk.nanmean(dim=(1, 2))  # 10,
    unit_inv_cc_dict[layer] = invar_cc
    pop_inv_cc_dict[layer] = pop_invar_cc
    print(f"{layer} unit invariance {torch.nanmean(invar_cc):.3f}+-{np.nanstd(invar_cc):.3f}\t "
          f"object invariance {torch.nanmean(pop_invar_cc):.3f}+-{np.nanstd(pop_invar_cc):.3f}")
    df = pd.DataFrame({"unit_inv": unit_inv_cc_dict[layer]})
    df["layer"] = layer
    df["unitid"] = np.arange(len(df))
    df_inv_all = pd.concat((df_inv_all, df), axis=0)

df_inv_all["layer_s"] = df_inv_all.layer.apply(shorten_layername)
#%%
msk = (df_all.sparseness.isna()) | (df_inv_all.unit_inv.isna())
pearsonr(df_all.sparseness[~msk], df_inv_all.unit_inv[~msk], )
#%%
df_comb = pd.concat((df_all, df_inv_all["unit_inv"]),axis=1)
#%%
df_comb.to_csv(join(figdir, "resnet50_linf8_unit_sparse_unitinv.csv"))
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.lineplot(x="layer_s", y="unit_inv", data=df_comb[~df_comb.unit_inv.isna()], ax=ax)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_line.png"))
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_line.pdf"))
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.stripplot(x="layer_s", y="unit_inv", data=df_comb[~df_comb.unit_inv.isna()], ax=ax,
              jitter=0.3,alpha=0.2)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_strip.png"))
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_strip.pdf"))
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=[6, 5])
sns.violinplot(x="layer_s", y="unit_inv", data=df_comb[~df_comb.unit_inv.isna()], ax=ax,)
plt.title("Resnet50 robust (linf8)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_violin.png"))
plt.savefig(join(figdir, "resnet50_linf8_unit_inv_violin.pdf"))
plt.show()
#%%
for layer in reclayers:
    msk = (df_comb.layer==layer) & (~df_comb.unit_inv.isna()) & (~df_comb.sparseness.isna())
    ccval, pval = pearsonr(df_all.sparseness[msk], df_inv_all.unit_inv[msk], )
    figh, ax = plt.subplots(1, 1, figsize=[6,6])
    sns.scatterplot(x="unit_inv", y="sparseness",
                data=df_comb[msk], ax=ax) # '.layer1.Bottleneck3'
    plt.xlabel("Unit invariance (corr)")
    plt.title(f"{layer}\nPearson R={ccval:.3f} P={pval:.1e} (N={msk.sum()})")
    plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_{layer}.png"))
    # plt.savefig(join(figdir, f"resnet50_linf8_sparseness_line.pdf"))
    plt.show()
#%%
msk = (~df_comb.unit_inv.isna()) & (~df_comb.sparseness.isna())
ccval, pval = pearsonr(df_all.sparseness[msk], df_inv_all.unit_inv[msk], )
figh, ax = plt.subplots(1, 1, figsize=[6,6])
sns.scatterplot(x="unit_inv", y="sparseness", hue="layer_s",
            data=df_comb[msk], ax=ax, alpha=0.2) # '.layer1.Bottleneck3'
plt.xlabel("Unit invariance (corr)")
plt.title(f"All layers\nPearson R={ccval:.3f} P={pval:.1e} (N={msk.sum()})")
plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_all_merge.png"))
# plt.savefig(join(figdir, f"resnet50_linf8_sparseness_line.pdf"))
plt.show()
#%%
for layer in Invfeatdata.keys():
    featmat = Invfeatdata[layer]
    s_coef = (1 - featmat.mean(dim=0)**2 / featmat.pow(2).mean(dim=0)) / (1 - 1 / featmat.shape[0])
    # sparseness_coef_D[layer] = s_coef
    print(f"{layer} Sparseness {torch.nanmean(s_coef):.3f}+-{np.nanstd(s_coef):.3f}")
#%%
df_layer = df_comb.groupby("layer_s").agg("mean")
ccval, pval = pearsonr(df_layer.sparseness, df_layer.unit_inv, )
figh, ax = plt.subplots(1, 1, figsize=[6,6])
sns.scatterplot(x="unit_inv", y="sparseness", hue="layer_s",
            data=df_layer, ax=ax, alpha=0.9) # '.layer1.Bottleneck3'
plt.xlabel("Unit invariance (corr)")
plt.title(f"All layers\nPearson R={ccval:.3f} P={pval:.1e} (N={len(df_layer)})")
plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_per_layer.png"))
plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_per_layer.pdf"))
plt.show()
#%%
figh, ax = plt.subplots(1, 1, figsize=[6,6])
sns.pointplot(x="unit_inv", y="sparseness", hue="layer_s",
            data=df_comb, ax=ax, alpha=0.9) # '.layer1.Bottleneck3'
plt.xlabel("Unit invariance (corr)")
plt.title(f"All layers\nPearson R={ccval:.3f} P={pval:.1e} (N={len(df_layer)})")
plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_per_layer_err.png"))
plt.savefig(join(figdir, f"resnet50_linf8_sparse-invar_per_layer_err.pdf"))
plt.show()


#%%
"""
Experiment on VGG16
"""
unit_list = [("vgg16", "conv2", 5, 112, 112, True),
            ("vgg16", "conv3", 5, 56, 56, True),
            ("vgg16", "conv4", 5, 56, 56, True),
            ("vgg16", "conv5", 5, 28, 28, True),
            ("vgg16", "conv6", 5, 28, 28, True),
            ("vgg16", "conv7", 5, 28, 28, True),
            ("vgg16", "conv9", 5, 14, 14, True),
            ("vgg16", "conv10", 5, 14, 14, True),
            ("vgg16", "conv12", 5, 7, 7, True),
            ("vgg16", "conv13", 5, 7, 7, True),
            ("vgg16", "fc1", 1, False),
            ("vgg16", "fc2", 1, False),
            ("vgg16", "fc3", 1, False), ]
layermap = {"conv2": ".features.ReLU3",
            "conv3": ".features.ReLU6",
            "conv4": ".features.ReLU8",
            "conv5": ".features.ReLU11",
            "conv6": ".features.ReLU13",
            "conv7": ".features.ReLU15",
            "conv9": ".features.ReLU20",
            "conv10": ".features.ReLU22",
            "conv12": ".features.ReLU27",
            "conv13": ".features.ReLU29",
            "fc1": ".classifier.ReLU1",
            "fc2": ".classifier.ReLU4",
            "fc3": ".classifier.Linear6",}
layermap_inv = {v: k for k, v in layermap.items()}
netname = "vgg16"
model, model_full = load_featnet(netname)
model.eval().cuda()

#%%  Record all layers for imagenet validation set
reclayers = [layermap[unit[1]] for unit in unit_list]
dataset = create_imagenet_valid_dataset()
feattsrs = record_dataset(model_full, reclayers, dataset, return_input=False,
                   batch_size=75, num_workers=8)
torch.save(feattsrs, join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))

df_all_spars, _ = calculate_sparseness(feattsrs)

#%% Invariance measure
Inv_data = Invariance_dataset()
Invfeatdata = record_dataset(model_full, reclayers, Inv_data, return_input=False,
                   batch_size=125, num_workers=8)
torch.save(Invfeatdata, join(outdir, "%s_invariance_feattsrs.pt"%(netname)))

df_inv_all, df_inv_all_pop = calculate_invariance(Invfeatdata, subsample=True, popsize=64, reps=100)

#%% Reload the feature tensors
feattsrs = torch.load(join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))
Invfeatdata = torch.load(join(outdir, "%s_invariance_feattsrs.pt"%(netname)))
#%%
sns.lineplot(x="layer_s", y="sparseness", data=df_all,)
plt.xticks(rotation=30)
plt.show()
#%%
sns.lineplot(x="layer_s", y="unit_inv", data=df_inv_all[~df_inv_all.unit_inv.isna()],)
plt.xticks(rotation=30)
plt.show()

#%% Load Manifold data and find kappa values
from Manifold.Manifold_Tuning_lib import load_fit_manif2table, fit_Kent_Stats, \
    violins_regress, add_regcurve
dataroot = r"E:\Cluster_Backup\manif_allchan"
netname = "vgg16"
df_all_manif = load_fit_manif2table(unit_list, netname, dataroot, ang_step=9, save=True, load=False, GANname="", savestr="")
#%%
df_all_spars = df_all_spars.rename(columns={"unitid": "iCh"})
df_all_spars = df_all_spars.rename(columns={"layer": "layer_full"})
#%%
df_inv_all = df_inv_all.rename(columns={"layer":"layer_full"})
#%%
df_kappa_merge = pd.merge(df_all_manif, df_all_spars, left_on=["layer", "iCh"], right_on=["layer_s", "iCh"])
df_kappa_merge = pd.merge(df_kappa_merge, df_inv_all, left_on=["layer", "iCh"], right_on=["layer_s", "unitid"])

#%%
PC12tab = df_kappa_merge[df_kappa_merge.space == 0]
#%%
sns.scatterplot(x="kappa", y="sparseness", hue="layer", data=PC12tab, alpha=0.2)
plt.xlim([0, 8])
plt.show()

#%% plotting utils
from NN_sparseness.sparse_plot_utils import pearson_by_layer, scatter_by_layer
#%% Print out correlations per layer.
pearson_by_layer(PC12tab, xvar="kappa", yvar="sparseness", type="spearman")
pearson_by_layer(PC12tab, xvar="kappa", yvar="unit_inv", type="spearman")
pearson_by_layer(PC12tab, xvar="sparseness", yvar="unit_inv", type="spearman")
#%%
df_kappa_merge.to_csv(join(figdir, f"{netname}_kappa_sparse_invar_merge.csv"))
#%%

scatter_by_layer(PC12tab, xvar="kappa", yvar="sparseness", type="spearman",
                 prefix=f"{netname}_manif", figdir=figdir)
#%% Visualize sparseess
plt.hist(feattsrs['.features.ReLU3'].numpy()[:, 10], bins=100)
plt.show()
#%% Visualize sparseess
plt.hist(feattsrs[layermap["conv9"]].numpy()[:, 10], bins=100)
plt.show()
#%%
for layer in layermap.keys():
    print(layer, "Sparse ratio",(feattsrs[layermap[layer]].numpy() == 0.0).mean())
#%%
df_sprs, sparseness_D = calculate_sparseness(feattsrs, subsample=True, sample_size=10000)
df_invar, df_invar_pop = calculate_invariance(Invfeatdata, subsample=True, popsize=64, reps=100)
#%%
df_merge = pd.merge(df_sprs, df_invar, on=["layer", "layer_s", "unitid"], )
#%%
manif_sum = f"E:\Cluster_Backup\manif_allchan\summary"
df_kappa = pd.read_csv(join(manif_sum, f"{netname}_ManifExpFitSum.csv"), index_col=0)
#%%
df_kappa_merge = pd.merge(df_merge, df_kappa[['layer','iCh','space','kappa', 'beta','R2']], left_on=["layer_s", "unitid"], right_on=["layer", "iCh"])
#%%
df_merge.to_csv(join(figdir, f"{netname}_sparse_invar_merge.csv"))
df_kappa_merge.to_csv(join(figdir, f"{netname}_kent_sparse_invar_merge.csv"))
#%%
# note correlation is affected by the number of channels in a layer
sns.lineplot(x="layer_s", y="pop_inv", data=df_invar_pop)
plt.show()
#%%
sns.lineplot(x="layer_s", y="unit_inv", data=df_invar[~df_invar.unit_inv.isna()])
plt.show()
#%%
sns.lineplot(x="layer_s", y="sparseness", data=df_sprs[~df_sprs.sparseness.isna()])
plt.show()
#%%
""" Sparseness vs kappa 
conv2 corr 0.066 P=6.0e-01 N=64
conv3 corr 0.418 P=8.9e-07 N=128
conv4 corr 0.110 P=2.2e-01 N=128
conv5 corr 0.302 P=8.4e-07 N=256
conv6 corr 0.172 P=5.8e-03 N=256
conv7 corr 0.179 P=4.0e-03 N=256
conv9 corr 0.200 P=8.7e-06 N=487
conv10 corr 0.244 P=2.2e-08 N=512
conv12 corr 0.120 P=6.6e-03 N=512
conv13 corr 0.190 P=1.4e-05 N=512
fc1 corr 0.179 P=2.2e-30 N=4038
fc2 corr 0.135 P=4.1e-18 N=4094
fc3 corr 0.068 P=3.1e-02 N=1000
All corr 0.107 P=1.4e-32 N=12243
"""
layeridxmap = {L:i for i, L in enumerate(layermap)}
df_kappa_merge["layer_depth"] = df_kappa_merge.layer_s.apply(lambda x: layeridxmap[x])
corrtab_pool = df_kappa_merge[["sparseness", "zero_ratio", "kurtosis", "kappa", "unit_inv", "layer_depth"]].corr(method="spearman")
corrtab_pool.to_csv(join(figdir, f"{netname}_kappa_sparse_invar_corr_pool.csv"))
#%%
df_sprs.groupby("layer_s")[["sparseness", "zero_ratio", "kurtosis"]].corr(method="spearman")
#%%
corrtab = df_kappa_merge[df_kappa_merge.space == 0].groupby("layer_y", sort=False)[["sparseness", "zero_ratio", "kurtosis", "kappa", "unit_inv"]]\
    .corr(method="spearman")
corrtab.to_csv(join(figdir, f"{netname}_kappa_sparse_invar_corr_by_layer.csv"))
#%%
