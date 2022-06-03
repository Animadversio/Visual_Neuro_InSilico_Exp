import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from os.path import join
from glob import glob
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *
from scipy.stats import pearsonr, spearmanr
from build_montages import make_grid_np, build_montages
from NN_sparseness.EM_proto_utils import _load_proto_montage, _load_proto_info
proto_dir = r"E:\Cluster_Backup\manif_allchan\prototypes"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"
outdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\proto_summary"
#%%
netname = "vgg16"
df_kappa_merge = pd.read_csv(join(sumdir, f"{netname}_kent_sparse_invar_merge.csv"), index_col=0)
#%%
# proto_dir = r"N:\Data-Computational\prototypes\vgg16_conv5_manifold-"
layerlist = df_kappa_merge.layer_s.unique()
for layer in layerlist[-3:]:#["conv7", "conv9", ]:  # layerlist:
    layerdir = join(proto_dir, f"vgg16_{layer}_manifold-")
    msk = (df_kappa_merge.space == 0) & (df_kappa_merge.layer_s == layer)
    print(layer, df_kappa_merge[msk].unit_inv.mean())
    tab = df_kappa_merge[msk].nlargest(5, "unit_inv") # decreasing order of unit_inv
    mtg_inv_max, protos_inv_max = _load_proto_montage(tab, layerdir)
    plt.imsave(join(outdir, f"{netname}_{layer}_montage_inv_max.png"), mtg_inv_max)
    tab = df_kappa_merge[msk].nsmallest(5, "unit_inv")
    mtg_inv_min, protos_inv_min = _load_proto_montage(tab, layerdir)
    plt.imsave(join(outdir, f"{netname}_{layer}_montage_inv_min.png"), mtg_inv_min)
    tab = df_kappa_merge[msk].nlargest(5, "sparseness")
    mtg_sprs_max, protos_sprs_max = _load_proto_montage(tab, layerdir)
    plt.imsave(join(outdir, f"{netname}_{layer}_montage_sprs_max.png"), mtg_sprs_max)
    tab = df_kappa_merge[msk].nsmallest(5, "sparseness") # increasing order
    mtg_sprs_min, protos_sprs_min = _load_proto_montage(tab, layerdir)
    plt.imsave(join(outdir, f"{netname}_{layer}_montage_sprs_min.png"), mtg_sprs_min)
    raise Exception

#%%
from NN_sparseness.sparse_plot_utils import imgscatter
#%% plot images as scatter plots
layerlist = df_kappa_merge.layer_s.unique()
for layer in layerlist[:]:#["conv7", "conv9", ]:  # layerlist:
    fig, ax = plt.subplots(figsize=(10, 10))
    layerdir = join(proto_dir, f"vgg16_{layer}_manifold-")
    msk = (df_kappa_merge.space == 0) & (df_kappa_merge.layer_s == layer)
    df_layer = df_kappa_merge[msk]
    print(layer, df_layer.unit_inv.mean(), df_layer.sparseness.mean())
    ax.scatter(df_layer.sparseness, df_layer.unit_inv, alpha=0.3)
    tab = df_layer.nlargest(5, "unit_inv") # decreasing order of unit_inv
    mtg_inv_max, protos_inv_max = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_inv_max, zoom=0.2, ax=ax)
    tab = df_layer.nsmallest(5, "unit_inv")
    mtg_inv_min, protos_inv_min = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_inv_min, zoom=0.2, ax=ax)
    tab = df_layer.nlargest(5, "sparseness")
    mtg_sprs_max, protos_sprs_max = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_sprs_max, zoom=0.2, ax=ax)
    tab = df_layer.nsmallest(5, "sparseness") # increasing order
    mtg_sprs_min, protos_sprs_min = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_sprs_min, zoom=0.2, ax=ax)
    plt.xlabel("Sparseness", fontsize=14)
    plt.ylabel("Invariance (UnitLevel)", fontsize=14)
    plt.title(f"{netname} {layer} Sparseness - Invariance", fontsize=16)
    plt.tight_layout()
    plt.savefig(join(outdir, f"{netname}_{layer}_proto_scatter.png"))
    plt.show()

#%% Plot invariance as a function of response range
sprs_dir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
inv_feattsrs  = torch.load(join(sprs_dir, "vgg16_invariance_feattsrs.pt"))
INet_feattsrs = torch.load(join(sprs_dir, "vgg16_INvalid_feattsrs.pt"))
#%%
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
layeridxmap = {L:i for i, L in enumerate(layermap)}
#%% Plotting utils
from NN_sparseness.sparse_plot_utils import plot_invariance_tuning, plot_resp_histogram, \
                            plot_prototype, plot_Manifold_maps, get_invariance_image_labels

imglist, tfmlabels, objlabels, mapper = get_invariance_image_labels()
#%%
dataset = create_imagenet_valid_dataset(normalize=False)

#%%
layer_x = layermap["conv9"] #'.features.ReLU6'#'.classifier.ReLU1' #'.features.ReLU20'#".features.ReLU27"
msk = (df_kappa_merge.space == 0) & (df_kappa_merge.layer_x == layer_x)
df_layer = df_kappa_merge[msk]
# tab = df_layer.nlargest(5, "kappa")  # decreasing order of unit_inv
# tab = df_layer.nsmallest(5, "unit_inv")  # decreasing order of unit_inv
# tab = df_layer.nsmallest(5, "sparseness")  # decreasing order of unit_inv
# tab = df_layer.nlargest(30, "unit_inv")  # decreasing order of unit_inv
tab = df_layer[df_layer.sparseness < 0.9].nsmallest(10,"unit_inv")#nlargest(50, "unit_inv")  # decreasing order of unit_inv
# unitrow = tab.iloc[10]
unitrow = tab.sample(1).iloc[0]
#%%
# unitrow = df_layer.sample(1).iloc[0]
layer = unitrow.layer_s
unitid = unitrow.unitid
sprs = unitrow.sparseness
zeroratio = unitrow.zero_ratio
unitinv = unitrow.unit_inv
kappa = unitrow.kappa
beta = unitrow.beta
inv_resps = inv_feattsrs[layer_x][:, unitid]
INet_resps = INet_feattsrs[layer_x][:, unitid]
statstr = f"{netname} {layer} Unit {unitid} Invariance {unitinv:.2f}\nSparseness {sprs:.2f} Zero ratio {zeroratio:.2f}\nKappa {kappa:.2f} Beta {beta:.2f}"
layerdir = join(proto_dir, f"vgg16_{layer}_manifold-")
layerfulldir = join(r"E:\Cluster_Backup\manif_allchan", f"vgg16_{layer}_manifold-")
protoimg, Edata, Mdata = _load_proto_info(unitrow, layerdir, layerfulldir)
evollastgen = Edata.evol_score[Edata.evol_gen == 99].mean()
evolmax = Edata.evol_score.max()
natimgtsr, _ = dataset[INet_resps.argmax()]
natimg = natimgtsr.permute(1, 2, 0).numpy()
#%
fig = plt.figure(figsize=(12, 8), constrained_layout=False)
plt.suptitle(statstr, fontsize=20)
gs = fig.add_gridspec(2, 4)
ax1 = fig.add_subplot(gs[:, 0:2])
ax2 = fig.add_subplot(gs[0, 2:])
ax3 = fig.add_subplot(gs[1, 2])
ax4 = fig.add_subplot(gs[1, 3])
plot_resp_histogram(INet_resps, inv_resps, "", ax=ax1)
plot_invariance_tuning(inv_resps, "", ax=ax2)
plot_prototype(protoimg, f"Last mean {evollastgen:.1f} Max {evolmax:.1f}", ax=ax3)
plot_prototype(natimg, f"Score {INet_resps.max():.1f}", ax=ax4)
plt.tight_layout()
plt.show()
#%%
plot_Manifold_maps(Mdata, )

#%% Calculate additional statistics for each unit with ImageNet / Invariance data.
from NN_sparseness.sparse_invariance_lib import \
    calculate_sparseness, calculate_invariance, calculate_percentile
from NN_sparseness.sparse_plot_utils import scatter_density_grid, annotate_corrfunc

#%%
df_prct_all = calculate_percentile(INet_feattsrs, inv_feattsrs, layeralias=layermap_inv)
df_prct_all.to_csv(join(sumdir, f"{netname}_inv_resp_prctl.csv"))
#%%
df_kappa_prct_merge = df_kappa_merge.merge(df_prct_all, on=["layer_x", "unitid"])
df_kappa_prct_merge["layer_depth"] = df_kappa_prct_merge.layer_s_x.apply(lambda x: layeridxmap[x])
df_kappa_prct_merge.to_csv(join(sumdir, f"{netname}_kent_sparse_invar_prctl_merge.csv"))
#%%
msk = df_kappa_prct_merge.space == 0
df_kappa_prct_merge[["layer_depth",
                     "unit_inv","sparseness",
                     "inv_resp_norm_max",
                     "prct_mean",
                     "inv_zero_ratio"]][msk]\
            .corr(method="spearman")
#%%
df_kappa_prct_merge.groupby("layer_s_x", sort=False)\
            [["sparseness", "unit_inv", "inv_resp_norm_mean"]]\
            .corr(method="spearman")
#%%
# df_layer = df_kappa_merge[msk]
# pd.DataFrame(columns=["layer", "unitid", "prct_mean", "prct_std", "prct_max", "prct_min"])
#%%
for layer_s in layermap.keys():
    msk = (df_kappa_prct_merge.space == 0) & \
          (df_kappa_prct_merge.layer_s_x == layer_s) & \
          (df_kappa_prct_merge.inv_resp_norm_max < 20)
    df_layer = df_kappa_prct_merge[msk]
    g = scatter_density_grid(df_layer,  ["sparseness",
             "unit_inv", "inv_resp_norm_max", "prct_max", ])
    g.fig.suptitle(f"Layer {layer_s} (Spearman correlation)")
    g.fig.tight_layout()
    g.fig.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.png"))
    g.fig.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.pdf"))
    plt.show()
#%%
g = scatter_density_grid(df_layer,  ["sparseness",
                                "unit_inv",
                                "inv_resp_norm_max",
                                "inv_resp_norm_mean",
                                ])
#%%
df_inv_all, df_inv_all_pop = calculate_invariance(inv_feattsrs, layeralias=layermap_inv,)
df_inv_all_pop.to_csv(join(sumdir, f"{netname}_pop_obj_invariance.csv"))
#%% Population invariance across layers strip point plot
plt.figure(figsize=(5, 5))
sns.stripplot(x="layer_s", y="pop_inv", data=df_inv_all_pop, jitter=True, alpha=0.6)
sns.pointplot(x="layer_s", y="pop_inv", data=df_inv_all_pop, color="black", alpha=0.6)
plt.xticks(rotation=30)
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Population Invariance (per obj)", fontsize=14)
plt.title(f"{netname} Population Invariance", fontsize=16)
plt.tight_layout()
plt.savefig(join(sumdir, f"{netname}_pop_obj_invariance_strip.png"))
plt.savefig(join(sumdir, f"{netname}_pop_obj_invariance_strip.pdf"))
plt.show()
#%% Unit level Invariance across layers strip point plot
plt.figure(figsize=(5, 5))
# sns.stripplot(x="layer_s", y="unit_inv", data=df_inv_all, jitter=True, alpha=0.1)
sns.violinplot(x="layer_s", y="unit_inv", data=df_inv_all,  alpha=0.1, cut=0)
sns.pointplot(x="layer_s", y="unit_inv", data=df_inv_all, color="black", alpha=0.6)
plt.xticks(rotation=30)
plt.xlabel("Layer", fontsize=14)
plt.ylabel("Unit Invariance", fontsize=14)
plt.title(f"{netname} Unit Invariance", fontsize=16)
plt.tight_layout()
plt.savefig(join(sumdir, f"{netname}_unit_invariance_violin.png"))
plt.savefig(join(sumdir, f"{netname}_unit_invariance_violin.pdf"))
plt.show()

#%%