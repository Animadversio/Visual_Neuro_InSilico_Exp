import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from glob import glob
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *
from scipy.stats import pearsonr, spearmanr
from build_montages import make_grid_np, build_montages
from easydict import EasyDict as edict

def _load_proto_montage(tab, layerdir, layerfulldir=None):
    if isinstance(tab, pd.DataFrame):
        layer, unitid = tab.layer_s.iloc[0], tab.unitid
    elif isinstance(tab, pd.Series):
        layer, unitid = tab.layer_s, [tab.unitid]
    else:
        raise ValueError("tab must be a pandas.DataFrame or pandas.Series")
    if "fc" in layer:
        suffix = "original"
    else:
        suffix = "rf_fit_full"
    imgcol = []
    filenametemplate = glob(join(layerdir, f"*_{suffix}.png"))[0]
    unitpos = filenametemplate.split("\\")[-1].split("_")[3:5]
    for unit in unitid:
        if "fc" in layer:
            img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{suffix}.png"))
        else:
            img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.png"))
        # img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_rf_fit.png"))
        imgcol.append(img)
    return make_grid_np(imgcol, nrow=5), imgcol


def _load_proto_info(tabrow, layerdir, layerfulldir):
    if isinstance(tabrow, pd.Series):
        layer, unitid = tabrow.layer_s, tabrow.unitid
    elif isinstance(tabrow, pd.DataFrame):
        layer, unitid = tabrow.layer_s.iloc[0], tabrow.unitid[0]
    else:
        raise ValueError("tab must be a pandas.DataFrame or pandas.Series")
    if "fc" in layer:
        suffix = "original"
    else:
        suffix = "rf_fit_full"
    filenametemplate = glob(join(layerdir, f"*_{suffix}.png"))[0]
    unitpos = filenametemplate.split("\\")[-1].split("_")[3:5]
    unit = unitid
    if "fc" in layer:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{suffix}.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{suffix}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{suffix}.npy"))
    else:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix[:-5]}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix[:-5]}.npy"))
    return img, edict(Edata), Mdata

outdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
proto_dir = r"E:\Cluster_Backup\manif_allchan\prototypes"
outdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\proto_summary"
#%%
netname = "vgg16"
df_kappa_merge = pd.read_csv(join(figdir, f"{netname}_kent_sparse_invar_merge.csv"), index_col=0)
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
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def imgscatter(x, y, imgs, zoom=1.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y)
    for x0, y0, img in zip(x, y, imgs):
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x0, y0), frameon=False, )
        ax.add_artist(ab)
    return ax


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
    # plt.imsave(join(outdir, f"{netname}_{layer}_montage_inv_max.png"), mtg_inv_max)
    tab = df_layer.nsmallest(5, "unit_inv")
    mtg_inv_min, protos_inv_min = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_inv_min, zoom=0.2, ax=ax)
    # plt.imsave(join(outdir, f"{netname}_{layer}_montage_inv_min.png"), mtg_inv_min)
    tab = df_layer.nlargest(5, "sparseness")
    mtg_sprs_max, protos_sprs_max = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_sprs_max, zoom=0.2, ax=ax)
    # plt.imsave(join(outdir, f"{netname}_{layer}_montage_sprs_max.png"), mtg_sprs_max)
    tab = df_layer.nsmallest(5, "sparseness") # increasing order
    mtg_sprs_min, protos_sprs_min = _load_proto_montage(tab, layerdir)
    imgscatter(tab.sparseness, tab.unit_inv, protos_sprs_min, zoom=0.2, ax=ax)
    # plt.imsave(join(outdir, f"{netname}_{layer}_montage_sprs_min.png"), mtg_sprs_min)
    plt.xlabel("Sparseness", fontsize=14)
    plt.ylabel("Invariance (UnitLevel)", fontsize=14)
    plt.title(f"{netname} {layer} Sparseness - Invariance", fontsize=16)
    plt.tight_layout()
    plt.savefig(join(outdir, f"{netname}_{layer}_proto_scatter.png"))
    plt.show()
# fig, ax = plt.subplots(figsize=(6,6))
# ax.scatter(tab.sparseness, tab.unit_inv)
# for x0, y0, img in zip(tab.sparseness, tab.unit_inv, protos_sprs_min):
#     ab = AnnotationBbox(OffsetImage(img, zoom=0.2), (x0, y0), frameon=True,)
#     ax.add_artist(ab)
#%% Plot invariance as a function of response range
import torch
from os.path import join
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
#%% Plotting utils
def get_invariance_image_labels():
    img_src = r"N:\Stimuli\Invariance\Project_Manifold\ready"
    imglist = sorted(glob(join(img_src, "*.jpg")))
    tfmlabels = ["bkgrd", "left", "large", "med", "right", "small",]
    objlabels = ["birdcage", "squirrel", "monkeyL", "monkeyM", "gear", "guitar", "fruits", "pancake", "tree", "magiccube"]
    mapper = dict({"bing_birdcage_0001_seg":"birdcage",
                    "n02357911_47_seg": "squirrel",
                    "n02487347_3641_seg": "monkeyL",
                    "n02487547_1709_seg": "monkeyM",
                    "n03430551_637_seg": "gear",
                    "n03716887_63_seg": "guitar",
                    "n07753592_1991_seg": "fruits",
                    "n07880968_399_seg": "pancake",
                    "n13912260_18694_seg": "tree",
                    "n13914608_726_seg": "magiccube",})
    return imglist, tfmlabels, objlabels, mapper


def plot_invariance_tuning(inv_resps, statstr="", ax=None):
    remap_idx = [3, 0, 1, 3, 4, 5, 3, 2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    plt.sca(ax)
    sns.heatmap(inv_resps.reshape(10, 6).T[remap_idx], cmap="inferno", ax=ax)
    ax.hlines([2, 5], 0, 10, linestyles="dashed", colors="red",)
    plt.axis("image")
    plt.xticks(np.arange(10)+0.5, objlabels, rotation=45)
    plt.yticks(np.arange(8)+0.5, np.array(tfmlabels)[remap_idx], rotation=0)
    plt.title(statstr)
    plt.tight_layout()
    if ax is None: plt.show()


def plot_resp_histogram(INet_resps, inv_resps, statstr="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    plt.sca(ax)
    sns.histplot(INet_resps, bins=100, label="INet", color="blue", # [INet_resps > 0]
                 alpha=0.5, stat="density", ax=ax)
    sns.histplot(inv_resps, label="Inv", color="red",
                 alpha=0.5, stat="density", ax=ax)
    ax.eventplot(inv_resps, color="k", alpha=0.5,
                  lineoffsets=0.105, linelengths=0.2, )
    plt.ylim((-0.05, 1))
    plt.title(statstr)
    plt.legend()
    if ax is None: plt.show()


def plot_prototype(protoimg, titlestr=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    plt.sca(ax)
    plt.imshow(protoimg)
    plt.axis("off")
    if titlestr is not None:
        plt.title(titlestr)
    if ax is None: plt.show()

imglist, tfmlabels, objlabels, mapper = get_invariance_image_labels()
#%%
dataset = create_imagenet_valid_dataset(normalize=False)
#%%

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
figh, axs = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axs):
    plt.sca(ax)
    sns.heatmap(Mdata[i], cmap="inferno", ax=ax)
    plt.axis("image")
figh.tight_layout()
plt.show()

#%% Calculate additional statistics for each unit with ImageNet / Invariance data.
# layer_x = layermap["fc2"] #'.features.ReLU6'#'.classifier.ReLU1' #'.features.ReLU20'#".features.ReLU27"
df_prct_all = None
for layer_s, layer_x in layermap.items():
    df_prct = []
    for unitid in tqdm(range(inv_feattsrs[layer_x].shape[1])):
        inv_resps = inv_feattsrs[layer_x][:, unitid]
        INet_resps = INet_feattsrs[layer_x][:, unitid]
        inv_resp_np = inv_resps.numpy()
        INet_resp_np = INet_resps.numpy()
        INet_idx = np.argsort(INet_resp_np)
        top100_resp = INet_resp_np[INet_idx[-100]]
        top_norm_inv_resp = inv_resp_np / top100_resp
        rank_prct = np.searchsorted(INet_resp_np, inv_resp_np, sorter=INet_idx) / 50000
        df_prct.append({"layer_x": layer_x,
                         "layer_s": layer_s,
                         "unitid": unitid,
                         "prct_mean": rank_prct.mean(),
                         "prct_std": rank_prct.std(),
                         "prct_max": rank_prct.max(),
                         "prct_min": rank_prct.min(),
                         "top100_resp": top100_resp,
                         "inv_resp_norm_mean": top_norm_inv_resp.mean(),
                         "inv_resp_norm_std": top_norm_inv_resp.std(),
                         "inv_resp_norm_max": top_norm_inv_resp.max(),})
    df_prct = pd.DataFrame(df_prct)
    inv_zero_ratio = inv_feattsrs[layer_x].count_nonzero(dim=0) / 60
    df_prct["inv_zero_ratio"] = 1 - inv_zero_ratio

    df_prct_all = pd.concat([df_prct_all, df_prct]) if df_prct_all is not None else df_prct
#%%
df_prct_all.to_csv(join(sumdir, f"{netname}_inv_resp_prctl.csv"))
# df_layer_prct = df_layer.merge(df_prct, on=["layer_x", "unitid"], )
#%%
df_kappa_prct_merge = df_kappa_merge.merge(df_prct_all, on=["layer_x", "unitid"])
df_kappa_prct_merge.to_csv(join(sumdir, f"{netname}_kent_sparse_invar_prctl_merge.csv"))
#%%
# df_layer_prct[["sparseness","unit_inv",
#                "prct_mean","prct_std","prct_max","prct_min"]].corr(method="spearman")
layeridxmap = {L:i for i, L in enumerate(layermap)}
df_kappa_prct_merge["layer_depth"] = df_kappa_prct_merge.layer_s_x.apply(lambda x: layeridxmap[x])

#%%
df_layer_prct[["sparseness","unit_inv",
               "inv_resp_norm_mean",
               "inv_resp_norm_std",
               "inv_resp_norm_max",
               "top100_resp"]].corr(method="spearman")
#%%
df_layer_prct[["sparseness","unit_inv",
                "inv_zero_ratio",
               "inv_resp_norm_mean",
               "inv_resp_norm_max",
               "top100_resp"]].corr(method="spearman")
#%%
msk = df_kappa_prct_merge.space==0
df_kappa_prct_merge[["layer_depth",
                     "unit_inv","sparseness",
                     "inv_resp_norm_max",
                     "prct_mean",
                     "inv_zero_ratio"]][msk]\
            .corr(method="spearman")
            # .corr(method="pearson")
#%%
df_kappa_prct_merge.groupby("layer_s_x", sort=False)\
            [["sparseness", "unit_inv", "inv_resp_norm_mean"]]\
            .corr(method="spearman")
#%%
# df_layer = df_kappa_merge[msk]
# pd.DataFrame(columns=["layer", "unitid", "prct_mean", "prct_std", "prct_max", "prct_min"])
#%%
def annotate_corrfunc(x, y, hue=None, ax=None, **kws):
    # r, _ = pearsonr(x, y)
    r, pval = spearmanr(x, y, nan_policy='omit')
    ax = ax or plt.gca()
    ax.annotate("ρ = {:.3f} ({:.1e})".format(r, pval), color="red", fontsize=12,
                xy=(.1, .9), xycoords=ax.transAxes)  # xycoords='subfigure fraction')

#%%
figdir =r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"
for layer_s in layermap.keys():
    msk = (df_kappa_prct_merge.space == 0) & (df_kappa_prct_merge.layer_s_x == layer_s)
    df_layer_prct = df_kappa_prct_merge[msk]
    msk = df_layer_prct.inv_resp_norm_max < 20
    plt.figure()
    g = sns.PairGrid(df_layer_prct[["sparseness",
                                    "unit_inv",
                                    "inv_resp_norm_max",
                                    "prct_max", ]][msk]
                     , diag_sharey=False,)
    # g.map(sns.scatterplot)
    g.map_upper(sns.scatterplot, alpha=0.5)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(annotate_corrfunc, )
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    plt.suptitle(f"Layer {layer_s} (Spearman correlation)")
    plt.tight_layout()
    plt.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.png"))
    plt.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.pdf"))
    plt.show()
#%%
g = sns.PairGrid(df_layer_prct[["sparseness",
                                "unit_inv",
                                "inv_resp_norm_max",
                                "inv_resp_norm_mean",
                                ]][msk]
                 , diag_sharey=False,)
# g.map(sns.scatterplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)
plt.show()
#%%
from NN_sparseness.sparse_invariance_lib import \
    calculate_sparseness, calculate_invariance, calculate_percentile
sumdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
df_inv_all, df_inv_all_pop = calculate_invariance(inv_feattsrs, layeralias=layermap_inv,)
df_inv_all_pop.to_csv(join(sumdir, f"{netname}_pop_obj_invariance.csv"))
#%% Population invariance
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
#%% Unit level Invariance
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