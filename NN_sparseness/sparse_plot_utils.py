import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import matplotlib
from os.path import join
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def pearson_by_layer(PC12tab, xvar="kappa", yvar="sparseness", groupvar="layer", type="pearson"):
    corrfun = pearsonr if type == "pearson" else spearmanr
    print(f"{type} {xvar} vs {yvar}")
    validmask = (~PC12tab[xvar].isna()) & (~PC12tab[yvar].isna())
    for layer in PC12tab[groupvar].unique():
        cval, pval = corrfun(PC12tab[validmask & (PC12tab[groupvar] == layer)][xvar],
                              PC12tab[validmask & (PC12tab[groupvar] == layer)][yvar])
        print(f"{layer} corr {cval:.3f} P={pval:.1e} N={(validmask & (PC12tab.layer == layer)).sum()}")

    cval, pval = corrfun(PC12tab[validmask][xvar], PC12tab[validmask][yvar])
    print(f"{'All'} corr {cval:.3f} P={pval:.1e} N={(validmask).sum()}")
    return cval, pval


def scatter_by_layer(PC12tab, xvar="kappa", yvar="sparseness", groupvar="layer",
                     type="pearson", prefix="", figdir=None):
    corrfun = pearsonr if type == "pearson" else spearmanr
    print(f"{type} {xvar} vs {yvar}")
    validmask = (~PC12tab[xvar].isna()) & (~PC12tab[yvar].isna())
    for layer in PC12tab[groupvar].unique():
        cval, pval = corrfun(PC12tab[validmask & (PC12tab[groupvar] == layer)][xvar],
                              PC12tab[validmask & (PC12tab[groupvar] == layer)][yvar])
        print(f"{layer} corr {cval:.3f} P={pval:.1e} N={(validmask & (PC12tab.layer == layer)).sum()}")
        figh,ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(x=xvar, y=yvar, hue=groupvar, data=PC12tab[validmask & (PC12tab[groupvar] == layer)],ax=ax,alpha=0.2)
        plt.title(f"{type} {xvar} vs {yvar}\n{layer} corr {cval:.3f} P={pval:.1e} N={(validmask).sum()}")
        # plt.axis("square")
        plt.savefig(join(figdir,f"{prefix}_{layer}_{xvar}_{yvar}.png"))
        plt.show()

    cval, pval = corrfun(PC12tab[validmask][xvar], PC12tab[validmask][yvar])
    print(f"{'All'} corr {cval:.3f} P={pval:.1e} N={(validmask).sum()}")
    figh, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=xvar, y=yvar, hue=groupvar, data=PC12tab[validmask], ax=ax,alpha=0.2)
    # plt.axis("square")
    plt.title(f"{type} {xvar} vs {yvar}\n{'All'} corr {cval:.3f} P={pval:.1e} N={(validmask).sum()}")
    plt.savefig(join(figdir, f"{prefix}_All_{xvar}_{yvar}.png"))
    plt.show()
    return cval, pval


def annotate_corrfunc(x, y, hue=None, ax=None, **kws):
    # r, _ = pearsonr(x, y)
    r, pval = spearmanr(x, y, nan_policy='omit')
    ax = ax or plt.gca()
    ax.annotate("ρ = {:.3f} ({:.1e})".format(r, pval), color="red", fontsize=12,
                xy=(.1, .9), xycoords=ax.transAxes)  # xycoords='subfigure fraction')


def scatter_density_grid(df_layer, cols, ):
    plt.figure()
    g = sns.PairGrid(df_layer[cols], diag_sharey=False, )
    g.map_upper(sns.scatterplot, alpha=0.5)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(annotate_corrfunc, )
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    # plt.suptitle(f"Layer {layer_s} (Spearman correlation)")
    plt.tight_layout()
    # plt.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.png"))
    # plt.savefig(join(figdir, f"{netname}_layer_{layer_s}_inv_sprs_prctl_scatter.pdf"))
    plt.show()
    return g


def get_invariance_image_labels():
    try:
        from glob import glob
        img_src = r"N:\Stimuli\Invariance\Project_Manifold\ready"
        imglist = sorted(glob(join(img_src, "*.jpg")))
    except:
        imglist = None
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


def plot_Manifold_maps(Mdata, titlestr=None, ax=None):
    figh, axs = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axs):
        plt.sca(ax)
        sns.heatmap(Mdata[i], cmap="inferno", ax=ax)
        plt.axis("image")
    if titlestr is not None:
        plt.title(titlestr)
    figh.tight_layout()
    plt.show()

inv_imglist, tfmlabels, objlabels, mapper = get_invariance_image_labels()

#%% Scatter images on the plot
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