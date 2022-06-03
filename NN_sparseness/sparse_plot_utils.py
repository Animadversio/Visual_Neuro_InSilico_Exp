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