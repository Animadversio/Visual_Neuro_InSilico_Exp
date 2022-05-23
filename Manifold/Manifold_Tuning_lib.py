"""
Reuseable analysis code for analyzing in silico manifold experiments.
Key functions:
    load_fit_manif2table(unit_list, netname, dataroot, ang_step=9,
        save=True, load=False, GANname="", savestr="")

    Find experiments done to all layers in a CNN and fit it with Kent,
    then save the results into a table.

"""
import pandas as pd
import numpy as np
from glob import glob
import os, re
from os.path import join
import matplotlib as mpl
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import ttest_1samp
from scipy.stats import linregress, spearmanr
from easydict import EasyDict
from Manifold.Kent_fit_utils import fit_Kent_Stats

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

param_names = ["theta", "phi", "psi", "kappa", "beta", "A", "bsl"]
param_std_names = [p+"_std" for p in param_names]
def load_fit_manif2table(unit_list, netname, dataroot, ang_step=9, save=True, load=False, GANname="", savestr=""):
    """Load experiments into table, Algorithmic version
    Esp. it load evolution information into the tab.
    load: if true, it will load saved stats table instead of computing a new one.
    """
    if load:
        nettab = pd.from_csv(join(dataroot, "summary", '%s_ManifExpFitSum%s.csv'%(netname, savestr)))
        return nettab
    theta_arr = np.arange(-90, 90.1, ang_step) / 180 * np.pi
    phi_arr = np.arange(-90, 90.1, ang_step) / 180 * np.pi
    stat_col = []
    for unit in unit_list[:]:
        layer = unit[1]
        layerdir = "%s_%s_manifold-%s" % (netname, layer, GANname)
        RFfit = unit[-1]
        suffix = "rf_fit" if RFfit else "original"
        npyfns = glob(join(dataroot, layerdir, "*.npy"))
        if len(unit) == 6:
            pattern = re.compile("Manifold_score_%s_(\d*)_%d_%d_%s.npy"%(layer, unit[3], unit[4], suffix))
        else:
            pattern = re.compile("Manifold_score_%s_(\d*)_%s.npy"%(layer, suffix))
        matchpatt = [pattern.findall(fn) for fn in npyfns]
        iChlist = [int(mat[0]) for mat in matchpatt if len(mat)==1]
        fnlist = [fn for mat, fn in zip(matchpatt, npyfns) if len(mat) == 1]
        print("Found %d units in %s - %s layer!"%(len(iChlist), netname, layer))
        for iCh in iChlist: # range
            if len(unit) == 6:
                unit_lab = "%s_%d_%d_%d"%(layer, iCh, unit[3], unit[4])
            elif len(unit) == 4:
                unit_lab = "%s_%d" % (layer, iCh, )
            else:
                raise NotImplementedError
            explabel = "%s_%s" % (unit_lab, suffix)
            data = np.load(join(dataroot, layerdir, "Manifold_score_%s.npy"%(explabel)))
            Mdata = np.load(join(dataroot, layerdir, "Manifold_set_%s.npz"%(explabel)))
            # final generation activation from Evolution
            gens = Mdata["evol_gen"]
            finalscores = Mdata["evol_score"][gens == gens.max()]
            initscores = Mdata["evol_score"][gens == (gens.min()+1)]
            tval, pval = ttest_1samp(finalscores, initscores.mean())
            for spi in range(data.shape[0]): # all spaces
                unitstat = EasyDict()
                if len(unit) == 6:
                    unitstat.pos = (unit[3], unit[4])
                elif len(unit) == 4:
                    unitstat.pos = None
                actmap = data[spi, :, :]  # PC2-3 space
                param, param_std, _, R2 = fit_Kent_Stats(theta_arr=theta_arr, phi_arr=phi_arr, act_map=actmap)
                unitstat.netname = netname
                unitstat.layer = layer
                unitstat.iCh = iCh
                unitstat.explabel = explabel
                unitstat.space = spi
                unitstat.RFfit = RFfit
                unitstat.imgsize = Mdata["imgsize"]
                unitstat.corner = Mdata["corner"]
                # Maximal activation from Manifold,
                unitstat.actmax = actmap.max()
                unitstat.actmin = actmap.min()
                unitstat.evolfinact = finalscores.mean()
                unitstat.evolttest = tval
                unitstat.evolttest_p = pval
                # Fitting stats
                unitstat.R2 = R2
                for i, pnm in enumerate(param_names):
                    unitstat[pnm] = param[i]
                    unitstat[pnm+"_std"] = param_std[i]
                # Append to collection
                stat_col.append(unitstat)

    nettab = pd.DataFrame(stat_col)
    if save:
        os.makedirs(join(dataroot, "summary"), exist_ok=True)
        nettab.to_csv(join(dataroot, "summary", '%s_ManifExpFitSum%s.csv'%(netname, savestr)))
    return nettab


def add_regcurve(ax, slope, intercept, **kwargs):
    XLIM = ax.get_xlim()
    ax.plot(XLIM, np.array(XLIM) * slope + intercept, **kwargs)


def violins_regress(nettab, netname, layerlist, figdir="", varnm="kappa", savestr="RFfit_cmb_bsl", titstr="",
        layernummap=None, msk=slice(None), violinalpha=0.3, pointalpha=0.2, layerlabel=None):
    """major figure format for the progressions plot multiple regressions + regression line.
    It's a wrapper around sns.violinplot, adding statistical testing for trend and regression line to the plot.
    layerlist: layers to plot and their order.
    layernummap: mapping from layer name as in layerlist to a number. Used to do trend testing.
    layerlabel: name to show for each layer in the xlabel.
    """
    # msk = (nettab.R2 > 0.5) * (nettab.evolfinact > 0.1)
    # layerlist = [unit[1] for unit in unit_list]
    if layernummap is None:
        layernummap = {v: k for k, v in enumerate(layerlist)}
    fig = plt.figure(figsize=(6, 6))
    ax = sns.violinplot(x="layer", y=varnm, name=layerlist, dodge=True, order=layerlist,
                        data=nettab[msk], inner="point", meanline_visible=True, jitter=True)
    for violin in zip(ax.collections[::2]):
        violin[0].set_alpha(violinalpha)
    for dots in zip(ax.collections[1::2]):
        dots[0].set_alpha(pointalpha)
    if layerlabel is not None:
        ax.set_xticklabels(layerlabel)
    plt.xticks(rotation=30)
    laynumvec = nettab[msk]["layer"].map(layernummap)
    nanmsk = laynumvec.isna()
    ccval, cc_pval = spearmanr(laynumvec[~nanmsk], nettab[msk][~nanmsk][varnm])
    slope, intercept, r_val, p_val, stderr = linregress(laynumvec[~nanmsk], nettab[msk][~nanmsk][varnm])
    statstr = "All layers %s value vs layer num:\n%s = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
              "p=%.1e N=%d\n Spearman Corr %.3f p=%.1e" % (varnm, varnm, slope, intercept, stderr, r_val, p_val, len(nettab[msk]), ccval, cc_pval)
    add_regcurve(ax, slope, intercept, alpha=0.5, color="gray")
    plt.title("CNN %s Manifold Exp %s Progression %s\n" % (varnm, netname, titstr) + statstr)
    plt.savefig(join(figdir, "%s_%s%s_violin.png" % (netname, varnm, savestr)))
    plt.savefig(join(figdir, "%s_%s%s_violin.pdf" % (netname, varnm, savestr)))
    plt.show()
    return fig