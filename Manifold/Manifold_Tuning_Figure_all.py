""" Do the plot for all the networks processed online.
Networks
Final weapon of mass generating figures of same setup.
Initiated around Jan, Feb 2021.
Updated and finalize June 22nd.
Binxu
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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%% Fit Kent function, append to the list . and form a table.
from Manifold.Kent_fit_utils import fit_Kent_Stats

dataroot = r"E:\Cluster_Backup\CNN_manifold"
sumdir = r"E:\Cluster_Backup\CNN_manifold\summary"
figdir = sumdir
#%%
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
#%%
#%% CorNet_s model comparison
unit_list = [("Cornet_s", ".V1.ReLUnonlin1", 5, 57, 57, True),
        ("Cornet_s", ".V1.ReLUnonlin2", 5, 28, 28, True),
        ("Cornet_s", ".V2.Conv2dconv_input", 5, 28, 28, True),
        ("Cornet_s", ".CORblock_SV2", 5, 14, 14, True),
        ("Cornet_s", ".V4.Conv2dconv_input", 5, 14, 14, True),
        ("Cornet_s", ".CORblock_SV4", 5, 7, 7, True),
        ("Cornet_s", ".IT.Conv2dconv_input", 5, 7, 7, True),
        ("Cornet_s", ".CORblock_SIT", 5, 3, 3, True),
        ("Cornet_s", ".decoder.Linearlinear", 5, False), ]
layerlist = [unit[1] for unit in unit_list]
nettab_c = load_fit_manif2table(unit_list, "cornet_s", dataroot, save=True, savestr="_RFfit")
unit_list += [("Cornet_s", ".V1.ReLUnonlin1", 5, 57, 57, False),
        ("Cornet_s", ".V1.ReLUnonlin2", 5, 28, 28, False),
        ("Cornet_s", ".V2.Conv2dconv_input", 5, 28, 28, False),
        ("Cornet_s", ".CORblock_SV2", 5, 14, 14, False),
        ("Cornet_s", ".V4.Conv2dconv_input", 5, 14, 14, False),
        ("Cornet_s", ".CORblock_SV4", 5, 7, 7, False),
        ("Cornet_s", ".IT.Conv2dconv_input", 5, 7, 7, False),
        ("Cornet_s", ".CORblock_SIT", 5, 3, 3, False),]
nettab_f = load_fit_manif2table(unit_list, "cornet_s", dataroot, save=True, savestr="_All")
#%%
layerlist = ['.V1.ReLUnonlin1',
             '.V1.ReLUnonlin2',
             '.V2.Conv2dconv_input',
             '.CORblock_SV2',
             '.V4.Conv2dconv_input',
             '.CORblock_SV4',
             '.IT.Conv2dconv_input',
             '.CORblock_SIT',
             '.decoder.Linearlinear']
layerlabel = ['V1.ReLU1',
             'V1.ReLU2',
             'V2.conv_input',
             'CORblock_SV2',
             'V4.conv_input',
             'CORblock_SV4',
             'IT.conv_input',
             'CORblock_SIT',
             'lineardecoder']
#%%
msk = (nettab_c.R2>0.5) & (nettab_c.evolfinact>0.1) & (nettab_c.evolttest_p<0.01) & (nettab_c.space==0)
fig1 = violins_regress(nettab_c, "cornet_s", layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
fig2 = violins_regress(nettab_c, "cornet_s", layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
#%%
msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>0.1) & (nettab_f.evolttest_p<0.01) & (
        nettab_f.space==0)
fig3 = violins_regress(nettab_f, "cornet_s", layerlist, figdir=figdir, msk=msk,\
        varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing", layerlabel=layerlabel)
fig3 = violins_regress(nettab_f, "cornet_s", layerlist[:-1], figdir=figdir, msk=msk,\
        varnm="kappa", savestr="RFfit_nonRF-1_bsl", titstr="No RF resizing", layerlabel=layerlabel)
fig3 = violins_regress(nettab_f, "cornet_s", layerlist[:-2], figdir=figdir, msk=msk,\
        varnm="kappa", savestr="RFfit_nonRF-2_bsl", titstr="No RF resizing", layerlabel=layerlabel)
#%% DenseNet169
netname = "densenet169"
unit_list = [("densenet169", ".features.ReLUrelu0", 5, 57, 57, True),
             ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28, True),
             ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28, True),
             ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14, True),
             ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14, True),
             ("densenet169", ".features._DenseBlockdenseblock3", 5, 7, 7, False),
             ("densenet169", ".features.transition3.Conv2dconv", 5, 7, 7, False),
             ("densenet169", ".features._DenseBlockdenseblock4", 5, 3, 3, False),
             ("densenet169", ".Linearclassifier", 5, False), ]
layerlist = [unit[1] for unit in unit_list]
layerlabel = ["features.relu0", 
             "features.denseblock1", 
             "features.transition1", 
             "features.denseblock2", 
             "features.transition2", 
             "features.denseblock3", 
             "features.transition3", 
             "features.denseblock4", 
             "Linearclassifier"]
nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")
unit_list += [("densenet169", ".features.ReLUrelu0", 5, 57, 57, False),
             ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28, False),
             ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28, False),
             ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14, False),
             ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14, False),]
nettab_f = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_All")
#%%
nettab_d = pd.read_csv(join(sumdir, "densenet169"+"_ManifExpFitSum_RFfit.csv"))
msk = (nettab_d.R2>0.5) & (nettab_d.evolfinact>0.4) & (nettab_d.evolttest_p<0.01) & (nettab_d.space==0)
fig1 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
fig1 = violins_regress(nettab_d, netname, layerlist[:-1], figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb-1_bsl", layerlabel=layerlabel)
fig2 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>0.4)\
        & (nettab_f.evolttest_p<0.01) & (nettab_f.space==0)
fig3 = violins_regress(nettab_f, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing", layerlabel=layerlabel)
#%% AlexNet
netname = "alexnet"
unit_list = [("alexnet", "conv1_relu", 5, 28, 28, True),
            ("alexnet", "conv2_relu", 5, 13, 13, True),
            ("alexnet", "conv3_relu", 5, 6, 6, True),
            ("alexnet", "conv4_relu", 5, 6, 6, True),
            ("alexnet", "conv5_relu", 5, 6, 6, True),
            ("alexnet", "fc6", 5, False),
            ("alexnet", "fc7", 5, False),
            ("alexnet", "fc8", 5, False),]
layerlist = [unit[1] for unit in unit_list]
nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")
unit_list += [("alexnet", "conv1_relu", 5, 28, 28, False),
            ("alexnet", "conv2_relu", 5, 13, 13, False),
            ("alexnet", "conv3_relu", 5, 6, 6, False),
            ("alexnet", "conv4_relu", 5, 6, 6, False),
            ("alexnet", "conv5_relu", 5, 6, 6, False),]
nettab_f = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_All")
#%%
msk = (nettab_d.R2>0.5) & (nettab_d.evolfinact>5) & (nettab_d.evolttest_p<0.001) & (nettab_d.space == 0)
fig1 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl")
fig1 = violins_regress(nettab_d, netname, layerlist[:-1], figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb-1_bsl")
fig2 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl")
msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>5) \
        & (nettab_f.evolttest_p<0.001) & (nettab_f.space == 0)
fig3 = violins_regress(nettab_f, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing")
#%% ResNet101
netname = "resnet101"
unit_list = [("resnet101", ".ReLUrelu", 5, 56, 56, True), 
            ("resnet101", ".layer1.Bottleneck0", 5, 28, 28, True), 
            ("resnet101", ".layer1.Bottleneck1", 5, 28, 28, True), 
            ("resnet101", ".layer2.Bottleneck0", 5, 14, 14, True), 
            ("resnet101", ".layer2.Bottleneck3", 5, 14, 14, True), 
            ("resnet101", ".layer3.Bottleneck0", 5, 7, 7, True), 
            ("resnet101", ".layer3.Bottleneck2", 5, 7, 7, True), 
            ("resnet101", ".layer3.Bottleneck6", 5, 7, 7, True), 
            ("resnet101", ".layer3.Bottleneck10", 5, 7, 7, True), 
            ("resnet101", ".layer3.Bottleneck14", 5, 7, 7, False), 
            ("resnet101", ".layer3.Bottleneck18", 5, 7, 7, False), 
            ("resnet101", ".layer3.Bottleneck22", 5, 7, 7, False), 
            ("resnet101", ".layer4.Bottleneck0", 5, 4, 4, False), 
            ("resnet101", ".layer4.Bottleneck2", 5, 4, 4, False), 
            ("resnet101", ".Linearfc", 5, False)]
layerlist = [unit[1] for unit in unit_list]
layerlabel = [name[1:] for name in layerlist]
layerlabel[0] = "Relu"
nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")

unit_list_full = unit_list + \
            [("resnet101", ".ReLUrelu", 5, 56, 56, False), 
            ("resnet101", ".layer1.Bottleneck0", 5, 28, 28, False), 
            ("resnet101", ".layer1.Bottleneck1", 5, 28, 28, False), 
            ("resnet101", ".layer2.Bottleneck0", 5, 14, 14, False), 
            ("resnet101", ".layer2.Bottleneck3", 5, 14, 14, False), 
            ("resnet101", ".layer3.Bottleneck0", 5, 7, 7, False), 
            ("resnet101", ".layer3.Bottleneck2", 5, 7, 7, False), 
            ("resnet101", ".layer3.Bottleneck6", 5, 7, 7, False), 
            ("resnet101", ".layer3.Bottleneck10", 5, 7, 7, False)] 
nettab_f = load_fit_manif2table(unit_list_full, netname, dataroot, save=True, savestr="_All")
#%%
msk = (nettab_d.R2>0.5) & (nettab_d.evolfinact>0.2) & (nettab_d.evolttest_p<0.001) & (nettab_d.space == 0)
fig1 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
fig1 = violins_regress(nettab_d, netname, layerlist[:-1], figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb-1_bsl", layerlabel=layerlabel)
fig2 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>0.2) \
      & (nettab_f.evolttest_p<0.001) & (nettab_f.space == 0)
fig3 = violins_regress(nettab_f, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing", layerlabel=layerlabel)

#%% resnet50
netname = "resnet50"
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57, True),
             ("resnet50", ".layer1.Bottleneck1", 5, 28, 28, True),
             ("resnet50", ".layer2.Bottleneck0", 5, 14, 14, True),
             ("resnet50", ".layer2.Bottleneck2", 5, 14, 14, True),
             ("resnet50", ".layer3.Bottleneck0", 5, 7, 7, True),
             ("resnet50", ".layer3.Bottleneck2", 5, 7, 7, True),
             ("resnet50", ".layer3.Bottleneck4", 5, 7, 7, True),
             ("resnet50", ".layer4.Bottleneck0", 5, 4, 4, False),
             ("resnet50", ".layer4.Bottleneck2", 5, 4, 4, False),
             ("resnet50", ".Linearfc", 5, False)]
layerlist = [unit[1] for unit in unit_list]
layerlabel = [name[1:] for name in layerlist]
layerlabel[0] = "Relu"
nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")
# Forget to do the RF not fit experiments
msk = (nettab_d.R2>0.5) & (nettab_d.evolfinact>0.2) & (nettab_d.evolttest_p<0.001) & (nettab_d.space == 0)
fig1 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
fig1 = violins_regress(nettab_d, netname, layerlist[:-1], figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb-1_bsl", layerlabel=layerlabel)
fig2 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
# msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>0.2)
# fig3 = violins_regress(nettab_f, netname, layerlist, figdir=figdir, msk=msk,\
#                 varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing")
#%% resnet50_robust
netname = "resnet50_linf_8"
unit_list = [("resnet50_linf_8", ".ReLUrelu", 5, 57, 57, True),
             ("resnet50_linf_8", ".layer1.Bottleneck1", 5, 28, 28, True),
             ("resnet50_linf_8", ".layer2.Bottleneck0", 5, 14, 14, True),
             ("resnet50_linf_8", ".layer2.Bottleneck2", 5, 14, 14, True),
             ("resnet50_linf_8", ".layer3.Bottleneck0", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer3.Bottleneck2", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer3.Bottleneck4", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer4.Bottleneck0", 5, 4, 4, False),
             ("resnet50_linf_8", ".layer4.Bottleneck2", 5, 4, 4, False),
             ("resnet50_linf_8", ".Linearfc", 5, False)]
layerlist = [unit[1] for unit in unit_list]
layerlabel = [name[1:] for name in layerlist]
layerlabel[0] = "Relu"
nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")
# Forget to do the RF not fit experiments
msk = (nettab_d.R2>0.5) & (nettab_d.evolfinact>0.2) & (nettab_d.evolttest_p<0.001) & (nettab_d.space == 0)
fig1 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
fig1 = violins_regress(nettab_d, netname, layerlist[:-1], figdir=figdir, msk=msk,\
                varnm="kappa", savestr="RFfit_cmb-1_bsl", layerlabel=layerlabel)
fig2 = violins_regress(nettab_d, netname, layerlist, figdir=figdir, msk=msk,\
                varnm="beta", savestr="RFfit_cmb_bsl", layerlabel=layerlabel)
# msk = (~nettab_f.RFfit) & (nettab_f.R2>0.5) & (nettab_f.evolfinact>0.2)
# fig3 = violins_regress(nettab_f, netname, layerlist, figdir=figdir, msk=msk,\
#                 varnm="kappa", savestr="RFfit_nonRF_bsl", titstr="No RF resizing")




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Plotting Development zone %%%
netname = "resnet50"  # "resnet50_linf_8"  #
nettab = pd.read_csv(join(sumdir, '%s_Select_expFitSum.csv'%netname))
#%% Kappa version
msk = (nettab.R2>0.5) * (nettab.evolfinact>0.1)
layerlist = [unit[1] for unit in unit_list]
layermap = {v:k for k, v in enumerate(layerlist)}
plt.figure()
ax = sns.violinplot(x="layer", y="kappa", name=layerlist, dodge=True,
            data=nettab[msk], inner="point", meanline_visible=True, jitter=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
for dots in zip(ax.collections[1::2]):
    dots[0].set_alpha(0.2)
plt.xticks(rotation=30)
slope, intercept, r_val, p_val, stderr = linregress(nettab[msk]["layer"].map(layermap), nettab[msk].kappa)
statstr = "All layers Kappa value vs layer num:\nkappa = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(nettab[msk]))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Kappa Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_kappaRFfit_cmb_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaRFfit_cmb_bsl_pur_violin.pdf"%netname))
plt.show()
#%% Beta version
plt.figure()
ax = sns.violinplot(x="layer", y="beta", name=layerlist, dodge=True,
            data=nettab[msk], inner="point", meanline_visible=True, jitter=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
for dots in zip(ax.collections[1::2]):
    dots[0].set_alpha(0.2)
plt.xticks(rotation=30)
slope, intercept, r_val, p_val, stderr = linregress(nettab[msk]["layer"].map(layermap), nettab[msk].beta)
statstr = "All layers Beta value vs layer num:\nbeta = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(nettab[msk]))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Beta Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_betaRFfit_cmb_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_betaRFfit_cmb_bsl_pur_violin.pdf"%netname))
plt.show()

#%% Strings from bash commands 

expcmdstrs = ["--units resnet50_linf_8 .ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27  --RFfi --chan_rng 0 75",
"--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 0 75",
"--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 0 75",
"--units resnet50_linf_8 .Linearfc 5 --chan_rng 0 75",]


expcmdstrs = ["--units resnet50 .ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 75",
"--units resnet50 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101 --RFfit --chan_rng 0 75",
"--units resnet50 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit --chan_rng 0 75",
"--units resnet50 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit --chan_rng 0 75",
"--units resnet50 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 0 75",
"--units resnet50 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 0 75",
"--units resnet50 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 0 75",
"--units resnet50 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0 --RFfit --chan_rng 0 75",
"--units resnet50 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0 --RFfit --chan_rng 0 75",
"--units resnet50 .Linearfc 5 --chan_rng 0 75",]
#%%
#%% Development zone !
netname = "resnet50_linf_8"  # "resnet50"
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57, True), # last entry signify if we do RF resizing or not.
            ("resnet50", ".layer1.Bottleneck1", 5, 28, 28, True),
            ("resnet50", ".layer2.Bottleneck0", 5, 14, 14, True),
            ("resnet50", ".layer2.Bottleneck2", 5, 14, 14, True),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck4", 5, 7, 7, True),
            ("resnet50", ".layer4.Bottleneck0", 5, 4, 4, False),
            ("resnet50", ".layer4.Bottleneck2", 5, 4, 4, False),
            ("resnet50", ".Linearfc", 5, False), ]

# for layer, RFfit in layerlist:
ang_step = 9
theta_arr = np.arange(-90, 90.1, ang_step) / 180 * np.pi
phi_arr = np.arange(-90, 90.1, ang_step) / 180 * np.pi
param_names = ["theta", "phi", "psi", "kappa", "beta", "A", "bsl"]
param_std_names = [p+"_std" for p in param_names]
stat_col = []
for unit in unit_list[:]:
    layer = unit[1]
    layerdir = "%s_%s_manifold-" % (netname, layer)
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
    for iCh in iChlist: # range
        unitstat = EasyDict()
        if len(unit) == 6:
            unit_lab = "%s_%d_%d_%d"%(layer, iCh, unit[3], unit[4])
            unitstat.pos = (unit[3], unit[4])
        elif len(unit) == 4:
            unit_lab = "%s_%d" % (layer, iCh, )
            unitstat.pos = None
        else:
            raise NotImplementedError
        explabel = "%s_%s" % (unit_lab, suffix)
        data = np.load(join(dataroot, layerdir, "Manifold_score_%s.npy"%(explabel)))
        Mdata = np.load(join(dataroot, layerdir, "Manifold_set_%s.npz"%(explabel)))
        spi = 0
        actmap = data[spi, :, :]
        param, param_std, _, R2 = fit_Kent_Stats(theta_arr=theta_arr, phi_arr=phi_arr, act_map=actmap)
        unitstat.netname = netname
        unitstat.layer = layer
        unitstat.iCh = iCh
        unitstat.explabel = explabel
        unitstat.space = spi
        unitstat.RFfit = RFfit
        unitstat.imgsize = Mdata["imgsize"]
        unitstat.corner = Mdata["corner"]
        # Maximal activation from Manifold, Evolution
        unitstat.actmax = actmap.max()
        unitstat.actmin = actmap.min()
        gens = Mdata["evol_gen"]
        unitstat.evolfinact = Mdata["evol_score"][gens == gens.max()].mean()
        # Fitting stats
        unitstat.R2 = R2
        for i, pnm in enumerate(param_names):
            unitstat[pnm] = param[i]
            unitstat[pnm+"_std"] = param_std[i]
        # Append to collection
        stat_col.append(unitstat)

nettab = pd.DataFrame(stat_col)

#%%
alltab = []
subsp_nm = ["PC23","PC2526","PC4950","RND12"]
for li in range(param_col_arr.shape[0]):
    for ui in range(param_col_arr.shape[1]):
        for si in range(param_col_arr.shape[2]):
            alltab.append([layers[li],ui,si,subsp_nm[si],stat_col_arr[li,ui,si]] \
                          + list(param_col_arr[li,ui,si,:]) + list(sigma_col_arr[li,ui,si,:]))
param_names = list(param_name)
param_std_names = [p+"_std" for p in param_names]
# alltab = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2", \
#             "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
alltab_bsl = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2", ] + param_names +
                                          param_std_names)
