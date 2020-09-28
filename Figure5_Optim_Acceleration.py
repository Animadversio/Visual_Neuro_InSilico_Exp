""" This script is devoted to plot the method comparison between
1. Hessian Adam > Basin CMA, and normal Adam

Comparison between Hessian CMA and normal CMA.
1. Plot separating layers in CNN.
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sys
import os
from os.path import join
from time import time
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.cm as cm
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\Figure5"
#%% Simple Adam VS Adam with Hessian basis
#  Same result for ImageNet and BigGAN generated images
summarydir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet\summary"
expalign_tab_imgnt = pd.read_csv(join(summarydir, "exprecord_align.csv"))
#%%
summarydir = r"E:\Cluster_Backup\BigGAN_invert\BigGAN_rnd\summary"
expalign_tab_rnd = pd.read_csv(join(summarydir, "exprecord_align.csv"))
#%%
method_list = ['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none']
label_list = ["BasinCMA Hess", "BasinCMA none"]
savestr = "BasinCMA"

method_list = ['CMA50Adam0Final500_postAdam_all',
               'CMA50Adam0Final500_postAdam_none',]
label_list = ["CMAAdam Hess", "CMAAdam none"]
savestr = "CMAAdam"

method_list = ['CMA1Adam30Final600_postAdam_all',
               'CMA1Adam30Final600_postAdam_none',]
label_list = ["Adam Hess", "Adam none"]
savestr = "Adam"
nmeth = 2
imgnet_msk = (~ expalign_tab_imgnt[method_list[0]].isna()) * (~ expalign_tab_imgnt[method_list[1]].isna())
rand_msk = (~ expalign_tab_rnd[method_list[0]].isna()) * (~ expalign_tab_rnd[method_list[1]].isna())
imgnet_mean = [expalign_tab_imgnt[method][imgnet_msk].mean() for method in method_list]
imgnet_sem = [expalign_tab_imgnt[method][imgnet_msk].sem() for method in method_list]
rnd_mean = [expalign_tab_rnd[method][rand_msk].mean() for method in method_list]
rnd_sem = [expalign_tab_rnd[method][rand_msk].sem() for method in method_list]
#%
plt.figure(figsize=[4,3])
intvs = np.arange(nmeth)[:,np.newaxis]
nsamps = sum(imgnet_msk)
xjit = np.random.randn(1, nsamps) * 0.1
plt.plot(0.05 + intvs.repeat(nsamps, 1) + xjit, expalign_tab_imgnt[method_list][imgnet_msk].T,
             color="gray", alpha=0.15)

intvs = np.arange(nmeth, 2*nmeth)[:, np.newaxis]
nsamps = sum(rand_msk)
xjit = np.random.randn(1, nsamps) * 0.1
plt.plot(0.05 + intvs.repeat(nsamps, 1) + xjit, expalign_tab_rnd[method_list][rand_msk].T,
             color="gray", alpha=0.15)
plt.errorbar(range(nmeth), imgnet_mean, yerr=imgnet_sem, capthick=2, capsize=5, lw=3, alpha=0.7)
plt.errorbar(range(nmeth, 2*nmeth), rnd_mean, yerr=rnd_sem, capthick=2, capsize=5, lw=3, alpha=0.7)
plt.xticks(range(2*nmeth), ["ImageNet\n%s"% label for label in label_list] +
                           ["BigGAN rand\n%s"%label for label in label_list])
# plt.xticks(range(2*nmeth), ["ImageNet\nBasinCMA Hess", "ImageNet\nBasinCMA none",] +
#                            ["BigGAN rand\nBasinCMA Hess", "BigGAN rand\nBasinCMA none"])
plt.ylabel("LPIPS Image Dist")

stat_imgnt = ttest_rel(expalign_tab_imgnt[method_list[0]], expalign_tab_imgnt[method_list[1]], nan_policy="omit")
stat_rand = ttest_rel(expalign_tab_rnd[method_list[0]], expalign_tab_rnd[method_list[1]], nan_policy="omit")
dof_imgnt = sum(imgnet_msk) - 1
dof_rand = sum(rand_msk) - 1
plt.title("ImageNet: t=%.1f p=%.1e(dof=%d)\n"
          "BigGAN rand: t=%.1f p=%.1e(dof=%d)"%(stat_imgnt.statistic, stat_imgnt.pvalue, dof_imgnt,
                                                    stat_rand.statistic, stat_rand.pvalue, dof_rand, ))
plt.savefig(join(figdir, "%s_xspace_Hess_cmp.png"%savestr))
plt.savefig(join(figdir, "%s_xspace_Hess_cmp.pdf"%savestr))
# plt.savefig(join(figdir, "BasinCMA_xspace_Hess_cmp.png"))
# plt.savefig(join(figdir, "BasinCMA_xspace_Hess_cmp.pdf"))
# plt.savefig(join(figdir, "Adam_xspace_Hess_cmp.png"))
# plt.savefig(join(figdir, "Adam_xspace_Hess_cmp.pdf"))
plt.show()
#%%

#%%
def var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none',
                      'CMA50Adam0Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_none'],
                labels=['BasinCMA_all', 'BasinCMA_none', 'CMA-Adam_all', 'CMA-Adam_none'],
                 data=None, msk=None, jitter=False, cmap=cm.RdBu, titstr="",
                 ):
    """Designed to plot paired scatter plot for discrete categories. Paired t test is performed at the end and stats
    are returned.
    Input is a pandas dataframe and variable names in it."""
    varn = len(var)
    clist = [cmap(float((vari + .5) / (varn + 1))) for vari in range(varn)]
    fig, ax = plt.subplots(figsize=[6, 8])
    xjit = np.random.randn(data.shape[0]) * 0.1 if jitter else np.zeros(data.shape[0])
    for vari, varnm in enumerate(var):
        plt.scatter(vari + 1 + xjit, data[varnm], s=9, color=clist[vari], alpha=0.6,
                    label=labels[vari])
    plt.legend()
    intvs = np.arange(varn).reshape(1, -1)
    plt.plot(1 + intvs.repeat(data.shape[0], 0).T + xjit[np.newaxis, :], data[var].T,
             color="gray", alpha=0.1)
    plt.xticks(np.arange(len(labels))+1, labels)
    stats = {}
    stats["T01"] = ttest_rel(data[var[0]], data[var[1]], nan_policy='omit')
    stats["T02"] = ttest_rel(data[var[0]], data[var[2]], nan_policy='omit')
    stats["T12"] = ttest_rel(data[var[1]], data[var[2]], nan_policy='omit')
    plt.title(
        "%s\nT: %s - %s:%.1f(%.1e)\n%s - %s:%.1f(%.1e)\n"
        "%s - %s:%.1f(%.1e)" % (titstr, labels[0], labels[1], stats["T01"].statistic, stats["T01"].pvalue,
                                     labels[0], labels[2], stats["T02"].statistic, stats["T02"].pvalue,
                                     labels[1], labels[2], stats["T12"].statistic, stats["T12"].pvalue))
    return fig, stats

def var_stripe_plot(var=[], labels=[], jitter=False, cmap=cm.RdBu, titstr="", tests=[(0,1),(1,2),(2,3)], median=None):
    """Designed to plot paired scatter plot for discrete categories. Paired t test is performed at the end and stats
    are returned.
    Input is a pandas dataframe and variable names in it."""
    varn = len(var)
    clist = [cmap(float((vari + .5) / (varn + 1))) for vari in range(varn)]
    fig, ax = plt.subplots(figsize=[6, 8])

    for vari, varnm in enumerate(var):
        xjit = np.random.randn(len(var[vari])) * 0.1 if jitter else np.zeros(len(var[vari]))
        plt.scatter(vari + 1 + xjit, var[vari], s=9, color=clist[vari], alpha=0.6,
                    label=labels[vari])
    plt.legend()
    ticks = np.arange(varn).reshape(1, -1)
    # plt.plot(1 + ticks.repeat(data.shape[0], 0).T + xjit[np.newaxis, :], data[var].T,
    #          color="gray", alpha=0.1)
    plt.xticks(np.arange(len(labels))+1, labels)
    stats = {}
    medstr = ""
    if median is None: median = list(range(varn))
    for vari in median:
        med = np.nanmedian(var[vari])
        stats["M%d" % vari] = med
        medstr += "%s:%.2f " % (labels[vari], med)
        if (vari+1)%2==0: medstr+="\n"
    statstr = ""
    for pair in tests:
        t_res = ttest_ind(var[pair[0]], var[pair[1]], nan_policy='omit')
        stats["T%d%d" % pair] = t_res
        statstr += "%s - %s:%.1f(%.1e)\n"%(labels[pair[0]], labels[pair[1]], t_res.statistic, t_res.pvalue)
    plt.title(
        "%s\nMed:%s\nT: %s" % (titstr, medstr, statstr))
    return fig, stats
#%%

"""Activation Maximization Cmp"""
rootdir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
summarydir = join(rootdir, "summary")
exprec_tab = pd.read_csv(join(summarydir, "optim_raw_score_tab.csv"))
align_tab = pd.read_csv(join(summarydir, "optim_aligned_score_tab_BigGAN.csv"))
#%%
align_tab_FC6 = pd.read_csv(join(summarydir, "optim_aligned_score_tab_fc6.csv"))
#%%
layers = align_tab_FC6.layer.unique()
optim_list = ["HessCMA", 'CholCMA']
optim_list_fc6 = ['HessCMA500_1_fc6', "CholCMA_fc6"]
colorseq = [cm.jet(i/(len(layers)-1)) for i in range(len(layers))]
plt.figure(figsize=[4,6])
for Li, layer in enumerate(layers):
    xjit = np.random.randn(1) * 0.08
    msk = align_tab_FC6.layer==layer
    optim_fc6_mean = [align_tab_FC6[optim][msk].mean() for optim in optim_list_fc6]
    optim_fc6_sem = [align_tab_FC6[optim][msk].sem() for optim in optim_list_fc6]
    plt.errorbar(xjit+np.arange(2,4), optim_fc6_mean, yerr=optim_fc6_sem,
                 capthick=2, capsize=5, lw=3, alpha=0.55,  color=colorseq[Li])
    msk = align_tab.layer==layer
    optim_mean = [align_tab[optim][msk].mean() for optim in optim_list]
    optim_sem = [align_tab[optim][msk].sem() for optim in optim_list]
    plt.errorbar(xjit+np.arange(0,2), optim_mean, yerr=optim_sem,
                 capthick=2, capsize=5, lw=3, alpha=0.55,  color=colorseq[Li])
optim_fc6_mean = [align_tab_FC6[optim].mean() for optim in optim_list_fc6]
optim_fc6_sem = [align_tab_FC6[optim].sem() for optim in optim_list_fc6]
plt.errorbar(np.arange(2,4), optim_fc6_mean, yerr=optim_fc6_sem,
             capthick=2, capsize=5, lw=3, alpha=0.9,  color='black')
optim_mean = [align_tab[optim].mean() for optim in optim_list]
optim_sem = [align_tab[optim].sem() for optim in optim_list]
plt.errorbar(np.arange(0,2), optim_mean, yerr=optim_sem,
             capthick=2, capsize=5, lw=3, alpha=0.9,  color='black')
plt.ylabel("Unit Activation")
plt.xticks([0, 1, 2, 3], ["BigGAN\nHess CMA", "BigGAN\nChol CMA", "FC6GAN\nHess CMA\n(500d)",
                          "FC6GAN\nChol CMA\n(4096d)", ])

dof_BG = sum((~align_tab[optim_list[0]].isna())*(~align_tab[optim_list[1]].isna())) - 1
dof_FC6 = sum((~align_tab_FC6[optim_list_fc6[0]].isna())*(~align_tab_FC6[optim_list_fc6[1]].isna())) - 1
BG_tstat = ttest_rel(align_tab[optim_list[0]], align_tab[optim_list[1]],nan_policy='omit')
FC6_tstat = ttest_rel(align_tab_FC6[optim_list_fc6[0]], align_tab_FC6[optim_list_fc6[1]],nan_policy='omit')
plt.title("BigGAN cmp: t=%.1f p=%.1e(dof=%d)\n"
          "FC6GAN cmp: t=%.1f p=%.1e(dof=%d)"%(BG_tstat.statistic, BG_tstat.pvalue, dof_BG,
                                               FC6_tstat.statistic, FC6_tstat.pvalue, dof_FC6, ))
plt.savefig(join(figdir, "HessianCMA_ActMax_perlayer_cmp.png"))
plt.savefig(join(figdir, "HessianCMA_ActMax_perlayer_cmp.pdf"))
plt.show()
#%%
layers = align_tab_FC6.layer.unique()
optim_list = ["HessCMA", 'CholCMA']
optim_list_fc6 = ['HessCMA500_1_fc6', "CholCMA_fc6"]
colorseq = [cm.jet(i/(len(layers)-1)) for i in range(len(layers))]
plt.figure(figsize=[4,6])
for Li, layer in enumerate(layers):
    Lmsk = align_tab_FC6.layer==layer
    Ulist_FC6 = align_tab_FC6.unit[Lmsk].unique()
    for Ui, unit in enumerate(Ulist_FC6):
        msk = (align_tab_FC6.layer==layer) * (align_tab_FC6.unit == unit)
        optim_fc6_mean = [align_tab_FC6[optim][msk].mean() for optim in optim_list_fc6]
        optim_fc6_sem = [align_tab_FC6[optim][msk].sem() for optim in optim_list_fc6]
        plt.errorbar(np.arange(2,4), optim_fc6_mean, yerr=optim_fc6_sem,
                     capthick=1, capsize=2, lw=1, alpha=0.4,  color=colorseq[Li])

    Lmsk = align_tab.layer==layer
    Ulist = align_tab.unit[Lmsk].unique()
    for Ui, unit in enumerate(Ulist):
        msk = (align_tab.layer==layer)*(align_tab.unit==unit)
        optim_mean = [align_tab[optim][msk].mean() for optim in optim_list]
        optim_sem = [align_tab[optim][msk].sem() for optim in optim_list]
        plt.errorbar(range(2), optim_mean, yerr=optim_sem,
                     capthick=1, capsize=2, lw=1, alpha=0.4,  color=colorseq[Li])
plt.ylabel("Unit Activation")
plt.xticks([0,1,2,3], ["BigGAN\nHess CMA", "BigGAN\nChol CMA", "FC6GAN\nHess CMA", "FC6GAN\nChol CMA", ])
plt.savefig(join(figdir, "HessianCMA_ActMax_perUnit_cmp.png"))
plt.savefig(join(figdir, "HessianCMA_ActMax_perUnit_cmp.pdf"))
plt.show()
#%%
layers = align_tab_FC6.layer.unique()
optim_list = ["HessCMA", 'CholCMA']
colorseq = [cm.jet(i/(len(layers)-1)) for i in range(len(layers))]
plt.figure(figsize=[3,4])
for Li, layer in enumerate(layers):
    xjit = np.random.randn(1) * 0.08
    msk = align_tab.layer==layer
    optim_mean = [align_tab[optim][msk].mean() for optim in optim_list]
    optim_sem = [align_tab[optim][msk].sem() for optim in optim_list]
    plt.errorbar(xjit+np.arange(0,2), optim_mean, yerr=optim_sem,
                 capthick=2, capsize=5, lw=3, alpha=0.55,  color=colorseq[Li])
optim_mean = [align_tab[optim].mean() for optim in optim_list]
optim_sem = [align_tab[optim].sem() for optim in optim_list]
plt.errorbar(np.arange(0,2), optim_mean, yerr=optim_sem,
             capthick=6, capsize=5, lw=5, alpha=1,  color='black')
plt.ylabel("Unit Activation")
plt.xticks([0, 1,], ["BigGAN\nHess CMA", "BigGAN\nChol CMA", ])
dof_BG = sum((~align_tab[optim_list[0]].isna())*(~align_tab[optim_list[1]].isna())) - 1
BG_tstat = ttest_rel(align_tab[optim_list[0]], align_tab[optim_list[1]],nan_policy='omit')
plt.title("BigGAN cmp: t=%.1f\n p=%.1e(dof=%d)"%(BG_tstat.statistic, BG_tstat.pvalue, dof_BG,))
plt.savefig(join(figdir, "HessianCMA_ActMax_perlayer_cmp_BigGAN.png"))
plt.savefig(join(figdir, "HessianCMA_ActMax_perlayer_cmp_BigGAN.pdf"))
plt.show()

