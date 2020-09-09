#%%
from pytorch_pretrained_biggan import BigGAN, BigGANConfig, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
from skimage.transform import resize, rescale
from imageio import imread, imsave
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sys
import os
from os.path import join
from time import time
from scipy.stats import ttest_ind, ttest_rel
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
#%% Load saved formatted record.
summarydir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet\summary"
# data = np.load(join(savedir, "cmp_exprecord_%d-%d.npz"%(csr_min, csr_max)), )
data = np.load(join(summarydir, "cmp_exprecord.npz"), )
exprecord = data["exprecord"]
exprec_table = pd.DataFrame(exprecord, columns=["imgid", "imgnm", "triali", "dsim_all", "dsim_sep", "dsim_none",
                                               "RNDid"])
#%% Summary figure
exprec_table.to_csv(join(summarydir, r"exprecord_part.csv"))
#%% Load other figures
# pd.read_csv(join(summarydir, r"exprecord_part.csv"), index_col=0)
exprec_table = pd.read_csv(join(summarydir, r"exprecord_all.csv"), index_col=0)
#%%
"""
Draw figures comparing the scores of 
"""
#%%
meanscore = []
minscore = []
for imgid in np.unique(exprec_table.imgid):
    meanscore.append([imgid, exprec_table.dsim_all[exprec_table.imgid == imgid].mean(), exprec_table.dsim_sep[
        exprec_table.imgid == imgid].mean(), exprec_table.dsim_none[exprec_table.imgid == imgid].mean()])
    minscore.append([imgid, exprec_table.dsim_all[exprec_table.imgid == imgid].min(), exprec_table.dsim_sep[
        exprec_table.imgid == imgid].min(), exprec_table.dsim_none[exprec_table.imgid == imgid].min()])

meanscore_table = pd.DataFrame(meanscore, columns=["imgid", "dsim_all", "dsim_sep", "dsim_none", ])
minscore_table = pd.DataFrame(minscore, columns=["imgid", "dsim_all", "dsim_sep", "dsim_none", ])
meanscore = np.array(meanscore)
minscore = np.array(minscore)
#%%
jitter = 0.1*np.random.randn(minscore_table.shape[0])
plt.figure(figsize=[6,8])
plt.plot(np.array([[1, 2, 3]]).T+jitter[np.newaxis, :], minscore[:,1:4].T, color="gray", alpha=0.5)
plt.scatter(1+jitter, minscore_table.dsim_all.array)
plt.scatter(2+jitter, minscore_table.dsim_sep.array)
plt.scatter(3+jitter, minscore_table.dsim_none.array)
plt.ylabel("Dissimilarity Metric (min in 5 seeds)")
plt.xlabel("Basis used")
plt.xticks([1,2,3],["H all", "H sep", "none"])
allnone_t = ttest_rel(minscore_table.dsim_all, minscore_table.dsim_none)
sepnone_t = ttest_rel(minscore_table.dsim_sep, minscore_table.dsim_none)
allsep_t = ttest_rel(minscore_table.dsim_all, minscore_table.dsim_sep)
plt.title("Comparing ADAM performance on different basis of BigGAN space\n"
          "paired-t: all-none:t=%.1f(p=%.1E)\nsep-none:t=%.1f(p=%.1E)\nall-sep:t=%.1f(p=%.1E)"%
          (allnone_t.statistic, allnone_t.pvalue, sepnone_t.statistic, sepnone_t.pvalue,
           allsep_t.statistic, allsep_t.pvalue))
plt.savefig(join(summarydir, "5seeds_min_score.jpg"))
plt.show()
#%%
jitter = 0.1*np.random.randn(meanscore_table.shape[0])
plt.figure(figsize=[6,8])
plt.plot(np.array([[1, 2, 3]]).T+jitter[np.newaxis, :], meanscore[:,1:4].T, color="gray", alpha=0.5)
plt.scatter(1+jitter, meanscore_table.dsim_all.array)
plt.scatter(2+jitter, meanscore_table.dsim_sep.array)
plt.scatter(3+jitter, meanscore_table.dsim_none.array)
plt.ylabel("Dissimilarity Metric (mean in 5 seeds)")
plt.xlabel("Basis used")
plt.xticks([1,2,3],["H all", "H sep", "none"])
allnone_t = ttest_rel(meanscore_table.dsim_all, meanscore_table.dsim_none)
sepnone_t = ttest_rel(meanscore_table.dsim_sep, meanscore_table.dsim_none)
allsep_t = ttest_rel(meanscore_table.dsim_all, meanscore_table.dsim_sep)
plt.title("Comparing ADAM performance on different basis of BigGAN space\n"
          "paired-t: all-none:t=%.1f(p=%.1E)\nsep-none:t=%.1f(p=%.1E)\nall-sep:t=%.1f(p=%.1E)"%
          (allnone_t.statistic, allnone_t.pvalue, sepnone_t.statistic, sepnone_t.pvalue,
           allsep_t.statistic, allsep_t.pvalue))
plt.savefig(join(summarydir, "5seeds_mean_score.jpg"))
plt.show()
#%%
jitter = 0.1*np.random.randn(exprec_table.shape[0])
plt.figure(figsize=[6,8])
plt.plot(np.array([[1, 2, 3]]).T+jitter[np.newaxis, :], exprec_table[['dsim_all','dsim_sep','dsim_none']].to_numpy().T, color="gray", alpha=0.5)
plt.scatter(1+jitter, exprec_table.dsim_all.array)
plt.scatter(2+jitter, exprec_table.dsim_sep.array)
plt.scatter(3+jitter, exprec_table.dsim_none.array)
plt.ylabel("Dissimilarity Metric (all trials)")
plt.xlabel("Basis used")
plt.xticks([1,2,3],["H all", "H sep", "none"])
allnone_t = ttest_rel(exprec_table.dsim_all, exprec_table.dsim_none)
sepnone_t = ttest_rel(exprec_table.dsim_sep, exprec_table.dsim_none)
allsep_t = ttest_rel(exprec_table.dsim_all, exprec_table.dsim_sep)
plt.title("Comparing ADAM performance on different basis of BigGAN space\n"
          "paired-t: all-none:t=%.1f(p=%.1E)\nsep-none:t=%.1f(p=%.1E)\nall-sep:t=%.1f(p=%.1E)"%
          (allnone_t.statistic, allnone_t.pvalue, sepnone_t.statistic, sepnone_t.pvalue,
           allsep_t.statistic, allsep_t.pvalue))
plt.savefig(join(summarydir, "5seeds_trials_score.jpg"))
plt.show()

#%%  Non-matching trials.
"""Newer version of matched samples for BigGAN generated Images. """
from glob import glob
import re
from tqdm import tqdm
datadir = r"E:\Cluster_Backup\BigGAN_invert\BigGAN_rnd"
summarydir = r"E:\Cluster_Backup\BigGAN_invert\BigGAN_rnd\summary"
os.makedirs(summarydir, exist_ok=True)
# BigGAN_rnd_0000_CMA_final711071.jpg
subfds = glob(datadir+"\\CMA*")
methods = [fd.split('\\')[-1] for fd in subfds]
fdpattern = re.compile("CMA(\d*)Adam(\d*)Final(\d*)_postAdam_(.*)")
npzpattern = re.compile("BigGAN_rnd_(\d\d\d\d)optim_data_(\d*).npz") # BigGAN_rnd_0000optim_data_711071.npz
raw_record = []
for method in methods:
    CMAsteps, Adamsteps, Finalsteps, Hspace = fdpattern.findall(method)[0]
    npzlist = glob(join(datadir, method, "BigGAN_rnd_*.npz"))
    npzlist = sorted(npzlist)
    for idx in tqdm(range(len(npzlist))):
        npzname = os.path.split(npzlist[idx])[1]
        parts = npzpattern.findall(npzname)[0]
        imgid = int(parts[0])
        imgnm = "BigGAN_rnd_%04d" % imgid
        RNDid = int(parts[1])
        tmp = np.load(npzlist[idx])
        dsims = np.load(npzlist[idx])['dsims']
        raw_record.append([method, imgid, imgnm, CMAsteps, Adamsteps, Finalsteps, Hspace, np.mean(dsims),
                           np.min(dsims), RNDid])
#%%
raw_rec_tab = pd.DataFrame(raw_record, columns=["method", "imgid", "imgnm", "CMAsteps", "Adamsteps", "Finalsteps",
                                                "space", "dsim_mean", "dsim_min", "RNDid"])
raw_rec_tab.to_csv(join(summarydir, "raw_exprec_tab.csv"))

#%%
imgidx_uniq = np.unique(raw_rec_tab[["imgid"]], axis=0)
written_id = np.array([])
format_tab = []
for rowid in range(imgidx_uniq.shape[0]):
    imgid = imgidx_uniq[rowid, 0]
    imgnm = "BigGAN_rnd_%04d" % imgid
    triali = sum(imgid == written_id)
    mask = (raw_rec_tab.imgid == imgid)
    try:
        score_meth = [raw_rec_tab[mask & (raw_rec_tab.method==method)].dsim_min.mean() for method in methods]
    except ValueError:
        print("Method not matched %d" % (imgid))
        continue
    format_tab.append([imgid, imgnm, triali, *score_meth])
    written_id = np.append(written_id, imgid)
#%%
format_tab = pd.DataFrame(format_tab, columns=["imgid", "imgnm", "triali", *methods])
format_tab.to_csv(join(summarydir, "exprecord_align.csv"))
#%%
from matplotlib import cm
def var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none',
                      'CMA50Adam0Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_none'],
                labels=['BasinCMA_all', 'BasinCMA_none', 'CMA-Adam_all', 'CMA-Adam_none'],
                 data=format_tab, msk=None, jitter=False, cmap=cm.RdBu, titstr="",
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
fig, stats = var_cmp_plot(jitter=True, cmap=cm.jet, titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA w/ w/out Hessian basis")
plt.savefig(join(summarydir, "BigGAN_rand_cmp.png"))
plt.savefig(join(summarydir, "BigGAN_rand_cmp.pdf"))
plt.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none',
                      'CMA50Adam0Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_none',
                      'CMA1Adam30Final600_postAdam_all', 'CMA1Adam30Final600_postAdam_none'],
                labels=['BasinCMA_all', 'BasinCMA_none', 'CMA-Adam_all', 'CMA-Adam_none', "Adam_all", "Adam_none"],
                jitter=True,cmap=cm.jet,titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA w/ w/out Hessian basis")
fig.set_figwidth(8)
fig.savefig(join(summarydir, "BigGAN_rand_all_cmp.png"))
fig.savefig(join(summarydir, "BigGAN_rand_all_cmp.pdf"))
fig.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_all',
                      'CMA1Adam30Final600_postAdam_all', ],
                labels=['BasinCMA_all', 'CMA-Adam_all', "Adam_all",],
                jitter=True,cmap=cm.jet,titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA CMA+Hess ADAM (w/ Hessian)")
fig.savefig(join(summarydir, "BigGAN_rand_HessAll_cmp.png"))
fig.savefig(join(summarydir, "BigGAN_rand_HessAll_cmp.pdf"))
fig.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_none', 'CMA50Adam0Final500_postAdam_none',
                      'CMA1Adam30Final600_postAdam_none', ],
                labels=['BasinCMA_none', 'CMA-Adam_none', "Adam_none",],
                jitter=True,cmap=cm.jet,titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA CMA+Hess ADAM (w/out Hessian)")
fig.savefig(join(summarydir, "BigGAN_rand_noHess_cmp.png"))
fig.savefig(join(summarydir, "BigGAN_rand_noHess_cmp.pdf"))
fig.show()
#%%
from glob import glob
import re
from tqdm import tqdm
datadir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet"
summarydir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet\summary"
os.makedirs(summarydir, exist_ok=True)
# BigGAN_rnd_0000_CMA_final711071.jpg
subfds = glob(datadir+"\\CMA*")
methods = [fd.split('\\')[-1] for fd in subfds]
fdpattern = re.compile("CMA(\d*)Adam(\d*)Final(\d*)_postAdam_(.*)")
INnpzpattern = re.compile("val_crop_(\d\d\d\d\d\d\d\d)optim_data_(\d*).npz") # BigGAN_rnd_0000optim_data_711071.npz
raw_record = []
for method in methods:
    CMAsteps, Adamsteps, Finalsteps, Hspace = fdpattern.findall(method)[0]
    npzlist = glob(join(datadir, method, "val_crop_*.npz"))
    npzlist = sorted(npzlist)
    for idx in tqdm(range(len(npzlist))):
        npzname = os.path.split(npzlist[idx])[1]
        parts = INnpzpattern.findall(npzname)[0]
        imgid = int(parts[0])
        imgnm = "val_crop_%08d" % imgid
        RNDid = int(parts[1])
        tmp = np.load(npzlist[idx])
        dsims = np.load(npzlist[idx])['dsims']
        raw_record.append([method, imgid, imgnm, CMAsteps, Adamsteps, Finalsteps, Hspace, np.mean(dsims),
                           np.min(dsims), RNDid])
#%%
INraw_rec_tab = pd.DataFrame(raw_record, columns=["method", "imgid", "imgnm", "CMAsteps", "Adamsteps", "Finalsteps",
                                                "space", "dsim_mean", "dsim_min", "RNDid"])
INraw_rec_tab.to_csv(join(summarydir, "ImageNet_raw_exprec_tab.csv"))
#%%
imgidx_uniq = np.unique(INraw_rec_tab[["imgid"]], axis=0)
written_id = np.array([])
INformat_tab = []
for rowid in range(imgidx_uniq.shape[0]):
    imgid = imgidx_uniq[rowid, 0]
    imgnm = "val_crop_%08d" % imgid
    triali = sum(imgid == written_id)
    mask = (INraw_rec_tab.imgid == imgid)
    try:
        score_meth = [INraw_rec_tab[mask & (raw_rec_tab.method==method)].dsim_min.mean() for method in methods]
    except ValueError:
        print("Method not matched %d" % (imgid))
        continue
    INformat_tab.append([imgid, imgnm, triali, *score_meth])
    written_id = np.append(written_id, imgid)
#%
INformat_tab = pd.DataFrame(INformat_tab, columns=["imgid", "imgnm", "triali", *methods])
INformat_tab.to_csv(join(summarydir, "ImageNet_exprecord_align.csv"))
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_none', 'CMA50Adam0Final500_postAdam_none',
                      'CMA1Adam30Final600_postAdam_none', ],
                labels=['BasinCMA_none', 'CMA-Adam_none', "Adam_none",], data=INformat_tab,
                jitter=True, cmap=cm.jet, titstr="BigGAN inversion score for ImageNet images\n for "
                                                           "BasinCMA CMA+Hess ADAM (w/out Hessian)")
fig.savefig(join(summarydir, "ImageNet_noHess_cmp.png"))
fig.savefig(join(summarydir, "ImageNet_noHess_cmp.pdf"))
fig.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_all',
                      'CMA1Adam30Final600_postAdam_all', ],
                labels=['BasinCMA_all', 'CMA-Adam_all', "Adam_all",], data=INformat_tab,
                jitter=True, cmap=cm.jet, titstr="BigGAN inversion score for ImageNet images\n for "
                                                           "BasinCMA CMA+Hess ADAM (with Hessian)")
fig.savefig(join(summarydir, "ImageNet_allHess_cmp.png"))
fig.savefig(join(summarydir, "ImageNet_allHess_cmp.pdf"))
fig.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none',
                      'CMA50Adam0Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_none',
                      'CMA1Adam30Final600_postAdam_all', 'CMA1Adam30Final600_postAdam_none'], data=INformat_tab,
                labels=['BasinCMA_all', 'BasinCMA_none', 'CMA-Adam_all', 'CMA-Adam_none', "Adam_all", "Adam_none"],
                jitter=True,cmap=cm.jet,titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA CMA+ADAM ADAM w/ w/out Hessian basis")
fig.set_figwidth(8)
fig.savefig(join(summarydir, "ImageNet_all_cmp.png"))
fig.savefig(join(summarydir, "ImageNet_all_cmp.pdf"))
fig.show()
#%%
fig, stats = var_cmp_plot(var=['CMA10Adam10Final500_postAdam_all', 'CMA10Adam10Final500_postAdam_none',
                      'CMA50Adam0Final500_postAdam_all', 'CMA50Adam0Final500_postAdam_none',], data=INformat_tab,
                labels=['BasinCMA_all', 'BasinCMA_none', 'CMA-Adam_all', 'CMA-Adam_none',],
                jitter=True, cmap=cm.jet, titstr="BigGAN inversion score for BigGAN rand image\n for "
                                                           "BasinCMA CMA+ADAM w/ w/out Hessian basis")
plt.savefig(join(summarydir, "ImageNet_major_cmp.png"))
plt.savefig(join(summarydir, "ImageNet_major_cmp.pdf"))
plt.show()
#%%
fig, stats = var_stripe_plot(var=[format_tab['CMA10Adam10Final500_postAdam_all'], format_tab[
    'CMA10Adam10Final500_postAdam_none'],
               INformat_tab['CMA10Adam10Final500_postAdam_all'], INformat_tab['CMA10Adam10Final500_postAdam_none'],],
                labels=["GAN_Basin_All", "GAN_Basin_None", "Nat_Basin_All", "Nat_Basin_None"] , jitter=True,
              cmap=cm.jet, tests=[(0,1),(2,3),(0,2),(1,3)],titstr="Fitting Score Comparison Across Spaces (BigGAN and ImageNet)")
plt.savefig(join(summarydir, "BigGAN_ImageNet_cmp.png"))
plt.savefig(join(summarydir, "BigGAN_ImageNet_cmp.pdf"))
fig.show()
