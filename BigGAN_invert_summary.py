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

#%%
#%%  Sort out the npz to be analyzed.
#    Format the information in a table like  "exprecord_all.csv"
datadir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet"
from glob import glob
import re
npzpattern = re.compile("val_crop_(\d\d\d\d\d\d\d\d)_code_H(.*)reg(\d\d\d\d\d\d)")
npzlist = glob(join(datadir, "val_crop_*.npz"))
npzlist = sorted(npzlist)
raw_record = []
for idx in range(len(npzlist)):
    npzname = os.path.split(npzlist[idx])[1]
    parts = npzpattern.findall(npzname)[0]
    imgid = int(parts[0])
    imgnm = "val_crop_%08d" % imgid
    space = parts[1]
    RNDid = int(parts[2])
    tmp = np.load(npzlist[idx])
    dsim = np.load(npzlist[idx])['dsim']
    raw_record.append([imgid, imgnm, space, dsim.item(), RNDid])
#%%
raw_rec_tab = pd.DataFrame(raw_record, columns=["imgid", "imgnm", "space", "dsim", "RNDid"])
#%%
imgRND_idx = raw_rec_tab[["imgid", "RNDid"]].to_numpy()
imgRND_uniqidx = np.unique(imgRND_idx, axis=0)
written_id = np.array([])
format_tab = []
for rowid in range(imgRND_uniqidx.shape[0]):
    imgid = imgRND_uniqidx[rowid, 0]
    imgnm = "val_crop_%08d" % imgid
    RNDid = imgRND_uniqidx[rowid, 1]
    triali = sum(imgid == written_id)
    mask = (raw_rec_tab.imgid == imgid) & (raw_rec_tab.RNDid == RNDid)
    try:
        dsim_all = raw_rec_tab[mask & (raw_rec_tab.space=="all")].dsim.item()
        dsim_sep = raw_rec_tab[mask & (raw_rec_tab.space=="sep")].dsim.item()
        dsim_none = raw_rec_tab[mask & (raw_rec_tab.space=="none")].dsim.item()
    except ValueError:
        print("Imcomplete Entry %d (RND %d)" % (imgid, RNDid))
        continue
    format_tab.append([imgid, imgnm, triali, dsim_all, dsim_sep, dsim_none, RNDid])
    written_id = np.append(written_id, imgid)

#%%
format_tab = pd.DataFrame(format_tab, columns=["imgid", "imgnm", "triali", 'dsim_all', 'dsim_sep', 'dsim_none', "RNDid"])
format_tab.to_csv(join(summarydir, "exprecord_all.csv"))