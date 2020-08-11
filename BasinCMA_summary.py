import cma
import tqdm
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from os.path import join
from imageio import imread
from scipy.linalg import block_diag
#%%
rootdir = r"E:\Cluster_Backup\BasinCMA"
summarydir = r"E:\Cluster_Backup\BasinCMA\summary"
dataset = "BigGAN_rnd" # "ImageNet"
settings = os.listdir(join(rootdir, dataset))
#%%
tables = []
for si, setting in enumerate(settings):
    print(setting)
    fns = os.listdir(join(rootdir, dataset, setting))
    info_fns = [fn for fn in fns if "traj" in fn]
    #%
    entry_list = []
    namepatt = re.compile("(.*)traj_H(.*)_postAdam_(\d*)_dsim_([.\d]*)_L1_([.\d]*).jpg")
    for i, fn in enumerate(info_fns):
        result = namepatt.findall(info_fns[i])
        if len(result) > 0:
            result = result[0]
            entry = [result[0], result[1], int(result[2]), float(result[3]), float(result[4])]
            entry_list.append(entry)

    recordtable = pd.DataFrame(entry_list, columns=["Img", "Hbasis", "RND", "dsim", "L1"])
    recordtable.to_csv(join(rootdir, dataset, setting, "expRecord.csv"))
    tables.append(recordtable.copy())
#%%
_, subsp_mask = np.unique(tables[0].Img, return_index=True)
tables[0] = tables[0].iloc[subsp_mask, :]
tables[1] = tables[1].iloc[subsp_mask, :]
#%%
from scipy.stats import ttest_rel,ttest_ind
ttest_rel(tables[4].dsim, tables[5].dsim)
ttest_ind(tables[4].dsim, tables[5].dsim)
ttest_rel(tables[0].dsim, tables[1].dsim)
ttest_ind(tables[0].dsim, tables[1].dsim)
ttest_rel(tables[2].dsim, tables[3].dsim)
ttest_ind(tables[2].dsim, tables[3].dsim)
#%%

#%%
jitter = 0.1*np.random.randn(tables[0].shape[0])
plt.figure(figsize=[8,9])
plt.plot(np.array([[1, 2, ]]).T+jitter[np.newaxis, :], np.array([tables[0].dsim, tables[1].dsim]),
         color="gray", alpha=0.3)
plt.plot(np.array([[3, 4, ]]).T+jitter[np.newaxis, :], np.array([tables[2].dsim, tables[3].dsim]),
         color="gray", alpha=0.3)
plt.plot(np.array([[5, 6, ]]).T+jitter[np.newaxis, :], np.array([tables[4].dsim, tables[5].dsim]),
         color="gray", alpha=0.3)
plt.scatter(1+jitter, tables[0].dsim, label=settings[0])
plt.scatter(2+jitter, tables[1].dsim, label=settings[1])
plt.scatter(3+jitter, tables[2].dsim, label=settings[2])
plt.scatter(4+jitter, tables[3].dsim, label=settings[3])
plt.scatter(5+jitter, tables[4].dsim, label=settings[4])
plt.scatter(6+jitter, tables[5].dsim, label=settings[5])
# plt.scatter(3+jitter, exprec_table.dsim_none.array)
plt.ylabel("dssim (min value)")
plt.xlabel("Algorithm to Invert GAN")
# plt.legend()
plt.xticks([1,2,3,4,5,6], ["BasinCMA H all", "BasinCMA none", "Adam try H all", "Adam try none", "CMA H all", "CMA none"])
BasinCMA_H_cmp_t = ttest_rel(tables[0].dsim, tables[1].dsim)
Adam_H_cmp_t = ttest_rel(tables[2].dsim, tables[3].dsim)
CMA_H_cmp_t = ttest_rel(tables[4].dsim, tables[5].dsim)
BasinCMA_cma_cmp_t = ttest_rel(tables[0].dsim, tables[4].dsim)
BasinCMA_Adam_cmp_t = ttest_rel(tables[0].dsim, tables[2].dsim)
BasinCMA_cma_Hnone_cmp_t = ttest_rel(tables[1].dsim, tables[5].dsim)
BasinCMA_Adam_Hnone_cmp_t = ttest_rel(tables[1].dsim, tables[3].dsim)
plt.title("Comparing ADAM performance on different basis with Different Algorithm\n(Fitting BigGAN random images)\n"
          "H all-none paired-t: BasinCMA:t=%.1f(p=%.1E)\nAdam try:t=%.1f(p=%.1E) CMA:t=%.1f(p=%.1E)\nBasinCMA-CMA:t=%.1f("
          "p=%.1E) BasinCMA-Adam:t=%.1f(p=%.1E)"%
          (BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue, Adam_H_cmp_t.statistic, Adam_H_cmp_t.pvalue,
           CMA_H_cmp_t.statistic, CMA_H_cmp_t.pvalue, BasinCMA_cma_cmp_t.statistic, BasinCMA_cma_cmp_t.pvalue, BasinCMA_Adam_cmp_t.statistic, BasinCMA_Adam_cmp_t.pvalue))
plt.savefig(join(summarydir, "BigGAN_rnd_min_score_cmp_noleg.jpg"))
plt.show()

#%% L1 version
jitter = 0.1*np.random.randn(tables[0].shape[0])
plt.figure(figsize=[8,9])
plt.plot(np.array([[1, 2, ]]).T+jitter[np.newaxis, :], np.array([tables[0].L1, tables[1].L1]),
         color="gray", alpha=0.3)
plt.plot(np.array([[3, 4, ]]).T+jitter[np.newaxis, :], np.array([tables[2].L1, tables[3].L1]),
         color="gray", alpha=0.3)
plt.plot(np.array([[5, 6, ]]).T+jitter[np.newaxis, :], np.array([tables[4].L1, tables[5].L1]),
         color="gray", alpha=0.3)
plt.scatter(1+jitter, tables[0].L1, label=settings[0])
plt.scatter(2+jitter, tables[1].L1, label=settings[1])
plt.scatter(3+jitter, tables[2].L1, label=settings[2])
plt.scatter(4+jitter, tables[3].L1, label=settings[3])
plt.scatter(5+jitter, tables[4].L1, label=settings[4])
plt.scatter(6+jitter, tables[5].L1, label=settings[5])
# plt.scatter(3+jitter, exprec_table.L1_none.array)
plt.ylabel("L1 distance (min value)")
plt.xlabel("Algorithm to Invert GAN")
plt.legend()
plt.xticks([1,2,3,4,5,6], ["BasinCMA H all", "BasinCMA none", "Adam try H all", "Adam try none", "CMA H all", "CMA none"])
BasinCMA_H_cmp_t = ttest_rel(tables[0].L1, tables[1].L1)
Adam_H_cmp_t = ttest_rel(tables[2].L1, tables[3].L1)
CMA_H_cmp_t = ttest_rel(tables[4].L1, tables[5].L1)
BasinCMA_cma_cmp_t = ttest_rel(tables[0].L1, tables[4].L1)
BasinCMA_Adam_cmp_t = ttest_rel(tables[0].L1, tables[2].L1)
BasinCMA_cma_Hnone_cmp_t = ttest_rel(tables[1].L1, tables[5].L1)
BasinCMA_Adam_Hnone_cmp_t = ttest_rel(tables[1].L1, tables[3].L1)
plt.title("Comparing ADAM performance on different basis with Different Algorithm\n(Fitting BigGAN random images)\n"
          "H all-none paired-t: BasinCMA:t=%.1f(p=%.1E)\nAdam try:t=%.1f(p=%.1E) CMA:t=%.1f(p=%.1E)\nBasinCMA-CMA:t=%.1f("
          "p=%.1E) BasinCMA-Adam:t=%.1f(p=%.1E)"%
          (BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue, Adam_H_cmp_t.statistic, Adam_H_cmp_t.pvalue,
           CMA_H_cmp_t.statistic, CMA_H_cmp_t.pvalue, BasinCMA_cma_cmp_t.statistic, BasinCMA_cma_cmp_t.pvalue, BasinCMA_Adam_cmp_t.statistic, BasinCMA_Adam_cmp_t.pvalue))
plt.savefig(join(summarydir, "BigGAN_rnd_min_L1_cmp_noleg.jpg"))
plt.show()


#%% ImageNet Dataset
dataset = "ImageNet" # "ImageNet"
settings_imgnet = os.listdir(join(rootdir, dataset))
#%%
tables_imgnet = []
for si, setting in enumerate(settings_imgnet):
    fns = os.listdir(join(rootdir, dataset, setting))
    info_fns = [fn for fn in fns if "traj" in fn]
    #%
    entry_list = []
    namepatt = re.compile("(.*)traj_H(.*)_postAdam_(\d*)_dsim_([.\d]*)_L1_([.\d]*).jpg")
    for i, fn in enumerate(info_fns):
        result = namepatt.findall(info_fns[i])
        if len(result) > 0:
            result = result[0]
            entry = [result[0], result[1], int(result[2]), float(result[3]), float(result[4])]
            entry_list.append(entry)

    recordtable = pd.DataFrame(entry_list, columns=["Img", "Hbasis", "RND", "dsim", "L1"])
    recordtable.to_csv(join(rootdir, dataset, setting, "expRecord.csv"))
    tables_imgnet.append(recordtable.copy())
    print(setting, "record table shape (%d,%d)"%recordtable.shape)
#%%
_, subsp_mask2 = np.unique(tables_imgnet[0].Img, return_index=True)
tables_imgnet[0] = tables_imgnet[0].iloc[subsp_mask2, :]
tables_imgnet[1] = tables_imgnet[1].iloc[subsp_mask2, :]
#%%
#%% dsim version
jitter = 0.1*np.random.randn(tables_imgnet[0].shape[0])
plt.figure(figsize=[10,9])
plt.plot(np.array([[1, 2, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[0].dsim, tables_imgnet[1].dsim]),
         color="gray", alpha=0.3)
plt.plot(np.array([[3, 4, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[2].dsim, tables_imgnet[3].dsim]),
         color="gray", alpha=0.3)
plt.plot(np.array([[5, 6, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[4].dsim, tables_imgnet[5].dsim]),
         color="gray", alpha=0.3)
plt.plot(np.array([[7, 8, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[6].dsim, tables_imgnet[7].dsim]),
         color="gray", alpha=0.3)
plt.scatter(1+jitter, tables_imgnet[0].dsim, label=settings_imgnet[0])
plt.scatter(2+jitter, tables_imgnet[1].dsim, label=settings_imgnet[1])
plt.scatter(3+jitter, tables_imgnet[2].dsim, label=settings_imgnet[2])
plt.scatter(4+jitter, tables_imgnet[3].dsim, label=settings_imgnet[3])
plt.scatter(5+jitter, tables_imgnet[4].dsim, label=settings_imgnet[4])
plt.scatter(6+jitter, tables_imgnet[5].dsim, label=settings_imgnet[5])
plt.scatter(7+jitter, tables_imgnet[6].dsim, label=settings_imgnet[6])
plt.scatter(8+jitter, tables_imgnet[7].dsim, label=settings_imgnet[7])
# plt.scatter(3+jitter, exprec_table.dsim_none.array)
plt.ylabel("dssim (min value)")
plt.xlabel("Algorithm to Invert GAN")
plt.legend()
plt.xticks([1,2,3,4,5,6,7,8], ["BasinCMA H all", "BasinCMA none", "Adam H all", "Adam none", "Adam try H all",
                               "Adam try none", "CMA H all", "CMA none"])
BasinCMA_H_cmp_t = ttest_rel(tables_imgnet[0].dsim, tables_imgnet[1].dsim)
Adam_H_cmp_t = ttest_rel(tables_imgnet[4].dsim, tables_imgnet[5].dsim)
CMA_H_cmp_t = ttest_rel(tables_imgnet[6].dsim, tables_imgnet[7].dsim)
BasinCMA_cma_cmp_t = ttest_rel(tables_imgnet[0].dsim, tables_imgnet[6].dsim)
BasinCMA_Adam_cmp_t = ttest_rel(tables_imgnet[0].dsim, tables_imgnet[4].dsim)
BasinCMA_cma_Hnone_cmp_t = ttest_rel(tables_imgnet[1].dsim, tables_imgnet[7].dsim)
BasinCMA_Adam_Hnone_cmp_t = ttest_rel(tables_imgnet[1].dsim, tables_imgnet[5].dsim)
plt.title("Comparing ADAM performance on different basis with Different Algorithm\n(Fitting ImageNet "
          "images)\n"
          "H all-none paired-t: BasinCMA:t=%.1f(p=%.1E)\nAdam try:t=%.1f(p=%.1E) CMA:t=%.1f("
          "p=%.1E)\nBasinCMA-CMA:t=%.1f("
          "p=%.1E) BasinCMA-Adam:t=%.1f(p=%.1E)"%
          (BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue, Adam_H_cmp_t.statistic, Adam_H_cmp_t.pvalue,
           CMA_H_cmp_t.statistic, CMA_H_cmp_t.pvalue, BasinCMA_cma_cmp_t.statistic, BasinCMA_cma_cmp_t.pvalue, BasinCMA_Adam_cmp_t.statistic, BasinCMA_Adam_cmp_t.pvalue))
plt.savefig(join(summarydir, "ImageNet_min_score_cmp.jpg")) # _noleg
plt.show()
#%%
# L1 distance version
jitter = 0.1*np.random.randn(tables_imgnet[0].shape[0])
plt.figure(figsize=[10,9])
plt.plot(np.array([[1, 2, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[0].L1, tables_imgnet[1].L1]),
         color="gray", alpha=0.3)
plt.plot(np.array([[3, 4, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[2].L1, tables_imgnet[3].L1]),
         color="gray", alpha=0.3)
plt.plot(np.array([[5, 6, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[4].L1, tables_imgnet[5].L1]),
         color="gray", alpha=0.3)
plt.plot(np.array([[7, 8, ]]).T+jitter[np.newaxis, :], np.array([tables_imgnet[6].L1, tables_imgnet[7].L1]),
         color="gray", alpha=0.3)
plt.scatter(1+jitter, tables_imgnet[0].L1, label=settings_imgnet[0])
plt.scatter(2+jitter, tables_imgnet[1].L1, label=settings_imgnet[1])
plt.scatter(3+jitter, tables_imgnet[2].L1, label=settings_imgnet[2])
plt.scatter(4+jitter, tables_imgnet[3].L1, label=settings_imgnet[3])
plt.scatter(5+jitter, tables_imgnet[4].L1, label=settings_imgnet[4])
plt.scatter(6+jitter, tables_imgnet[5].L1, label=settings_imgnet[5])
plt.scatter(7+jitter, tables_imgnet[6].L1, label=settings_imgnet[6])
plt.scatter(8+jitter, tables_imgnet[7].L1, label=settings_imgnet[7])
# plt.scatter(3+jitter, exprec_table.L1_none.array)
plt.ylabel("L1 distance (min value)")
plt.xlabel("Algorithm to Invert GAN")
# plt.legend()
plt.xticks([1,2,3,4,5,6,7,8], ["BasinCMA H all", "BasinCMA none", "Adam H all", "Adam none", "Adam try H all",
                               "Adam try none", "CMA H all", "CMA none"])
BasinCMA_H_cmp_t = ttest_rel(tables_imgnet[0].L1, tables_imgnet[1].L1)
Adam_H_cmp_t = ttest_rel(tables_imgnet[4].L1, tables_imgnet[5].L1)
CMA_H_cmp_t = ttest_rel(tables_imgnet[6].L1, tables_imgnet[7].L1)
BasinCMA_cma_cmp_t = ttest_rel(tables_imgnet[0].L1, tables_imgnet[6].L1)
BasinCMA_Adam_cmp_t = ttest_rel(tables_imgnet[0].L1, tables_imgnet[4].L1)
BasinCMA_cma_Hnone_cmp_t = ttest_rel(tables_imgnet[1].L1, tables_imgnet[7].L1)
BasinCMA_Adam_Hnone_cmp_t = ttest_rel(tables_imgnet[1].L1, tables_imgnet[5].L1)
plt.title("Comparing ADAM performance on different basis with Different Algorithm\n(Fitting ImageNet "
          "images)\n"
          "H all-none paired-t: BasinCMA:t=%.1f(p=%.1E)\nAdam try:t=%.1f(p=%.1E) CMA:t=%.1f("
          "p=%.1E)\nBasinCMA-CMA:t=%.1f("
          "p=%.1E) BasinCMA-Adam:t=%.1f(p=%.1E)"%
          (BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue, Adam_H_cmp_t.statistic, Adam_H_cmp_t.pvalue,
           CMA_H_cmp_t.statistic, CMA_H_cmp_t.pvalue, BasinCMA_cma_cmp_t.statistic, BasinCMA_cma_cmp_t.pvalue, BasinCMA_Adam_cmp_t.statistic, BasinCMA_Adam_cmp_t.pvalue))
plt.savefig(join(summarydir, "ImageNet_min_L1_cmp_noleg.jpg")) #
plt.show()
#%% Cross Dataset Comparison
offset = 0.4
jitter = 0.1*np.random.randn(tables[0].shape[0])
jitter_imgnet = 0.1*np.random.randn(tables_imgnet[0].shape[0])
plt.figure(figsize=[10, 9])
plt.plot(np.array([[0, 1, ]]).T+jitter[np.newaxis, :], np.array([tables[0].dsim, tables[1].dsim]),
         color="gray", alpha=0.2)
plt.plot(np.array([[2, 3, ]]).T+jitter_imgnet[np.newaxis, :], np.array([tables_imgnet[0].dsim, tables_imgnet[1].dsim]),
         color="gray", alpha=0.2)

plt.scatter(0+jitter, tables[0].dsim, label=settings[0], alpha=0.4)
plt.scatter(1+jitter, tables[1].dsim, label=settings[1], alpha=0.4)
plt.scatter(2+jitter_imgnet, tables_imgnet[0].dsim, label=settings_imgnet[0], alpha=0.4)
plt.scatter(3+jitter_imgnet, tables_imgnet[1].dsim, label=settings_imgnet[1], alpha=0.4)

plt.errorbar(0-offset, tables[0].dsim.mean(), tables[0].dsim.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(1-offset, tables[1].dsim.mean(), tables[1].dsim.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(2-offset, tables_imgnet[0].dsim.mean(), tables_imgnet[0].dsim.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(3-offset, tables_imgnet[1].dsim.mean(), tables_imgnet[1].dsim.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.xticks([0, 1, 2, 3], ["BasinCMA H all\nBigGAN random", "BasinCMA none\nBigGAN random", "BasinCMA H all\nImageNet",
                        "BasinCMA none\nImageNet"])
plt.ylabel("dssim (min value)")
plt.xlabel("Algorithm x Dataset")
BasinCMA_H_cmp_t = ttest_rel(tables[0].dsim, tables[1].dsim)
BasinCMA_H_cmp_imgnet_t = ttest_rel(tables_imgnet[0].dsim, tables_imgnet[1].dsim)
BigGAN_imgnet_cmp_t = ttest_ind(tables[0].dsim, tables_imgnet[0].dsim)
BigGAN_imgnet_Hnone_cmp_t = ttest_ind(tables[1].dsim, tables_imgnet[1].dsim)
plt.title("Comparing BasinCMA performance on different basis of BigGAN space with Different Algorithm \n In fitting "
          "BigGAN random images and ImageNet images\n"
          "BigGAN random: H all: %.3f(%.3f), H none: %.3f(%.3f), t=%.1f(p=%.1E)\n"
          "ImageNet: H all: %.3f(%.3f), H none: %.3f(%.3f), t=%.1f(p=%.1E)\n"
          "BigGAN - ImageNet: H all: t=%.1f(p=%.1E), H none: t=%.1f(p=%.1E)"
          %(tables[0].dsim.mean(), tables[0].dsim.sem(), tables[1].dsim.mean(), tables[1].dsim.sem(),
            BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue,
            tables_imgnet[0].dsim.mean(), tables_imgnet[0].dsim.sem(), tables_imgnet[1].dsim.mean(), tables_imgnet[1].dsim.sem(),
            BasinCMA_H_cmp_imgnet_t.statistic, BasinCMA_H_cmp_imgnet_t.pvalue,
            BigGAN_imgnet_cmp_t.statistic, BigGAN_imgnet_cmp_t.pvalue, BigGAN_imgnet_Hnone_cmp_t.statistic,
            BigGAN_imgnet_Hnone_cmp_t.pvalue ))
plt.savefig(join(summarydir, "ImageNet_BigGAN_rand_score_cmp.jpg")) # _noleg
plt.show()
#%%
#%% L1 version
offset = 0.4
jitter = 0.1*np.random.randn(tables[0].shape[0])
jitter_imgnet = 0.1*np.random.randn(tables_imgnet[0].shape[0])
plt.figure(figsize=[10, 9])
plt.plot(np.array([[0, 1, ]]).T+jitter[np.newaxis, :], np.array([tables[0].L1, tables[1].L1]),
         color="gray", alpha=0.2)
plt.plot(np.array([[2, 3, ]]).T+jitter_imgnet[np.newaxis, :], np.array([tables_imgnet[0].L1, tables_imgnet[1].L1]),
         color="gray", alpha=0.2)

plt.scatter(0+jitter, tables[0].L1, label=settings[0], alpha=0.4)
plt.scatter(1+jitter, tables[1].L1, label=settings[1], alpha=0.4)
plt.scatter(2+jitter_imgnet, tables_imgnet[0].L1, label=settings_imgnet[0], alpha=0.4)
plt.scatter(3+jitter_imgnet, tables_imgnet[1].L1, label=settings_imgnet[1], alpha=0.4)

plt.errorbar(0-offset, tables[0].L1.mean(), tables[0].L1.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(1-offset, tables[1].L1.mean(), tables[1].L1.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(2-offset, tables_imgnet[0].L1.mean(), tables_imgnet[0].L1.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.errorbar(3-offset, tables_imgnet[1].L1.mean(), tables_imgnet[1].L1.sem(), marker='o', markersize=7, capsize=15, capthick=3)
plt.xticks([0, 1, 2, 3], ["BasinCMA H all\nBigGAN random", "BasinCMA none\nBigGAN random", "BasinCMA H all\nImageNet",
                        "BasinCMA none\nImageNet"])
plt.ylabel("L1 distance (min value)")
plt.xlabel("Algorithm x Dataset")
BasinCMA_H_cmp_t = ttest_rel(tables[0].L1, tables[1].L1)
BasinCMA_H_cmp_imgnet_t = ttest_rel(tables_imgnet[0].L1, tables_imgnet[1].L1)
BigGAN_imgnet_cmp_t = ttest_ind(tables[0].L1, tables_imgnet[0].L1)
BigGAN_imgnet_Hnone_cmp_t = ttest_ind(tables[1].L1, tables_imgnet[1].L1)
plt.title("Comparing BasinCMA performance on different basis of BigGAN space with Different Algorithm \n In fitting "
          "BigGAN random images and ImageNet images\n"
          "BigGAN random: H all: %.3f(%.3f), H none: %.3f(%.3f), t=%.1f(p=%.1E)\n"
          "ImageNet: H all: %.3f(%.3f), H none: %.3f(%.3f), t=%.1f(p=%.1E)\n"
          "BigGAN - ImageNet: H all: t=%.1f(p=%.1E), H none: t=%.1f(p=%.1E)"
          %(tables[0].L1.mean(), tables[0].L1.sem(), tables[1].L1.mean(), tables[1].L1.sem(),
            BasinCMA_H_cmp_t.statistic, BasinCMA_H_cmp_t.pvalue,
            tables_imgnet[0].L1.mean(), tables_imgnet[0].L1.sem(), tables_imgnet[1].L1.mean(), tables_imgnet[1].L1.sem(),
            BasinCMA_H_cmp_imgnet_t.statistic, BasinCMA_H_cmp_imgnet_t.pvalue,
            BigGAN_imgnet_cmp_t.statistic, BigGAN_imgnet_cmp_t.pvalue, BigGAN_imgnet_Hnone_cmp_t.statistic,
            BigGAN_imgnet_Hnone_cmp_t.pvalue ))
plt.savefig(join(summarydir, "ImageNet_BigGAN_rand_L1_cmp.jpg")) # _noleg
plt.show()