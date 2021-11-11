"""Data analysis for resize evolution """

import os
from os.path import join
from easydict import EasyDict
import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_ind, ttest_rel
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dataroot = r"E:\Cluster_Backup\CNN_manifold"
sumdir = r"E:\Cluster_Backup\CNN_manifold\summary"
os.makedirs(sumdir,exist_ok=True)
netname = "vgg16"
#%% Collect the statistics for the VGG evolution!
unit_list = [("vgg16", "conv2", 5, 112, 112), 
            ("vgg16", "conv3", 5, 56, 56), 
            ("vgg16", "conv4", 5, 56, 56), 
            ("vgg16", "conv5", 5, 28, 28), 
            ("vgg16", "conv6", 5, 28, 28), 
            ("vgg16", "conv7", 5, 28, 28), 
            # ("vgg16", "conv9", 5, 14, 14),
            ("vgg16", "conv10", 5, 14, 14), 
            ("vgg16", "conv12", 5, 7, 7), 
            ("vgg16", "conv13", 5, 7, 7)]
subsp_nm = ["PC23","PC2526","PC4950","RND12"]
RecAll = []
unit = ("vgg16", "conv10", 5, 14, 14)
for unit in unit_list:
    subfolder = "%s_%s_manifold"%(unit[0], unit[1])
    layerdir = join(dataroot, subfolder)
    for iCh in range(1, 51):
        unitRec = EasyDict()
        unit_lab = "%s_%d_%d_%d" % (unit[1], iCh, unit[3], unit[4])
        unitRec.update({"Layer":unit[1], "chan":iCh, "pos_x": unit[3], "pos_y": unit[4]})
        Edata = np.load(join(layerdir, "Manifold_set_%s_orig.npz"%unit_lab))
        #['Perturb_vec', 'imgsize', 'corner', 'evol_score', 'evol_gen']
        Mdata = np.load(join(layerdir, "Manifold_score_%s_orig.npy"%unit_lab))
        Edata_rf = np.load(join(layerdir, "Manifold_set_%s_rf_fit.npz"%unit_lab))
        Mdata_rf = np.load(join(layerdir, "Manifold_score_%s_rf_fit.npy"%unit_lab))
        lastgenscore = Edata["evol_score"][Edata["evol_gen"] == Edata["evol_gen"].max()]
        lastgenscore_rf = Edata_rf["evol_score"][Edata_rf["evol_gen"] == Edata_rf["evol_gen"].max()]
        unitRec.manifMax = Mdata.max(axis=(1,2))
        unitRec.manifMax_rf = Mdata_rf.max(axis=(1,2))
        unitRec.evolast_m = lastgenscore.mean()
        unitRec.evolast_std = lastgenscore.std()
        unitRec.evolast_m_rf = lastgenscore_rf.mean()
        unitRec.evolast_std_rf = lastgenscore_rf.std()
        RecAll.append(unitRec)

record_all = pd.DataFrame.from_records(RecAll)
#%%
record_all.to_csv(join(sumdir, "%s_conv_RFfit_rec.csv"%netname))
#%%
import matplotlib.pylab as plt
import seaborn as sns
#%%
def addDiagonal(ax):
    YLIM = ax.get_ylim()
    XLIM = ax.get_xlim()
    LIM = (max(0, min(XLIM[0], YLIM[0])), max(XLIM[1], YLIM[1]))
    RNG = LIM[1] - LIM[0]
    LIM = (LIM[0]-RNG*0.05, LIM[1]+RNG*0.05)
    ax.plot(LIM, LIM)
    ax.set_xlim(LIM)
    ax.set_ylim(LIM)

def addHline(ax, hval=0):
    XLIM = ax.get_xlim()
    LIM = XLIM
    RNG = LIM[1] - LIM[0]
    LIM = (LIM[0]-RNG*0.05, LIM[1]+RNG*0.05)
    ax.plot(LIM, [hval, hval], alpha=0.2, )

laynames = record_all.Layer.unique()
layermap = {nm: i for i, nm in enumerate(laynames)}
layermap_inall = {'conv1':0,'conv2':1,'conv3':2,'conv4':3,'conv5':4,'conv6':5,'conv7':6,'conv8':7,'conv9':8,\
                  'conv10':9,'conv11':10,'conv12':11,'conv13':12,'fc1':13,'fc2':14,'fc3':15}

#%%
ax = plt.subplot()
plt.scatter(record_all.manifMax.apply(lambda a:a[0]), record_all.manifMax_rf.apply(lambda a:a[0]),
            alpha=0.3, c=record_all.Layer.map(layermap_inall), )
plt.ylabel("Resized to match RF")
plt.xlabel("Full size")
addDiagonal(ax)
ax.set_aspect(1)
plt.title("Comparison of Max Score in Manifold Full vs Resized")
plt.savefig(join(sumdir, "xscatter_MaxScore_cmp.png"))
plt.savefig(join(sumdir, "xscatter_MaxScore_cmp.pdf"))
plt.show()
#%%
def pairedJitter(ax, tab, varlist, jitter=True, xoffset=0, xsigma=0.25):
    if jitter:
        xjit = np.random.randn(tab.shape[0])*xsigma
    else:
        xjit = np.zeros(tab.shape[0])
    for i, varnm in enumerate(varlist):
        ax.scatter(xoffset + i + xjit, tab[varnm], alpha=0.5)
    xtic = xoffset + np.arange(len(varlist))
    ax.plot(xtic[:, np.newaxis] + xjit[np.newaxis, :], tab[varlist].T, alpha=0.2, color=[0,0,0])

def pairedJitterVar(ax, var_col, jitter=True, xoffset=0, xsigma=0.2):
    if jitter:
        xjit = np.random.randn(var_col[0].shape[0])*xsigma
    else:
        xjit = np.zeros(var_col[0].shape[0])
    for i, vararr in enumerate(var_col):
        ax.scatter(xoffset + i + xjit, vararr, alpha=0.5)
    xtic = xoffset + np.arange(len(var_col))
    var_col_arr = np.array(var_col)
    ax.plot(xtic[:, np.newaxis] + xjit[np.newaxis, :], var_col_arr, alpha=0.2, color=[0,0,0])

ax = plt.subplot()
manif1Max = pd.DataFrame([record_all.manifMax.apply(lambda a:a[0]), record_all.manifMax_rf.apply(lambda a:a[0])]).T
for xi, L in enumerate(unit_list):
    msk = (record_all.Layer==L[1])
    pairedJitter(ax, manif1Max[msk], ["manifMax", "manifMax_rf"], xoffset=2.5*xi, xsigma=0.15)
    Tval, Pval = ttest_rel(manif1Max[msk]["manifMax"], manif1Max[msk]["manifMax_rf"])
    print("%s Layer %s t=%.3f(p=%.1e)"%(netname, L[1], Tval, Pval))
    ax.text(2.5*xi, 400, "%s\nt=%.2f\np=%.1e"%(L[1], Tval, Pval), fontsize=7)
addHline(ax, hval=0)
plt.savefig(join(sumdir, "MaxScore_layer_jitter_cmp.png"))
plt.savefig(join(sumdir, "MaxScore_layer_jitter_cmp.pdf"))
plt.show()
#%%
# record_all.manifMax.apply(lambda a:a[0])

# manifMaxAll = pd.DataFrame([record_all.manifMax.apply(lambda a:a[0]), record_all.manifMax_rf.apply(lambda a:a[0])]).T
manifMaxAll = pd.DataFrame([record_all.manifMax.apply(lambda a:a[0]),
              record_all.manifMax.apply(lambda a:a[1]),
              record_all.manifMax.apply(lambda a:a[2]),
              record_all.manifMax.apply(lambda a:a[3]),
              record_all.manifMax_rf.apply(lambda a:a[0]),
              record_all.manifMax_rf.apply(lambda a:a[1]),
              record_all.manifMax_rf.apply(lambda a:a[2]),
              record_all.manifMax_rf.apply(lambda a:a[3]),],
            index=["PC23","PC2526","PC4950","RND12","PC23_rf","PC2526_rf","PC4950_rf","RND12_rf"]).T
#%%
ax = plt.subplot()
for xi, L in enumerate(unit_list):
    msk = (record_all.Layer==L[1])
    pairedJitter(ax, manifMaxAll[msk], ["PC23", "PC2526","PC4950","RND12"], xoffset=5*xi, xsigma=0.15)
    # Tval, Pval = ttest_rel(manifMaxAll[msk]["manifMax"], manifMaxAll[msk]["manifMax_rf"])
    # print("%s Layer %s t=%.3f(p=%.1e)"%(netname, L[1], Tval, Pval))
    # ax.text(2.5*xi, 400, "%s\nt=%.2f\np=%.1e"%(L[1], Tval, Pval), fontsize=7)
addHline(ax, hval=0)
plt.ylabel("Max Score")
plt.savefig(join(sumdir, "MaxScore_layer_jitter_subsp_cmp.png"))
plt.savefig(join(sumdir, "MaxScore_layer_jitter_subsp_cmp.pdf"))
plt.show()
#%%
ax = plt.subplot()
for xi, L in enumerate(unit_list):
    msk = (record_all.Layer==L[1])
    pairedJitter(ax, manifMaxAll[msk], ["PC23_rf","PC2526_rf","PC4950_rf","RND12_rf"], xoffset=5*xi, xsigma=0.15)
    # Tval, Pval = ttest_rel(manifMaxAll[msk]["manifMax"], manifMaxAll[msk]["manifMax_rf"])
    # print("%s Layer %s t=%.3f(p=%.1e)"%(netname, L[1], Tval, Pval))
    # ax.text(2.5*xi, 400, "%s\nt=%.2f\np=%.1e"%(L[1], Tval, Pval), fontsize=7)
addHline(ax, hval=0)
plt.ylabel("Max Score")
plt.savefig(join(sumdir, "MaxScore_layer_jitter_RF_subsp_cmp.png"))
plt.savefig(join(sumdir, "MaxScore_layer_jitter_RF_subsp_cmp.pdf"))
plt.show()
#%%
ax = plt.subplot()
for xi, L in enumerate(unit_list):
    msk = (record_all.Layer==L[1])
    pairedJitter(ax, record_all[msk], ["evolast_std","evolast_std_rf"], xoffset=3*xi, xsigma=0.15)
    Tval, Pval = ttest_rel(record_all[msk]["evolast_std"], record_all[msk]["evolast_std_rf"])
    print("%s Layer %s t=%.3f(p=%.1e)"%(netname, L[1], Tval, Pval))
    ax.text(3*xi, 25, "%s\nt=%.2f\np=%.1e"%(L[1], Tval, Pval), fontsize=7)
addHline(ax, hval=0)
plt.ylabel("Score Std of last gen in Evol")
plt.savefig(join(sumdir, "EvoLastGenStd_layer_jitter_cmp.png"))
plt.savefig(join(sumdir, "EvoLastGenStd_layer_jitter_cmp.pdf"))
plt.show()