import os
import re
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from time import time
from os.path import join
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
#%% Summarize difference between methods when applying to fc6
rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune"
summarydir = join(rootdir, "summary")
os.makedirs(summarydir, exist_ok=True)
# savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"
#%% Do a survey of all the exp done, Put them in a pandas DataFrame
unit_strs = os.listdir(rootdir)
unit_strs = [unit_str for unit_str in unit_strs if "alexnet" in unit_str]
unit_pat = re.compile("(.*)_(.*)_(\d*)")
unit_tups = [unit_pat.findall(unit_str)[0] for unit_str in unit_strs]
unit_tups = [(tup[0],tup[1],int(tup[2])) for tup in unit_tups]
rec_col = []
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    fns = os.listdir(join(rootdir, unit_str))
    assert unit_str == "%s_%s_%d"%unit_tup
    trajfns = [fn for fn in fns if "traj" in fn]
    traj_fn_pat = re.compile("traj(.*)_(\d*)_score([\d.-]*).jpg")
    for trajfn in trajfns:
        parts = traj_fn_pat.findall(trajfn)[0]
        entry = (unit_str, *unit_tup, parts[0], int(parts[1]), float(parts[2]))
        rec_col.append(entry)

exprec_tab = pd.DataFrame(rec_col, columns=["unitstr", 'net', 'layer', 'unit', "optimizer", "RND", "score"])
#%%
exprec_tab.to_csv(join(summarydir, "optim_raw_score_tab.csv"))
#%% Align the experiments with same initialization (RND value) make an aligned DataFrame
align_col = []
methods = exprec_tab.optimizer.unique()
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    mask = exprec_tab.unitstr == unit_str
    RNDs = exprec_tab[mask].RND.unique()
    for RND in RNDs:
        entry = [unit_str, *unit_tup, RND, ]
        for method in methods:
            maskprm = mask & (exprec_tab.RND==RND) & (exprec_tab.optimizer==method)
            try:
                score = exprec_tab[maskprm].score.item()
            except ValueError:
                print("Imcomplete Entry %s (RND %d, unit %s)" % (method, RND, unit_str))
                score = np.nan
            entry.append(score)
        align_col.append(tuple(entry))
align_tab = pd.DataFrame(align_col, columns=["unitstr", "net", "layer", "unit", "RND"]+list(methods))
#%%
align_tab.to_csv(join(summarydir, "optim_aligned_score_tab.csv"))
#%% Hypothesis Testing with unaligned Data
ttest_rel(exprec_tab[exprec_tab.optimizer=="CMA_class"].score,
    exprec_tab[exprec_tab.optimizer=="CMA_prod"].score)
#%%
ttest_rel(align_tab.CholCMA, align_tab.CMA_all, nan_policy='omit')
ttest_ind(align_tab.CholCMA, align_tab.CMA_all, nan_policy='omit')
ttest_rel(align_tab.CholCMA, align_tab.CholCMA_noA, nan_policy='omit')
ttest_ind(align_tab.CholCMA, align_tab.CholCMA_noA, nan_policy='omit')
#%% Plot the aligned experiments and the scores comparison
jitter = 0.1*np.random.randn(align_tab.shape[0])
plt.figure(figsize=[8,8])
plt.plot(np.array([[1, 2, 3, 4,5]]).T+jitter[np.newaxis, :], align_tab[["CholCMA","CholCMA_noA","CMA_all","CMA_class",
                                                           "CMA_prod",]].to_numpy().T, color="gray", alpha=0.5)
plt.scatter(1+jitter, align_tab.CholCMA, label="CholCMA")
plt.scatter(2+jitter, align_tab.CholCMA_noA, label="CholCMA_noA")
plt.scatter(3+jitter, align_tab.CMA_all, label="CMA_all")
plt.scatter(4+jitter, align_tab.CMA_class, label="CMA_class")
plt.scatter(5+jitter, align_tab.CMA_prod, label="CMA_prod")
plt.ylabel("Activation Score")
plt.xlabel("Optimizer Used")
plt.xticks([1,2,3,4,5],["CholCMA","CholCMA_noA","CMA_all","CMA_class","CMA_prod"])
chol_all_t = ttest_rel(align_tab.CholCMA, align_tab.CMA_all, nan_policy='omit')
chol_prod_t = ttest_rel(align_tab.CholCMA, align_tab.CMA_prod, nan_policy='omit')
chol_class_t = ttest_rel(align_tab.CholCMA, align_tab.CMA_class, nan_policy='omit')
chol_noA_t = ttest_rel(align_tab.CholCMA, align_tab.CholCMA_noA, nan_policy='omit')
plt.title("Comparing Performance of Optimizers in Activation Maximizing Alexnet units\n"
          "paired-t: CholCMA-CMA_all:t=%.1f(p=%.1E)\nCholCMA-CMA_prod:t=%.1f(p=%.1E)\n CholCMA-CMA_class:t=%.1f("
          "p=%.1E) \nCholCMA-CholCMA_noA:t=%.1f(p=%.1E)"%
          (chol_all_t.statistic, chol_all_t.pvalue,chol_prod_t.statistic, chol_prod_t.pvalue,chol_class_t.statistic,
            chol_class_t.pvalue,chol_noA_t.statistic, chol_noA_t.pvalue,))
plt.axis('auto')
plt.savefig(join(summarydir, "fc6_optimizer_scores_cmp.jpg"))
plt.show()
#%%
"""
Load the cluster in silico exp results (newest ones). 
"""

rootdir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
summarydir = join(rootdir, "summary")
os.makedirs(summarydir, exist_ok=True)
# savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"
#%% Do a survey of all the exp done (Full version, not Receptive Field Matched)
unit_strs = os.listdir(rootdir)
unit_strs = [unit_str for unit_str in unit_strs if "alexnet" in unit_str]  # only keep the full size evolution.
unit_pat = re.compile("([^_]*)_([^_]*)_(\d*)(_RFrsz)?")  # 'alexnet_fc8_59_RFrsz'
# last part is a suffix indicating if it's doing resized evolution (Resize image to match RF)
unit_tups = [unit_pat.findall(unit_str)[0] for unit_str in unit_strs]
unit_tups = [(tup[0], tup[1], int(tup[2]), tup[3]) for tup in unit_tups]
rec_col = []
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    fns = os.listdir(join(rootdir, unit_str))
    assert unit_str == "%s_%s_%d%s" % unit_tup
    trajfns = [fn for fn in fns if "traj" in fn]
    traj_fn_pat = re.compile("traj(.*)_(\d*)_score([\d.-]*).jpg")  # e.g. 'trajHessCMA_noA_90152_score22.7.jpg'
    for trajfn in trajfns:
        parts = traj_fn_pat.findall(trajfn)[0]  # tuple of (optimizer, RND, score)
        GANname = "fc6" if "fc6" in parts[0] else "BigGAN"  # parse the GAN name from it
        entry = (unit_str, *unit_tup, parts[0], GANname, int(parts[1]), float(parts[2]))
        rec_col.append(entry)

exprec_tab = pd.DataFrame(rec_col, columns=["unitstr", 'net', 'layer', 'unit', 'suffix', "optimizer", "GAN", "RND",
                                            "score"])
#%% Formulate a experiment record
exprec_tab.to_csv(join(summarydir, "optim_raw_score_tab.csv"))
#%% Align the experiments with same initialization
align_col = []
methods_all = exprec_tab.optimizer.unique()
methods_BigGAN = [method for method in methods_all if not "_fc6" in method]  # Methods for BigGAN space
BigGAN_msk = exprec_tab.GAN == "BigGAN"
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    mask = (exprec_tab.unitstr == unit_str) & BigGAN_msk
    RNDs = exprec_tab[mask].RND.unique()
    for RND in RNDs:
        entry = [unit_str, *unit_tup, RND, ]
        for method in methods_BigGAN:
            maskprm = mask & (exprec_tab.RND==RND) & (exprec_tab.optimizer==method)
            try:
                score = exprec_tab[maskprm].score.item()
            except ValueError:
                print("Imcomplete Entry %s (RND %d, unit %s)" % (method, RND, unit_str))
                score = np.nan
            entry.append(score)
        align_col.append(tuple(entry))
align_tab = pd.DataFrame(align_col, columns=["unitstr", "net", "layer", "unit", 'suffix', "RND"]+list(methods_BigGAN))
#%%
align_tab.to_csv(join(summarydir, "optim_aligned_score_tab_BigGAN.csv"))
#%%
align_col_fc6 = []
methods_all = exprec_tab.optimizer.unique()
methods_fc6 = [method for method in methods_all if "_fc6" in method]
fc6_msk = exprec_tab.GAN == "fc6"
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    mask = (exprec_tab.unitstr == unit_str) & fc6_msk  # experiments using fc6 GAN
    RNDs = exprec_tab[mask].RND.unique()
    for RND in RNDs:
        entry = [unit_str, *unit_tup, RND, ]
        for method in methods_fc6:
            maskprm = mask & (exprec_tab.RND==RND) & (exprec_tab.optimizer==method)
            try:
                score = exprec_tab[maskprm].score.item()
            except ValueError:
                print("Imcomplete Entry %s (RND %d, unit %s)" % (method, RND, unit_str))
                score = np.nan
            entry.append(score)
        align_col_fc6.append(tuple(entry))
align_tab_fc6 = pd.DataFrame(align_col_fc6, columns=["unitstr", "net", "layer", "unit", 'suffix', "RND"]+list(methods_fc6))
align_tab_fc6.to_csv(join(summarydir, "optim_aligned_score_tab_fc6.csv"))
#%% Visualization
import seaborn as sns
sns.set()
#%%
plt.figure(figsize=[14, 6])
ax = sns.stripplot(x='optimizer', y='score', hue="layer", jitter=0.3,
                   order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                   data=exprec_tab, alpha=0.4)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_strip_layer_all.jpg"))
#%%
plt.figure(figsize=[14,6])
ax = sns.swarmplot(x='optimizer', y='score', hue="layer",
                   order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                   data=exprec_tab, alpha=0.4)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_swarm_layer_all_cat.jpg"))
#%%
plt.figure(figsize=[14,6])
ax = sns.violinplot(x="optimizer", y="score", hue="layer",
                order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=exprec_tab, palette="muted", linewidth=0.2, width=0.9)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_violin_layer_all.jpg"))
#%% M
plt.figure(figsize=[14,6])
ax = sns.violinplot(x="optimizer", y="score",
                order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=exprec_tab, palette="muted", linewidth=0.2, width=0.9, bw=0.1)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet (Pool all units in all layers)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_violin_layer_all_merged.jpg"))
#%%
plt.figure(figsize=[14,6])
ax = sns.violinplot(x="optimizer", y="score", hue="layer",
                order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=exprec_tab, palette="muted", linewidth=0.2, width=0.9, bw=0.1)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet (Layer by Layer)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_violin_layer_all_sbw.jpg"))
#%%
plt.figure(figsize=[14,6])
ax = sns.violinplot(x="layer", y="score", hue="optimizer", linewidth=0.2,
                hue_order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=exprec_tab, palette="muted", bw=0.1, width=0.9)
ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet")
ax.figure.show()
ax.figure.savefig(join(summarydir, "method_cmp_violin_method_all.jpg"))
#%%
ttest_rel(align_tab.HessCMA, align_tab.CholCMA, nan_policy='omit')
ttest_rel(align_tab.HessCMA, align_tab.HessCMA_class, nan_policy='omit')
ttest_rel(align_tab.HessCMA, align_tab.HessCMA_noA, nan_policy='omit')
#%%
"""
Get sample images for each method! and montage them together.
"""
from imageio import imread
from PIL import Image
from torchvision.utils import make_grid
from build_montages import build_montages, color_framed_montages, make_grid_np

exampdir = "E:\Cluster_Backup\BigGAN_Optim_Tune_new\summary\examplar"

def read_lastgen(rec, imgid=0, show=True):
    img = imread(
        join(rootdir, rec.unitstr, "lastgen%s_%05d_score%.1f.jpg" % (rec.optimizer, rec.RND, rec.score)))
    nrow, ncol = img.shape[0] // 258, img.shape[1] // 258
    ri, ci = np.unravel_index(imgid, (nrow, ncol))
    img_crop = img[2 + 258 * ri:258 + 258 * ri, 2 + 258 * ci:258 + 258 * ci, :]
    if show: Image.fromarray(img_crop).show()
    return img_crop

def read_bestimg_in_gen(rec, geni=100):
    try:
        img = imread(join(rootdir, rec.unitstr, "besteachgen%s_%05d.jpg"%(rec.optimizer, rec.RND, )))
    except FileNotFoundError:
        img = imread(join(rootdir, rec.unitstr, "besteachgen%s_%05d_score%.1f.jpg" % (rec.optimizer, rec.RND,
                                                                                     rec.score)))
    nrow, ncol = img.shape[0] // 258, img.shape[1] // 258  # 10, 10
    ri, ci = np.unravel_index(geni-1, (nrow, ncol))
    img_crop = img[2 + 258 * ri:258 + 258 * ri, 2 + 258 * ci:258 + 258 * ci, :]
    return img_crop

#%%
rectmp = exprec_tab.iloc[1000]
imgid = 0
img = imread(join(rootdir, rectmp.unitstr, "lastgen%s_%05d_score%.1f.jpg"%(rectmp.optimizer, rectmp.RND, rectmp.score)))
nrow, ncol = img.shape[0]//258, img.shape[1]//258
ri, ci = np.unravel_index(imgid, (nrow,ncol))
img_crop = img[2+258*ri:258+258*ri,2+258*ci:258+258*ci,:]
Image.fromarray(img_crop).show()

#%% See the scores and images from the same initialization (Method comparison)
# exprec_tab.optimizer.unique()
rec = align_tab.iloc[200]
recgroup = exprec_tab[(exprec_tab.unitstr == rec.unitstr) & (exprec_tab.RND == rec.RND)]
print(recgroup[["unitstr","optimizer","score","RND"]])
imgs_all = [read_bestimg_in_gen(reci) for _,reci in recgroup.iterrows()]
mtg = make_grid_np(np.stack(imgs_all, 3))
Image.fromarray(mtg).show()
# mtg = build_montages(imgs_all, image_shape=(256, 256), montage_shape=(6,1))
#%% See the scores and images from the Different initialization (Method x Init comparison)
rec = align_tab.iloc[200]
imgs_all = []
trialRNDs = align_tab[align_tab.unitstr == rec.unitstr].RND.unique()  # trials for this units
for trRND in trialRNDs:
    recgroup = exprec_tab[(exprec_tab.unitstr == rec.unitstr) & (exprec_tab.RND == trRND)]
    print(recgroup[["unitstr","optimizer","score","RND"]])
    imgs_all.extend([read_bestimg_in_gen(reci) for _, reci in recgroup.iterrows()])
mtg2 = make_grid_np(np.stack(imgs_all, 3), nrow=6)
Image.fromarray(mtg2).show()

#%% See the scores and images from the Different initialization (Method x Init x Space comparison)
rec = align_tab.iloc[200]
unitstr = rec.unitstr
imgs_all = []
scores_all = []
methods = []
trialRNDs = align_tab[align_tab.unitstr == unitstr].RND.unique()  # trials for this units
for trRND in trialRNDs:
    recgroup = exprec_tab[(exprec_tab.unitstr == unitstr) & (exprec_tab.RND == trRND)]
    print(recgroup[["unitstr", "optimizer", "score", "RND"]])
    imgs_all.extend([read_bestimg_in_gen(reci) for _, reci in recgroup.iterrows()])
    scores_all.append(recgroup['score'].array)
methods_BGAN = list(recgroup["optimizer"].array)
mtg2 = make_grid_np(np.stack(imgs_all, 3), nrow=6)
scores_all = np.array(scores_all)
#%% Image.fromarray(mtg2).show()
imgs_all_fc6 = []
scores_all_fc6 = []
trialRNDs_fc6 = align_tab_fc6[align_tab_fc6.unitstr == unitstr].RND.unique()  # trials for this units
for trRND in trialRNDs_fc6:
    recgroup = exprec_tab[(exprec_tab.unitstr == unitstr) & (exprec_tab.RND == trRND)]
    print(recgroup[["unitstr", "optimizer", "score", "RND"]])
    imgs_all_fc6.extend([read_bestimg_in_gen(reci) for _, reci in recgroup.iterrows()])
    scores_all_fc6.append(recgroup['score'].array)
methods_fc6 = list(recgroup["optimizer"].array)
mtg3 = make_grid_np(np.stack(imgs_all_fc6, 3), nrow=3)
scores_all_fc6 = np.array(scores_all_fc6)
#%% Image.fromarray(mtg3).show()
methods_all = methods_BGAN + methods_fc6
scores_cat = np.hstack((scores_all, scores_all_fc6))
mtg_all = Image.fromarray(np.hstack((mtg2, mtg3),))
mtg_all.show()
mtg_all.save(join(exampdir, "%s_repr_img.jpg" % unitstr))
#%%
# fig, ax = plt.subplots()
# ax.matshow(scores_cat, cmap='viridis') # seismic
# for (i, j), z in np.ndenumerate(scores_cat):
#     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
#             bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='0.9'))
# plt.show()
plt.figure(figsize=[8, 6])
ax = sns.heatmap(scores_cat, annot=True, fmt=".1f", xticklabels=methods_all)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t)
plt.xticks(rotation=30)
plt.title("%s Optimizer, Space Comparison"%unitstr, )
ax.figure.savefig(join(exampdir, "%s_scoremap.jpg" % unitstr))
ax.figure.show()
#%% Go through all the images, Draw the Result comparison plot for each unit.
# rec = align_tab.iloc[600]
unit_bothspace = np.intersect1d(align_tab.unitstr.unique(), align_tab_fc6.unitstr.unique())
for unitstr in unit_bothspace[11:]:
    # unitstr = rec.unitstr
    try:
        imgs_all = []
        scores_all = []
        methods = []
        trialRNDs = align_tab[align_tab.unitstr == unitstr].RND.unique()  # trials for this units
        for trRND in trialRNDs:
            recgroup = exprec_tab[(exprec_tab.unitstr == unitstr) & (exprec_tab.RND == trRND)]
            print(recgroup[["unitstr", "optimizer", "score", "RND"]])
            imgs_all.extend([read_bestimg_in_gen(reci) for _, reci in recgroup.iterrows()])
            scores_all.append(recgroup['score'].array)
        methods_BGAN = list(recgroup["optimizer"].array)
        mtg2 = make_grid_np(np.stack(imgs_all, 3), nrow=6)
        scores_all = np.array(scores_all)
        #% Image.fromarray(mtg2).show()
        imgs_all_fc6 = []
        scores_all_fc6 = []
        trialRNDs_fc6 = align_tab_fc6[align_tab_fc6.unitstr == unitstr].RND.unique()  # trials for this units
        for trRND in trialRNDs_fc6:
            recgroup = exprec_tab[(exprec_tab.unitstr == unitstr) & (exprec_tab.RND == trRND)]
            print(recgroup[["unitstr", "optimizer", "score", "RND"]])
            imgs_all_fc6.extend([read_bestimg_in_gen(reci) for _, reci in recgroup.iterrows()])
            scores_all_fc6.append(recgroup['score'].array)
        methods_fc6 = list(recgroup["optimizer"].array)
        mtg3 = make_grid_np(np.stack(imgs_all_fc6, 3), nrow=3)
        scores_all_fc6 = np.array(scores_all_fc6)
        #% Image.fromarray(mtg3).show()
        methods_all = methods_BGAN + methods_fc6
        scores_cat = np.hstack((scores_all, scores_all_fc6))
        mtg_all = Image.fromarray(np.hstack((mtg2, mtg3),))
        mtg_all.show()
        mtg_all.save(join(exampdir, "%s_repr_img.jpg" % unitstr))
        #%%
        plt.figure(figsize=[8, 6])
        ax = sns.heatmap(scores_cat, annot=True, fmt=".1f", xticklabels=methods_all)
        b, t = plt.ylim() # discover the values for bottom and top
        plt.ylim(b + 0.5, t - 0.5)
        plt.xticks(rotation=30)
        plt.title("%s Optimizer, Space Comparison"%unitstr, )
        plt.tight_layout()
        ax.figure.savefig(join(exampdir, "%s_scoremap.jpg" % unitstr))
        ax.figure.show()
    except:
        print("experiment non complete %s Check FC6 table"% unitstr)
        print(align_tab_fc6[align_tab_fc6.unitstr==unitstr])

#%%
"""
Analyze the trajectories for each layer.
Extract the timescale of evolution 
"""
unit_strs = os.listdir(rootdir)
unit_strs = [unit_str for unit_str in unit_strs if "alexnet" in unit_str]  # only keep the full size evolution.
unit_pat = re.compile("([^_]*)_([^_]*)_(\d*)(_RFrsz)?")  # 'alexnet_fc8_59_RFrsz'
# last part is a suffix indicating if it's doing resized evolution (Resize image to match RF)
unit_tups = [unit_pat.findall(unit_str)[0] for unit_str in unit_strs]
unit_tups = [(tup[0], tup[1], int(tup[2]), tup[3]) for tup in unit_tups]
rec_col = []
for ui, unit_str in enumerate(unit_strs):
    unit_tup = unit_tups[ui]
    fns = os.listdir(join(rootdir, unit_str))
    assert unit_str == "%s_%s_%d%s" % unit_tup
    trajfns = [fn for fn in fns if "traj" in fn]
    traj_fn_pat = re.compile("traj(.*)_(\d*)_score([\d.-]*).jpg")  # e.g. 'trajHessCMA_noA_90152_score22.7.jpg'

    for trajfn in trajfns:
        parts = traj_fn_pat.findall(trajfn)[0]  # tuple of (optimizer, RND, score)
        GANname = "fc6" if "fc6" in parts[0] else "BigGAN"  # parse the GAN name from it
        entry = (unit_str, *unit_tup, parts[0], GANname, int(parts[1]), float(parts[2]))
        rec_col.append(entry)

exprec_tab = pd.DataFrame(rec_col, columns=["unitstr", 'net', 'layer', 'unit', 'suffix', "optimizer", "GAN", "RND",
                                            "score"])
#%%
T0 = time()
tau_col = []
for rowi, obj in exprec_tab.iterrows():
    # data_pat = "scores%s_%05d.npz" % (optimname, RND)  # e.g. 'scoresHessCMA_noA_64323.npz'
    datapath = join(rootdir, obj.unitstr, "scores%s_%05d.npz" % (obj.optimizer, obj.RND))
    # os.path.exists(datapath)  # all exist
    try:
        data = np.load(datapath)
        maxgen = data["generations"].max() + 1
        avg_traj = np.array([data["scores_all"][data["generations"] == geni].mean() for geni in range(maxgen)])
        maxscore = avg_traj[-5:].mean() # avg_traj.max() # change to the mean of last few gens
        tau50 = (avg_traj < maxscore*0.5).sum()
        tau_e = (avg_traj < maxscore*0.632).sum()
        tau80 = (avg_traj < maxscore*0.8).sum()
        tau90 = (avg_traj < maxscore*0.9).sum()
        tau_col.append((tau50, tau_e, tau80, tau90))
    except:
        print("Some issue occur when computing the time constant for %s %s %d at row %d. (Fill NAN)" %
              (obj.unitstr, obj.optimizer, obj.RND, rowi))
        tau_col.append((np.nan, np.nan, np.nan, np.nan))
print(time() - T0)  # 1945.5 sec
#%%
tau_tab = pd.DataFrame(tau_col, columns=["tau50", "tau_e", "tau80", "tau90"])
tau_full_tab = pd.concat((exprec_tab, tau_tab), 1)
tau_full_tab.to_csv(join(summarydir, "optim_Timescale_tab_robust.csv"))
# tau_full_tab.to_csv(join(summarydir, "optim_Timescale_tab.csv"))  # original one using maximum activation
# samplepath: "alexnet_conv1_10_RFrsz\scoresHessCMA_noA_64323.npz"
#%%
"""
Visualize the dataset!
"""
mask = (tau_full_tab.suffix=="") & (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 1)
plt.figure(figsize=[8, 6])
ax = sns.violinplot(x="layer", y="tau50", hue="optimizer", linewidth=0.2,
                #hue_order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                #    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=tau_full_tab[mask], palette="muted", bw=0.1, width=0.9)
ax.set_title("Comparison of Convergence Speed over Layers of AlexNet (FC6 GAN)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "timescale_cmp_violin_method_fc6_success.jpg"))
#%%
mask = (tau_full_tab.suffix == "") & (tau_full_tab.GAN == "BigGAN") & (tau_full_tab.score > 1)
plt.figure(figsize=[8, 6])
ax = sns.violinplot(x="layer", y="tau50", hue="optimizer", linewidth=0.2,
                #hue_order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                #    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=tau_full_tab[mask], palette="muted", bw=0.1, width=0.9)
ax.set_title("Comparison of Convergence Speed over Layers of AlexNet (BigGAN)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "timescale_cmp_violin_method_BigGAN_success.jpg"))
#%%
mask = ((tau_full_tab.suffix != "") | ((tau_full_tab.suffix == "") & tau_full_tab.layer.str.contains("fc"))) & \
       (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 1)
plt.figure(figsize=[8, 6])
ax = sns.violinplot(x="layer", y="tau50", hue="optimizer", linewidth=0.2,
                #hue_order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                #    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=tau_full_tab[mask], palette="muted", bw=0.1, width=0.9)
ax.set_title("Comparison of Convergence Speed over Layers of AlexNet (FC6 GAN, Resized to RF)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "timescale_cmp_violin_method_fc6_Resize_success.jpg"))
#%%
mask = ((tau_full_tab.suffix != "") | ((tau_full_tab.suffix == "") & tau_full_tab.layer.str.contains("fc"))) & \
       (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 2) & (tau_full_tab.optimizer == 'CholCMA_fc6')
plt.figure(figsize=[8, 6])
ax = sns.violinplot(x="layer", y="tau50", hue="optimizer", linewidth=0.2,
                    #hue_order=['CholCMA', 'CholCMA_prod', 'CholCMA_class', 'HessCMA', 'HessCMA_class',
                    #    'HessCMA_noA', 'CholCMA_fc6', 'HessCMA500_1_fc6', 'HessCMA800_fc6',],
                    data=tau_full_tab[mask], palette="muted", bw=0.1, width=0.9)
ax.set_title("Comparison of Convergence Speed over Layers of AlexNet (FC6 GAN, Resized to RF)")
ax.figure.show()
ax.figure.savefig(join(summarydir, "timescale_cmp_violin_fc6_cholCMA.jpg"))
#%%
plt.figure(figsize=[5, 4])
ax2 = sns.pointplot(x="layer", y="tau50", data=tau_full_tab[mask], capsize=.2, label="tau50",)
ax2 = sns.pointplot(x="layer", y="tau80", data=tau_full_tab[mask], capsize=.2, ax=ax2, color="darkblue", label="tau80",)
ax2 = sns.pointplot(x="layer", y="tau90", data=tau_full_tab[mask], capsize=.2, ax=ax2, color="black", label="tau90",)
ax2.legend(["tau50", "tau80", "tau90"]) # ["tau50", "tau80", "tau90"]
ax2.set_title("Comparison of Convergence Speed over Layers of AlexNet\n (FC6 GAN, Resized to RF)")
plt.savefig(join(summarydir, "timescale_cmp_errorbar_fc6_cholCMA.pdf"))
ax2.figure.show()
#%%
ax2.figure.savefig(join(summarydir, "timescale_cmp_errorbar_fc6_cholCMA.jpg"))
#%%
mask = ((tau_full_tab.suffix != "") | ((tau_full_tab.suffix == "") & tau_full_tab.layer.str.contains("fc"))) & \
       (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 2) & (tau_full_tab.optimizer == 'CholCMA_fc6')
plt.figure(figsize=[5, 4])
ax2 = sns.pointplot(x="layer", y="tau50", data=tau_full_tab[mask], capsize=.2, label="tau50",)
ax2.legend(["tau50"]) # ["tau50", "tau80", "tau90"]
ax2.set_ylabel("Iterations to Reach Fraction of\nMaximal Activation")
ax2.set_title("Comparison of Convergence Speed over Layers of AlexNet\n (FC6 GAN, Resized to RF)")
plt.savefig(join(summarydir, "timescale_cmp_errorbar_fc6_cholCMA_50.pdf"))
ax2.figure.show()
#%%
mask = (tau_full_tab.suffix == "") & (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 2) & (tau_full_tab.optimizer == 'CholCMA_fc6')
plt.figure(figsize=[5, 4])
ax2 = sns.pointplot(x="layer", y="tau50", data=tau_full_tab[mask], capsize=.2, label="tau50",)
ax2.legend(["tau50"])  # ["tau50", "tau80", "tau90"]
ax2.set_ylabel("Iterations to Reach Fraction of\nMaximal Activation")
ax2.set_title("Comparison of Convergence Speed over Layers of AlexNet\n (FC6 GAN)")
plt.savefig(join(summarydir, "timescale_cmp_errorbar_fc6_cholCMA_50_noRsz.pdf"))
ax2.figure.show()

#%% Plot a few exemplar traj
from GAN_utils import upconvGAN
from torchvision.transforms import ToPILImage
G = upconvGAN("fc6")
traj_dir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new\summary\example_traj"
# Mask for all the resized evolutions in FC6 GAN.
gen_mask = ((tau_full_tab.suffix != "") | ((tau_full_tab.suffix == "") & tau_full_tab.layer.str.contains("fc"))) & \
        (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 2) & (tau_full_tab.optimizer == 'CholCMA_fc6')
# select channels randomly,
rnd_col = []
for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
    layermask = (tau_full_tab.layer == layer)
    obj = tau_full_tab[gen_mask & layermask].sample()
    rnd_col.append(obj)
rnd_col = pd.concat(tuple(rnd_col),axis=0)
rnd_col.to_csv(join(traj_dir, "sample_trials.csv"))
rnd_col
#%%
# Plot all the traces onto the same plot
_, axt = plt.subplots(figsize=[5, 5])
_, axt2 = plt.subplots(figsize=[5, 5])
for _, obj in rnd_col.iterrows():
    datapath = join(rootdir, obj.unitstr,
                    "scores%s_%05d.npz" % (obj.optimizer, obj.RND))  # 'scoresHessCMA_noA_64323.npz'
    data = np.load(datapath)
    # maxgen = data["generations"].max() + 1
    # avg_traj = np.array([data["scores_all"][data["generations"] == geni].mean() for geni in range(maxgen)])
    norm_act = data["scores_all"][data["generations"]>98].mean()
    # Plot the traj and examplar image
    axt = sns.lineplot(x="generations", y="scores_all", data=data, ax=axt, ci="sd")
    axt2 = sns.lineplot(x=data["generations"], y=data["scores_all"] / norm_act, ax=axt2, ci="sd")
    plt.figure(figsize=[5,5])
    ax = sns.lineplot(x="generations", y="scores_all", data=data, ci="sd")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Activation")
    ax.figure.savefig(join(traj_dir, "%s_traj_%05d.pdf" % (obj.unitstr, obj.RND)))
    finalimg = read_lastgen(obj, imgid=0, show=False)
    Image.fromarray(finalimg).save(join(traj_dir, "%s_img_%05d.pdf" % (obj.unitstr, obj.RND)))
    imgs = G.visualize_batch_np(data["codes_fin"][40:41, :])
    ToPILImage()(imgs[0, :]).save(join(traj_dir, "%s_imgfull_%05d.pdf" % (obj.unitstr, obj.RND)))
axt.set_xlabel("Iteration")
axt2.set_xlabel("Iteration")
axt.set_ylabel("Activation")
axt2.set_ylabel("Activation Normalized to Last Iteration")
axt.figure.legend(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'])
axt2.figure.legend(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'])
axt.figure.savefig(join(traj_dir, "All_traj_cmb.pdf"))
axt2.figure.savefig(join(traj_dir, "All_traj_cmb_normalize.pdf"))
# plt.show()
#%%
data["codes_fin"][0,:]
