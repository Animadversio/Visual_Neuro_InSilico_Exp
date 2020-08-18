import os
import re
import numpy as np
import matplotlib.pylab as plt
import seaborn
from time import time
from os.path import join
import pandas as pd
from scipy.stats import ttest_rel,ttest_ind
#%% summarize difference between methods when applying to fc6
rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune"
summarydir = join(rootdir, "summary")
os.makedirs(summarydir, exist_ok=True)
# savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"
#%% Do a survey of all the exp done
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
#%% Align the experiments with same initialization
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
#%%
ttest_rel(exprec_tab[exprec_tab.optimizer=="CMA_class"].score,
    exprec_tab[exprec_tab.optimizer=="CMA_prod"].score)
#%%
ttest_rel(align_tab.CholCMA, align_tab.CMA_all, nan_policy='omit')
ttest_ind(align_tab.CholCMA, align_tab.CMA_all, nan_policy='omit')
ttest_rel(align_tab.CholCMA, align_tab.CholCMA_noA, nan_policy='omit')
ttest_ind(align_tab.CholCMA, align_tab.CholCMA_noA, nan_policy='omit')
#%%

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
#%% Load the cluster result
rootdir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
summarydir = join(rootdir, "summary")
os.makedirs(summarydir, exist_ok=True)
# savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"
#%% Do a survey of all the exp done
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
#%% Formulate a experiment record
exprec_tab.to_csv(join(summarydir, "optim_raw_score_tab.csv"))
#%% Align the experiments with same initialization
align_col = []
methods_all = exprec_tab.optimizer.unique()
methods_BigGAN = [method for method in methods_all if not "_fc6" in method]
BigGAN_msk = ~exprec_tab.optimizer.str.contains("_fc6")
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
align_tab = pd.DataFrame(align_col, columns=["unitstr", "net", "layer", "unit", "RND"]+list(methods_BigGAN))
#%%
align_tab.to_csv(join(summarydir, "optim_aligned_score_tab_BigGAN.csv"))
#%%
align_col_fc6 = []
methods_all = exprec_tab.optimizer.unique()
methods_fc6 = [method for method in methods_all if "_fc6" in method]
fc6_msk = exprec_tab.optimizer.str.contains("_fc6")
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
align_tab_fc6 = pd.DataFrame(align_col_fc6, columns=["unitstr", "net", "layer", "unit", "RND"]+list(methods_fc6))
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
#%% Get sample images for each method! and montage them together.
from imageio import imread
from PIL import Image
from torchvision.utils import make_grid
from build_montages import build_montages, color_framed_montages, make_grid_np

exampdir = "E:\Cluster_Backup\BigGAN_Optim_Tune_new\summary\examplar"

def read_lastgen(rec, imgid):
    img = imread(
        join(rootdir, rectmp.unitstr, "lastgen%s_%05d_score%.1f.jpg" % (rec.optimizer, rec.RND, rec.score)))
    nrow, ncol = img.shape[0] // 258, img.shape[1] // 258
    ri, ci = np.unravel_index(imgid, (nrow, ncol))
    img_crop = img[2 + 258 * ri:258 + 258 * ri, 2 + 258 * ci:258 + 258 * ci, :]
    Image.fromarray(img_crop).show()
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
#%% Go through all the images

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

#%% Sample trajectory for each method!