"""Example code for generating figures of Restrict Evolution in silico experiments
Updated 2021 May 16
"""

from os.path import join
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def set_violin_color(violin_parts, clrcode):
    """Simple util that set the color from a plt violinplot"""
    for key, pc in violin_parts.items():
        if key is 'bodies':
            pc = pc[0]
        pc.set_facecolor(clrcode)
        pc.set_color(clrcode)

def saveallforms(figdirs, fignm, figh=None, fmts=["png","pdf"]):
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx))

def summary_by_block(scores_vec,gens,maxgen=100,sem=True):
    """Summarize a score trajectory and and generation vector into the mean vector, sem, """
    genmin = min(gens)
    genmax = max(gens)
    if maxgen is not None:
        genmax = min(maxgen, genmax)

    score_m = []
    score_s = []
    blockarr = []
    for geni in range(genmin, genmax+1):
        score_block = scores_vec[gens==geni]
        if len(score_block)==1:
            continue
        score_m.append(np.mean(score_block))
        if sem:
            score_s.append(np.std(score_block)/np.sqrt(len(score_block)))
        else:
            score_s.append(np.std(score_block))
        blockarr.append(geni)
    score_m = np.array(score_m)
    score_s = np.array(score_s)
    blockarr = np.array(blockarr)
    return score_m, score_s, blockarr

#%%
outdir = r"E:\OneDrive - Washington University in St. Louis\Manuscript_Manifold\Figure3\RedDimEffectProg"
figdir = r"D:\Generator_DB_Windows\data\with_CNN\RDevol_summary"
basedir = r"D:\Generator_DB_Windows\data\with_CNN"
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]
layer_names = [unit[1] for unit in unit_arr]

#%%
layer_names = [unit[1] for unit in unit_arr]
for unit in unit_arr[:1]:
    best_scores = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "best_scores.npy"))

best_scores.mean(axis=1)
#np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
#%%
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), frameon=False)
for i, unit in enumerate(unit_arr):
    best_scores = np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
    normalizer = best_scores.mean()
    parts = axes.violinplot(best_scores.mean(axis=1) / normalizer, [i], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#F77C62')#'#FF33FF'

    best_scores = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "best_scores.npy"))  # 50d evolution
    parts = axes.violinplot(best_scores.mean(axis=1) / normalizer, [i], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#3333FF')

    # if i > 1:
    #     unit = (unit[0], unit[1], 5)
    # # using 20 random direction
    # best_scores = np.load(join(basedir, "%s_%s_%d_subspac20" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
    # parts = axes.violinplot(best_scores.mean(axis=1) / normalizer, [i], points=20, widths=0.6,
    #                   showmeans=False, showextrema=False, showmedians=True)
    # set_violin_color(parts, '#3399FF')
#     # using 100 random direction
    best_scores = np.load(join(basedir, "%s_%s_%d_subspac200" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
    parts = axes.violinplot(best_scores.mean(axis=1) / normalizer, [i], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#33FFFF')

axes.set_xticks(range(len(unit_arr)))
# axes.set_xticklabels(["%s-unit%d"%(unit[1],unit[2]) for unit in unit_arr], rotation=30)
axes.set_xticklabels([unit[1] for unit in unit_arr], rotation=30)
axes.set_ylabel("activation / maxima of full evolution")
axes.set_title("Comparing Effect of Dimension Restriction on Evolution\n for CaffeNet (4096, 200, 50d)")
axes.legend(["full", "full", "50d", "50d", "200d", "200d"])
fig.show()
#%
saveallforms([figdir, outdir], "caffenet_restrict_cmp", figh=fig)

#%% Compute Spearman correlation and statisitcs string for the
basedir = r"D:\Generator_DB_Windows\data\with_CNN"
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]

statcol = []
layercol = []
for layeri, unit in enumerate(unit_arr):
    best_scores = np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "best_scores.npy"))
    best_scores_50 = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "best_scores.npy"))  # 50d
    # evolution
    normalizer = best_scores.mean()
    Cratio = best_scores_50.mean(axis=1) / normalizer
    statcol.append(Cratio)
    layercol.append(layeri*np.ones(Cratio.shape,))
statvec = np.concatenate(tuple(statcol))
layervec = np.concatenate(tuple(layercol))
# pd.DataFrame(statcol)
cval, pval = spearmanr(layervec, statvec)
print("Corr between layer idx and Reduction ratio %.3f (p=%.1e n=%d)"%(cval, pval, len(layervec)))
#%% compute d' integral

basedir = r"D:\Generator_DB_Windows\data\with_CNN"
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]

#%% Quantify the difference of score trajectory using Dprime integral violin plot
def calc_dprime(arr1, arr2):
    dprime = np.sqrt(2)*(np.mean(arr1) - np.mean(arr2))/ np.sqrt(np.var(arr1)+np.var(arr2))
    return dprime


def calc_ingrated_dprime(gens1, vec1, gens2, vec2, maxgen=None):
    genmin = max(min(gens1), min(gens2))
    genmax = min(max(gens1), max(gens2))
    if maxgen is not None:
        genmax = min(maxgen, genmax)
    dprime_col = []
    for geni in range(genmin, genmax+1):
        if len(vec1[gens1==geni])==1 or len(vec2[gens2==geni])==1:
            continue
        dpr = calc_dprime(vec1[gens1==geni], vec2[gens2==geni])
        dprime_col.append(dpr)

    dpr_mean = np.mean(dprime_col)
    return dpr_mean


dffull = pd.DataFrame()
unit = unit_arr[1]
for li, unit in enumerate(unit_arr):
    dpr_mean_col = []
    full_trajs = []
    for triali in range(20): # only 20 trials are done with full 4096 space
        data = np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "scores_trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        full_trajs.append((gens, scores_vec))

    rd50_trajs = []
    for triali in range(100): # 100 trials are done with full 4096 space
        data = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "scores_subspc50trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        rd50_trajs.append((gens, scores_vec))
        for j in range(5):
            dpr_mean = calc_ingrated_dprime(*full_trajs[j], gens, scores_vec, maxgen=80)
            dpr_mean_col.append(dpr_mean)

    dfpart = pd.DataFrame(dpr_mean_col, columns=["dpr_integr"])
    dfpart["layer"] = unit[1]
    dfpart["layeri"] = li
    dffull = pd.concat((dffull, dfpart), axis=0)

#%
sns.violinplot("layer", "dpr_integr", data=dffull)
plt.show()

#%%
maxgen = 200
full_trajs_col = {}
rd50_trajs_col = {}
rd200_trajs_col = {}
for li, unit in enumerate(unit_arr):
    full_trajs = []
    for triali in range(20): # only 20 trials are done with full 4096 space
        data = np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "scores_trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        # full_trajs.append((gens, scores_vec))
        score_m, score_s, blockarr = summary_by_block(scores_vec,gens,maxgen=maxgen)
        full_trajs.append((score_m, score_s, blockarr))

    rd200_trajs = []
    for triali in range(100):  # 100 trials are done with full 4096 space
        data = np.load(join(basedir, "%s_%s_%d_subspac200" % (unit[0], unit[1], unit[2]), "scores_subspc200trial%03d.npz" %
                            triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        # rd50_trajs.append((gens, scores_vec))
        score_m, score_s, blockarr = summary_by_block(scores_vec, gens, maxgen=maxgen)
        rd200_trajs.append((score_m, score_s, blockarr))

    rd50_trajs = []
    for triali in range(100): # 100 trials are done with full 4096 space
        data = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "scores_subspc50trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        # rd50_trajs.append((gens, scores_vec))
        score_m, score_s, blockarr = summary_by_block(scores_vec,gens,maxgen=maxgen)
        rd50_trajs.append((score_m, score_s, blockarr))
    
    full_trajs_col[li] = full_trajs.copy()
    rd200_trajs_col[li] = rd200_trajs.copy()
    rd50_trajs_col[li] = rd50_trajs.copy()

#%% Raw trajectories
cutoff = 100
figh, axes = plt.subplots(1,len(unit_arr),figsize=(15,2))
for li, unit in enumerate(unit_arr):
    full_trajs = full_trajs_col[li]
    rd50_trajs = rd50_trajs_col[li]
    scorefull_mat = np.array([sm[:cutoff] for sm, ss, gen in full_trajs])
    scorerd50_mat = np.array([sm[:cutoff] for sm, ss, gen in rd50_trajs])
    normalizer = scorefull_mat[:, -1].mean()
    for triali, (score_m, score_s, blockarr) in enumerate(full_trajs):
        axes[li].plot(blockarr[:cutoff], score_m[:cutoff]/normalizer, c=(1,0,0), alpha=0.2)
    
    for triali, (score_m, score_s, blockarr) in enumerate(rd50_trajs):
        axes[li].plot(blockarr[:cutoff], score_m[:cutoff]/normalizer, c=(0,0,1), alpha=0.05)
    axes[li].set_title("%s %s"%(unit[:2]))
    axes[li].set_xlim([0,cutoff])
axes[0].set_ylabel("Normalized Activation")
saveallforms([figdir, outdir], "caffenet_rawtrajs%d"%cutoff)
figh.tight_layout(pad=0.5)
figh.show()

#%% Stats plot: Integrated Difference Between Trajectories
cutoff = 100
figh, axs = plt.subplots()
for li, unit in enumerate(unit_arr):
    full_trajs = full_trajs_col[li]
    rd50_trajs = rd50_trajs_col[li]
    scorefull_mat = np.array([sm[:cutoff] for sm, ss, gen in full_trajs])
    scorerd50_mat = np.array([sm[:cutoff] for sm, ss, gen in rd50_trajs])
    normalizer = scorefull_mat[:, -1].mean()
    diff_int = (scorefull_mat[:,np.newaxis,:] - scorerd50_mat[np.newaxis,:,:]).mean(axis=2)/normalizer
    parts = axs.violinplot(diff_int.flatten(), [li], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#3333FF')

axs.set_ylabel("Integrated Normalized Diff between Trajs (%d gen)"%cutoff)
axs.set_xticks(range(len(unit_arr)))
axs.set_xticklabels([unit[1] for unit in unit_arr], rotation=30)
saveallforms([figdir, outdir], "caffenet_NormTrajDiff%d_progr"%cutoff)
figh.show()
#%% Stats plot: Fraction of differences of last gens
cutoff = 100
figh, axs = plt.subplots()
statcol = []
layercol = []
for li, unit in enumerate(unit_arr):
    full_trajs = full_trajs_col[li]
    rd50_trajs = rd50_trajs_col[li]
    scorefull_mat = np.array([sm[:cutoff] for sm, ss, gen in full_trajs])
    scorerd50_mat = np.array([sm[:cutoff] for sm, ss, gen in rd50_trajs])
    normalizer = scorefull_mat[:, -1].mean()
    activ_fract = (scorerd50_mat[:, np.newaxis, -1] / scorefull_mat[np.newaxis, :, -1]).flatten()
    # diff_int = (scorefull_mat[:,np.newaxis,:] - scorerd50_mat[np.newaxis,:,:]).mean(axis=2)/normalizer
    parts = axs.violinplot(activ_fract, [li], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#3333FF')
    statcol.append(activ_fract)
    layercol.append(li * np.ones(activ_fract.shape, ))

statvec = np.concatenate(tuple(statcol))
layervec = np.concatenate(tuple(layercol))
cval, pval = spearmanr(layervec, statvec)
print("Corr between layer idx and Reduction ratio %.3f (p=%.1e n=%d)" % (cval, pval, len(layervec)))
axs.set_title("Spearman Correlation ~ layer %.3f (p=%.1e n=%d)" % (cval, pval, len(layervec)))
axs.set_ylabel("Ratio of Final Activation (%d gen)"%cutoff)
axs.set_xticks(range(len(unit_arr)))
axs.set_xticklabels([unit[1] for unit in unit_arr], rotation=30)
saveallforms([figdir, outdir], "caffenet_RatioFinalGen%d_progr"%cutoff)
figh.show()
#%%
#%% Stats plot: Fraction of differences of last gens
cutoff = 100
statcol = []
layercol = []
figh, axs = plt.subplots()
for li, unit in enumerate(unit_arr):
    full_trajs = full_trajs_col[li]
    rd50_trajs = rd50_trajs_col[li]
    scorefull_mat = np.array([sm[:cutoff] for sm, ss, gen in full_trajs])
    scorerd50_mat = np.array([sm[:cutoff] for sm, ss, gen in rd50_trajs])
    # scorerd200_mat = np.array([sm[:cutoff] for sm, ss, gen in rd200_trajs])
    normalizer = scorefull_mat[:, -1].mean()
    area_fract = (scorerd50_mat.sum(axis=1)[:, np.newaxis] / scorefull_mat.sum(axis=1)[np.newaxis, :]).flatten()
    # area_fract200 = (scorerd200_mat.sum(axis=1)[:, np.newaxis] / scorefull_mat.sum(axis=1)[np.newaxis, :]).flatten()
    # diff_int = (scorefull_mat[:,np.newaxis,:] - scorerd50_mat[np.newaxis,:,:]).mean(axis=2)/normalizer
    parts = axs.violinplot(area_fract, [li], points=20, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
    set_violin_color(parts, '#3333FF')
    # parts = axs.violinplot(area_fract200, [li], points=20, widths=0.6,
    #                   showmeans=False, showextrema=False, showmedians=True)
    # set_violin_color(parts, '#33FFFF')
    statcol.append(area_fract)
    layercol.append(li * np.ones(area_fract.shape, ))

statvec = np.concatenate(tuple(statcol))
layervec = np.concatenate(tuple(layercol))
cval, pval = spearmanr(layervec, statvec)
print("Corr between layer idx and Reduction ratio %.3f (p=%.1e n=%d)" % (cval, pval, len(layervec)))
axs.set_title("Spearman Correlation ~ layer %.3f (p=%.1e n=%d)" % (cval, pval, len(layervec)))
axs.set_ylabel("Ratio of Area Under Opt Traj (%d gen)"%cutoff)
axs.set_xticks(range(len(unit_arr)))
axs.set_xticklabels([unit[1] for unit in unit_arr], rotation=30)
saveallforms([figdir, outdir], "caffenet_RatioUndTraj%d_progr200"%cutoff)
figh.show()