"""Example code for generating figures of Restrict Evolution in silico experiments
Updated 2021 May 16
"""

import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# import plotly.graph_objects as go
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

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

# Set box off
#%%
def set_violin_color(violin_parts, clrcode):
    for key, pc in violin_parts.items():
        if key is 'bodies':
            pc = pc[0]
        pc.set_facecolor(clrcode)
        pc.set_color(clrcode)

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
#%%
fig.savefig(join(figdir, "caffenet_restrict_cmp.png"))
fig.savefig(join(figdir, "caffenet_restrict_cmp.pdf"))
fig.savefig(join(outdir, "caffenet_restrict_cmp.png"))
fig.savefig(join(outdir, "caffenet_restrict_cmp.pdf"))


#%% Compute Spearman correlation and statisitcs string for the
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
basedir = r"D:\Generator_DB_Windows\data\with_CNN"
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1)]

#%%
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
    for triali in range(20):
        data = np.load(join(basedir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]), "scores_trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        full_trajs.append((gens, scores_vec))

    rd50_trajs = []
    for triali in range(100):
        data = np.load(join(basedir, "%s_%s_%d" % (unit[0], unit[1], unit[2]), "scores_subspc50trial%03d.npz"%triali))
        gens = data['generations']
        scores_vec = data['scores_all']
        rd50_trajs.append((gens, scores_vec))
        for j in range(20):
            dpr_mean = calc_ingrated_dprime(*full_trajs[j], gens, scores_vec, maxgen=100)
            dpr_mean_col.append(dpr_mean)

    dfpart = pd.DataFrame(dpr_mean_col, columns=["dpr_integr"])
    dfpart["layer"] = unit[1]
    dfpart["layeri"] = li
    dffull = pd.concat((dffull, dfpart), axis=0)

#%%
import seaborn as sns
sns.violinplot("layer", "dpr_integr", data=dffull)
plt.show()
