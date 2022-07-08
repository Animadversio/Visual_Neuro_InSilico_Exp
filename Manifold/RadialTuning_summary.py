import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_rel, ttest_1samp
from stats_utils import saveallforms
sumdir = r'E:\OneDrive - Washington University in St. Louis\ImMetricTuning\summary'
figdir = r"E:\OneDrive - Washington University in St. Louis\ImMetricTuning\summary_py"
#%%
df_all = pd.read_csv(join(sumdir, "Both_RadialTuningStatsTab_squ.csv"),)

#%%
def pair_compare_plot(df, name1, name2, valrange=(0, None), titstr="", figdir=figdir, ):
    valmsk = ~df[[name1, name2]].isna().any(axis=1)
    tval, pval = ttest_rel(df[name1], df[name2], nan_policy="omit")
    m1, s1 = df[name1][valmsk].mean(), df[name1][valmsk].sem()
    m2, s2 = df[name2][valmsk].mean(), df[name2][valmsk].sem()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(df[name1], df[name2],)
    # add diagonal axis line
    plt.axline((1,1), slope=1, color="k", linestyle="--")
    ax.set(xlim=valrange, ylim=valrange,)
    ax.set_title(f"{titstr}  "
                 f"{name1} {m1:.2f}+-{s1:.2f} \n vs {name2} {m2:.2f}+-{s2:.2f} \ntval={tval:.2f}, pval={pval:.1e} (N={len(df[valmsk])})")
    ax.set_aspect("equal")
    plt.show()
    if figdir is not None:
        saveallforms(figdir, f"{titstr}_{name1}_vs_{name2}", fig)
    return tval, pval, fig, ax


def pair_strip_plot(df, name1, name2, valrange=(0, None), titstr="", figdir=figdir, figsize=(3.5, 6)):
    valmsk = ~df[[name1, name2]].isna().any(axis=1)
    tval, pval = ttest_rel(df[name1], df[name2], nan_policy="omit")
    m1, s1 = df[name1][valmsk].mean(), df[name1][valmsk].sem()
    m2, s2 = df[name2][valmsk].mean(), df[name2][valmsk].sem()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xjitter = np.random.uniform(-0.1, 0.1, len(df[valmsk]))
    plt.scatter(xjitter, df[name1][valmsk], )
    plt.scatter(xjitter+1, df[name2][valmsk], )
    plt.plot(xjitter[np.newaxis,:]+
             np.arange(2)[:,np.newaxis],
             df[[name1, name2]][valmsk].T,
             color="k", alpha=0.2)
    plt.xticks([0, 1], [name1, name2])
    # add diagonal axis line
    # plt.axline((1, 1), slope=1, color="k", linestyle="--")
    # ax.set(xlim=valrange, ylim=valrange, )
    ax.set_title(f"{titstr}  "
                 f"{name1} {m1:.2f}+-{s1:.2f} \n vs {name2} {m2:.2f}+-{s2:.2f} \ntval={tval:.2f}, pval={pval:.1e} (N={len(df[valmsk])})")
    # ax.set_aspect("equal")
    plt.show()
    if figdir is not None:
        saveallforms(figdir, f"{titstr}_{name1}_vs_{name2}_strip", fig)
    return tval, pval, fig, ax
#%%

V4msk = (df_all["area"] == "V4")
tval, pval, fig, ax = pair_compare_plot(df_all[V4msk], "normAUC_mani", "normAUC_pasu", (0, 0.6), "V4")
tval, pval, fig, ax = pair_compare_plot(df_all[V4msk], "peak_mani", "peak_pasu", (0, 650), "V4")
tval, pval, fig, ax = pair_compare_plot(df_all[V4msk], "AUC_mani", "AUC_pasu", (0, 200), "V4")

#%%
V1msk = (df_all["area"] == "V1")
tval, pval, fig, ax = pair_compare_plot(df_all[V1msk], "normAUC_mani", "normAUC_gab", (0, 0.6), "V1")
tval, pval, fig, ax = pair_compare_plot(df_all[V1msk], "peak_mani", "peak_gab", (0, 800), "V1")
tval, pval, fig, ax = pair_compare_plot(df_all[V1msk], "AUC_mani", "AUC_gab", (0, 450), "V1")

#%%
tval, pval, fig, ax = pair_strip_plot(df_all[V4msk], "normAUC_mani", "normAUC_pasu", (0, 0.6), "V4")
tval, pval, fig, ax = pair_strip_plot(df_all[V4msk], "peak_mani", "peak_pasu", (0, 650), "V4")
tval, pval, fig, ax = pair_strip_plot(df_all[V4msk], "AUC_mani", "AUC_pasu", (0, 200), "V4")

tval, pval, fig, ax = pair_strip_plot(df_all[V1msk], "normAUC_mani", "normAUC_gab", (0, 0.6), "V1")
tval, pval, fig, ax = pair_strip_plot(df_all[V1msk], "peak_mani", "peak_gab", (0, 800), "V1")
tval, pval, fig, ax = pair_strip_plot(df_all[V1msk], "AUC_mani", "AUC_gab", (0, 450), "V1")

