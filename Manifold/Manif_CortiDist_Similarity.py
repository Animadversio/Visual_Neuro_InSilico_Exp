"""
Summary plot for comparing the cortical distance vs. the similarity of the
tuning maps
"""

import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from stats_utils import saveallforms
savedir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\Cortical_smoothness";
df = pd.read_csv(join(savedir, "Both_all_pair_df.csv"))

#%%
df_corr = df.corr()
#%%
df.groupby(["Animal", "area"], sort=False)["dist", "mapcorr"].corr()

#%%
corrtab = df[df.dist>0].groupby(["Animal", "area"], sort=False)["dist", "mapcorr"].corr("pearson")#(method="spearman")
corrtab.iloc[0::2,-1]
#%%
corrtab = df[:].groupby(["Animal", "area"], sort=False).apply(\
    lambda A: pd.Series([*pearsonr(A.dist, A.mapcorr), *spearmanr(A.dist, A.mapcorr), len(A)],\
                        index=["pearson", 'pval_p', "spearman", 'pval_s', "N"]))
corrtab.to_csv(join(savedir, "Both_all_pair_corr_summary.csv"))
#%%
corrtab_excl0 = df[df.dist>0].groupby(["Animal", "area"], sort=False).apply(\
    lambda A: pd.Series([*pearsonr(A.dist, A.mapcorr), *spearmanr(A.dist, A.mapcorr), len(A)],\
                        index=["pearson", 'pval_p', "spearman", 'pval_s', "N"]))
corrtab_excl0.to_csv(join(savedir, "Both_all_pair_corr_summary_excl0.csv"))
#%%
plt.figure(figsize=(6, 6))
sns.scatterplot(x="dist", y="mapcorr", s=36, alpha=0.15,
                data=df[df.area=="IT"], hue="Animal")
plt.show()
#%%
plt.figure(figsize=(6, 6))
sns.scatterplot(x="dist", y="mapcorr", s=49, alpha=0.075,
                data=df[df.area=="IT"], hue="Animal")
sns.lineplot(x="dist", y="mapcorr",
                data=df[df.area=="IT"], hue="Animal", lw=2.5)
plt.suptitle("Correlation between Cortical "
             "Distance \n and Tuning Map Similarity (IT)", fontsize=16)
plt.xlabel("Cortical Distance ($\mu$m)", fontsize=13)
plt.ylabel("Tuning map correlation", fontsize=13)
saveallforms(savedir, "Cortical_Distance_Tuning_Map_Similarity_IT")
plt.show()
