import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from os.path import join
import matplotlib
from scipy.stats import ttest_rel,ttest_ind,ranksums,wilcoxon
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
Animal = "Beto"
ImMetCorrTab = pd.read_csv("E:\OneDrive - Washington University in St. "
                           "Louis\ImMetricTuning\%s_ImMetricCorrTab.csv"%Animal)
#%%
summarydir = "E:\OneDrive - Washington University in St. Louis\ImMetricTuning\summary"
Animal = "Beto"
ImMetCorrTab1 = pd.read_csv("E:\OneDrive - Washington University in St. "
                           "Louis\ImMetricTuning\%s_ImMetricCorrTab.csv"%Animal)
Animal = "Alfa"
ImMetCorrTab2 = pd.read_csv("E:\OneDrive - Washington University in St. "
                           "Louis\ImMetricTuning\%s_ImMetricCorrTab.csv"%Animal)
ImMetCorrTab = pd.concat((ImMetCorrTab1,ImMetCorrTab2),axis=0)
#%%
fig, ax = plt.subplots()#palette="Blues", palette="Reds_r",palette="Greens_r",
sns.stripplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab, jitter=0.1, ax=ax, alpha=0.6)
sns.stripplot(x='area', y='Pasu_squ_pear', color="red",  data=ImMetCorrTab, jitter=0.2, ax=ax, alpha=0.6)
sns.stripplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab, jitter=0.2, ax=ax, alpha=0.6)
plt.legend(["Manif","Manif","Manif","Pasu","Pasu","Pasu","Gabor","Gabor","Gabor",])
plt.show()
#%%
fig, ax = plt.subplots()
sns.lineplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab.query("Animal=='Beto'"), ax=ax)
sns.lineplot(x='area', y='Pasu_squ_pear', color="red", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
sns.lineplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
plt.legend(["Manif", "Pasu", "Gabor"])
# sns.lineplot(x='area', y='Pasu_squ_pear', color="blue", data=ImMetCorrTab, sort=False,)
# sns.lineplot(x='area', y='Gabor_squ_pear', color="blue", data=ImMetCorrTab,sort=False,)
plt.show()
#%%
fig, ax = plt.subplots()
sns.stripplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab.query("Animal=='Beto'"), ax=ax)
sns.stripplot(x='area', y='Pasu_squ_pear', color="red", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
sns.stripplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
plt.legend(["Manif", "Pasu", "Gabor"])
plt.show()
#%%
fig, ax = plt.subplots()
sns.stripplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab.query("Animal=='Beto'"), ax=ax)
sns.stripplot(x='area', y='Pasu_squ_pear', color="red", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
sns.stripplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab.query("Gabor_incomp == 0 & Animal=='Beto'"),
             ax=ax)
plt.legend(["Manif", "Pasu", "Gabor"])
plt.show()
#%%
fig, ax = plt.subplots()
sns.stripplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab, ax=ax)
sns.stripplot(x='area', y='Pasu_squ_pear', color="red", data=ImMetCorrTab, ax=ax)
sns.stripplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab, ax=ax)
plt.legend(["Manif","Manif","Manif","Pasu","Pasu","Pasu","Gabor","Gabor","Gabor",], loc='best',)
plt.show()
#%%
fig, ax = plt.subplots()
sns.violinplot(x='area', y='Manif_squ_pear', color="blue", data=ImMetCorrTab, ax=ax, lw=0, saturation=0.7)
sns.violinplot(x='area', y='Pasu_squ_pear', color="red", data=ImMetCorrTab, ax=ax, lw=0, saturation=0.7)
sns.violinplot(x='area', y='Gabor_squ_pear', color="green", data=ImMetCorrTab, ax=ax, lw=0, saturation=0.7)
for violin in ax.collections[::2]:
    violin.set_alpha(0.4)
plt.legend(["Manif", "Manif", "Pasu", "Pasu", "Gabor", "Gabor"], loc='best',)
plt.show()

#%%
print(ttest_ind(ImMetCorrTab.Manif_FC6_pear[ImMetCorrTab.area=="V1"],ImMetCorrTab.Manif_FC6_pear[
    ImMetCorrTab.area=="V4"]))
print(ttest_ind(ImMetCorrTab.Manif_FC6_pear[ImMetCorrTab.area=="V1"],ImMetCorrTab.Manif_FC6_pear[
    ImMetCorrTab.area=="IT"]))
#%%
mask = ImMetCorrTab.Animal=="Alfa"
print(ttest_ind(ImMetCorrTab.Manif_FC6_pear[(ImMetCorrTab.area=="V1") & mask],ImMetCorrTab.Manif_FC6_pear[
    (ImMetCorrTab.area=="V4") & mask]))
print(ttest_ind(ImMetCorrTab.Manif_FC6_pear[(ImMetCorrTab.area=="V1") & mask],ImMetCorrTab.Manif_FC6_pear[
    (ImMetCorrTab.area=="IT") & mask]))

#%%
fig, ax = plt.subplots()
sns.violinplot(x="area", y=ImMetCorrTab.Manif_squ_reg_b_2/ImMetCorrTab.Manif_squ_reg_b_1, hue="Animal", palette="Blues",
               data=ImMetCorrTab)
sns.violinplot(x="area", y=ImMetCorrTab.Pasu_squ_reg_b_2/ImMetCorrTab.Pasu_squ_reg_b_1, hue="Animal", palette="Greens",
               data=ImMetCorrTab)
sns.violinplot(x="area", y=ImMetCorrTab.Gabor_squ_reg_b_2/ImMetCorrTab.Gabor_squ_reg_b_1, hue="Animal", palette="Reds",
               data=ImMetCorrTab)
for violin in ax.collections[::2]:
    violin.set_alpha(0.6)
plt.show()
#%%
ax = sns.catplot(x="area", y="Manif_squ_spear", col="Animal", capsize=.2, height=6, aspect=.75,
                kind="point", data=ImMetCorrTab) # palette="YlGnBu_d",
# .query("Gabor_incomp == 0
plt.show()
#%%
ax = sns.stripplot(x="area", y="Manif_squ_spear", hue="Animal", # kind="point", capsize=.2, height=6, aspect=.75,
                data=ImMetCorrTab, alpha=0.75) # palette="YlGnBu_d",
# .query("Gabor_incomp == 0
plt.show()
#%%
msk = ImMetCorrTab.area == "V1"
MGcmp = ttest_rel(ImMetCorrTab.Manif_rng_1[msk], ImMetCorrTab.Gabor_rng_1[msk],nan_policy='omit')
# Ttest_relResult(statistic=11.523355949867135, pvalue=9.774866064200571e-18)
MPcmp = ttest_rel(ImMetCorrTab.Manif_rng_1[msk], ImMetCorrTab.Pasu_rng_1[msk],nan_policy='omit')
# Ttest_relResult(statistic=10.602699480062102, pvalue=2.785799486620647e-16)
print(MGcmp, '\n', MPcmp)
#%%
fig, ax = plt.subplots()
ax = sns.stripplot(x="area", y="Manif_rng_1", data=ImMetCorrTab, ax=ax, palette="Blues_d")
ax = sns.stripplot(x="area", y="Pasu_rng_1", data=ImMetCorrTab, ax=ax, palette="Greens_d")
ax = sns.stripplot(x="area", y="Gabor_rng_1", data=ImMetCorrTab, ax=ax, palette="Reds_d") # hue="Animal",
plt.legend(["Manifold","Manifold","Manifold","Pasupathy","Pasupathy","Pasupathy","Gabor","Gabor","Gabor"])
plt.show()
#%%
V1msk = ImMetCorrTab.area == "V1"
V4msk = ImMetCorrTab.area == "V4"
ITmsk = ImMetCorrTab.area == "IT"
plt.subplots(figsize=[6,8])
plt.scatter(1+np.zeros(np.nansum(V1msk)), ImMetCorrTab['Manif_rng_1'][V1msk],color="blue",alpha=0.6,label="Manifold Space")
plt.scatter(2+np.zeros(np.nansum(V4msk)), ImMetCorrTab['Manif_rng_1'][V4msk],color="blue",alpha=0.6)
plt.scatter(3+np.zeros(np.nansum(ITmsk)), ImMetCorrTab['Manif_rng_1'][ITmsk],color="blue",alpha=0.6)
plt.scatter(1.3+np.zeros(np.nansum(V1msk)), ImMetCorrTab['Pasu_rng_1'][V1msk],color="green",alpha=0.6,label="Pasupathy Space")
plt.scatter(2.3+np.zeros(np.nansum(V4msk)), ImMetCorrTab['Pasu_rng_1'][V4msk],color="green",alpha=0.6)
plt.scatter(3.3+np.zeros(np.nansum(ITmsk)), ImMetCorrTab['Pasu_rng_1'][ITmsk],color="green",alpha=0.6)
plt.scatter(1.6+np.zeros(np.nansum(V1msk)), ImMetCorrTab['Gabor_rng_1'][V1msk],color="red",alpha=0.6,label="Gabor Space")
plt.scatter(2.6+np.zeros(np.nansum(V4msk)), ImMetCorrTab['Gabor_rng_1'][V4msk],color="red",alpha=0.6)
plt.scatter(3.6+np.zeros(np.nansum(ITmsk)), ImMetCorrTab['Gabor_rng_1'][ITmsk],color="red",alpha=0.6)
plt.legend()
# plt.scatter(1+np.array([[0, 0.2, 0.4]]).repeat(sum(V1msk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
#          'Gabor_rng_1']][V1msk])
# plt.scatter(2+np.array([[0, 0.2, 0.4]]).repeat(sum(V4msk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
#          'Gabor_rng_1']][V4msk])
# plt.scatter(3+np.array([[0, 0.2, 0.4]]).repeat(sum(ITmsk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
#          'Gabor_rng_1']][ITmsk])
plt.plot(1+np.array([[0, 0.3, 0.6]]).repeat(sum(V1msk),0).T, ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
         'Gabor_rng_1']][V1msk].T, color="gray", alpha=0.3)
plt.plot(2+np.array([[0, 0.3, 0.6]]).repeat(sum(V1msk),0).T, ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
         'Gabor_rng_1']][V4msk].T, color="gray", alpha=0.3)
plt.plot(3+np.array([[0, 0.3, 0.6]]).repeat(sum(V1msk),0).T, ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
         'Gabor_rng_1']][ITmsk].T, color="gray", alpha=0.3)
plt.xticks([1.3,2.3,3.3],["V1","V4","IT"])
plt.title("Peak Activity Comparison Across Image Spaces")
plt.savefig(join(summarydir, "peak_activ_xspace_cmp.png"))
plt.show()


#%%
from matplotlib import cm
def xspace_cmp_plot(var=['Manif_rng_1', 'Pasu_rng_1', 'Gabor_rng_1'], data=ImMetCorrTab,
                    labels=["Manifold","Pasupathy","Gabor"], cmap=cm.RdBu, titstr="", msk=None):
    V1msk = ImMetCorrTab.area == "V1"
    V4msk = ImMetCorrTab.area == "V4"
    ITmsk = ImMetCorrTab.area == "IT"
    varn = len(var)
    intv = 0.9 / varn
    clist = [cmap(float((vari+.5)/(varn+1))) for vari in range(varn)]
    fig, ax = plt.subplots(figsize=[6, 8])
    for vari, varnm in enumerate(var):
      plt.scatter(1 + intv*vari + np.zeros(np.nansum(V1msk)), data[varnm][V1msk], color=clist[vari], alpha=0.9,
                  label=labels[vari])
      plt.scatter(2 + intv*vari + np.zeros(np.nansum(V4msk)), data[varnm][V4msk], color=clist[vari], alpha=0.9)
      plt.scatter(3 + intv*vari + np.zeros(np.nansum(ITmsk)), data[varnm][ITmsk], color=clist[vari], alpha=0.9)
    
    plt.legend()
    # plt.scatter(1+np.array([[0, 0.2, 0.4]]).repeat(sum(V1msk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
    #          'Gabor_rng_1']][V1msk])
    # plt.scatter(2+np.array([[0, 0.2, 0.4]]).repeat(sum(V4msk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
    #          'Gabor_rng_1']][V4msk])
    # plt.scatter(3+np.array([[0, 0.2, 0.4]]).repeat(sum(ITmsk),0), ImMetCorrTab[['Manif_rng_1','Pasu_rng_1',
    #          'Gabor_rng_1']][ITmsk])
    intvs = (intv * np.arange(varn)).reshape(1, -1)
    plt.plot(1 + intvs.repeat(sum(V1msk), 0).T, data[var][V1msk].T,
             color="gray", alpha=0.3)
    plt.plot(2 + intvs.repeat(sum(V1msk), 0).T, data[var][V4msk].T,
             color="gray", alpha=0.3)
    plt.plot(3 + intvs.repeat(sum(V1msk), 0).T, data[var][ITmsk].T,
             color="gray", alpha=0.3)
    plt.xticks(np.arange(1,4)+intv, ["V1", "V4", "IT"])
    stats = {}
    stats["T01"] = ttest_rel(data[var[0]], data[var[1]], nan_policy='omit')
    stats["T02"] = ttest_rel(data[var[0]], data[var[2]], nan_policy='omit')
    stats["T12"] = ttest_rel(data[var[1]], data[var[2]], nan_policy='omit')
    plt.title(
        "%s\nT: Manif - Pasu:%.1f(%.1e)\nManif - Gabor:%.1f(%.1e)\n"
        "Pasu - Gabor:%.1f(%.1e)" % (titstr, stats["T01"].statistic, stats["T01"].pvalue, stats["T02"].statistic,
                                     stats["T02"].pvalue, stats["T12"].statistic, stats["T12"].pvalue))
    return fig, stats
#%%
fig, stats = xspace_cmp_plot(var=['Manif_rng_2', 'Pasu_rng_2', 'Gabor_rng_2'], data=ImMetCorrTab,
                cmap=cm.RdYlBu, titstr="Activation Lower Bound for Different Image Spaces")
plt.savefig(join(summarydir, "LB_activ_xspace_cmp.png"))
plt.savefig(join(summarydir, "LB_activ_xspace_cmp.pdf"))
plt.show()
#%%
xspace_cmp_plot(var=['Manif_rng_1', 'Pasu_rng_1', 'Gabor_rng_1'], data=ImMetCorrTab,
                cmap=cm.RdYlBu, titstr="Activation Upper Bound for Different Image Spaces")  # cmap=cm.
plt.savefig(join(summarydir, "UB_activ_xspace_cmp.png"))
plt.savefig(join(summarydir, "UB_activ_xspace_cmp.pdf"))
plt.show()
#%% Correlation
xspace_cmp_plot(var=['Manif_squ_spear', 'Pasu_squ_spear', 'Gabor_squ_spear'], data=ImMetCorrTab,
                titstr="Spearman Correlation between Activation and ImDistance to Peak",
                cmap=cm.RdYlBu)
plt.ylabel("Spearman Correlation")
plt.savefig(join(summarydir, "spear_corr_xspace_cmp.png"))
plt.savefig(join(summarydir, "spear_corr_xspace_cmp.pdf"))
plt.show()
#%% Correlation maximum across peak position
xspace_cmp_plot(var=['Manif_squ_spear_max', 'Pasu_squ_spear_max', 'Gabor_squ_spear_max'], data=ImMetCorrTab,
                titstr="Max Spearman Correlation between Activation and ImDistance to Peak",
                cmap=cm.RdYlBu)
plt.ylabel("Spearman Correlation")
plt.savefig(join(summarydir, "spear_max_corr_xspace_cmp.png"))
plt.savefig(join(summarydir, "spear_max_corr_xspace_cmp.pdf"))
plt.show()
#%% Correlation maximum across peak position
xspace_cmp_plot(var=['Manif_squ_spear_max', 'Pasu_squ_spear_max', 'Gabor_squ_spear_max'], data=ImMetCorrTab,
                titstr="Max Spearman Correlation between Activation and ImDistance to Peak",
                cmap=cm.RdYlBu)
plt.ylabel("Spearman Correlation")
plt.savefig(join(summarydir, "spear_max_corr_xspace_cmp.png"))
plt.savefig(join(summarydir, "spear_max_corr_xspace_cmp.pdf"))
plt.show()
#%%
ImMetCorrTab["Manif_norm_slope"] = ImMetCorrTab.Manif_squ_reg_b_1 / ImMetCorrTab.Manif_rng_1
ImMetCorrTab["Pasu_norm_slope"] = ImMetCorrTab.Pasu_squ_reg_b_1 / ImMetCorrTab.Pasu_rng_1
ImMetCorrTab["Gabor_norm_slope"] = ImMetCorrTab.Gabor_squ_reg_b_1 / ImMetCorrTab.Gabor_rng_1
#%%%
xspace_cmp_plot(var=['Manif_norm_slope', 'Pasu_norm_slope', 'Gabor_norm_slope'], data=ImMetCorrTab,
                titstr="Normalized Slope of Activation and ImDistance Regression",
                cmap=cm.RdYlBu)
plt.ylabel("Regression slope / peak activation")
plt.savefig(join(summarydir, "norm_slope_xspace_cmp.png"))
plt.savefig(join(summarydir, "norm_slope_xspace_cmp.pdf"))
plt.show()
#%%
#%%
maxActiv = ImMetCorrTab[["Manif_rng_1","Pasu_rng_1","Gabor_rng_1"]].max(1)
ImMetCorrTab["Manif_uninorm_slope"] = ImMetCorrTab.Manif_squ_reg_b_1 / maxActiv
ImMetCorrTab["Pasu_uninorm_slope"] = ImMetCorrTab.Pasu_squ_reg_b_1 / maxActiv
ImMetCorrTab["Gabor_uninorm_slope"] = ImMetCorrTab.Gabor_squ_reg_b_1 / maxActiv
#%%%
xspace_cmp_plot(var=['Manif_uninorm_slope', 'Pasu_uninorm_slope', 'Gabor_uninorm_slope'], data=ImMetCorrTab,
                titstr="Normalized Slope of Activation and ImDistance Regression",
                cmap=cm.RdYlBu)
plt.ylabel("Regression slope / peak activation in all spaces")
plt.savefig(join(summarydir, "uninorm_slope_xspace_cmp.png"))
plt.savefig(join(summarydir, "uninorm_slope_xspace_cmp.pdf"))
plt.show()
#%%%
xspace_cmp_plot(var=['Manif_squ_reg_Rsq', 'Pasu_squ_reg_Rsq', 'Gabor_squ_reg_Rsq'], data=ImMetCorrTab,
                titstr="R square of Activation and ImDistance Regression",
                cmap=cm.RdYlBu)
plt.ylabel("R square")
plt.savefig(join(summarydir, "Rsquare_xspace_cmp.png"))
plt.savefig(join(summarydir, "Rsquare_xspace_cmp.pdf"))
plt.show()
#%%
xspace_cmp_plot(var=['Manif_squ_reg_F', 'Pasu_squ_reg_F', 'Gabor_squ_reg_F'], data=ImMetCorrTab,
                titstr="F statistics of Activation and ImDistance Regression",
                cmap=cm.RdYlBu)
plt.ylabel("F")
plt.savefig(join(summarydir, "regression_F_xspace_cmp.png"))
plt.savefig(join(summarydir, "regression_F_xspace_cmp.pdf"))
plt.show()