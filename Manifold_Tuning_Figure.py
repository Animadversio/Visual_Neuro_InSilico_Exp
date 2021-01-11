"""
Experiment code is this one and associated bash
    insilico_manifold_vgg16.sh
    insilico_ResizeManifold_script.py
    insilico_manifold_resize.sh
    insilico_manifold_resize_vgg16.sh
"""
#

import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel,ttest_ind
from os.path import join, exists
from os import listdir
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
netname = "caffe-net"
result_dir = r"E:\OneDrive - Washington University in St. Louis\Artiphysiology\Manifold"
with np.load(join(result_dir, "%s_KentFit.npz"%netname)) as data:
    param_col_arr = data["param_col"]  # ["theta", "phi", "psi", "kappa", "beta", "A"]
    sigma_col_arr = data["sigma_col"]  # ["theta", "phi", "psi", "kappa", "beta", "A"]
    stat_col_arr = data["stat_col"]  # "r2"
    layers = data["layers"]
    subsp_axis = data["subsp_axis"]
#
with np.load(join(result_dir,"%s_KentFit_rf_fit.npz" % netname)) as data:
    param_col_arr_rf = data["param_col"]
    sigma_col_arr_rf = data["sigma_col"]
    stat_col_arr_rf = data["stat_col"]
#%%
print("Process Manifold Exps on %s"%netname)
print(layers)
print("Manif in %d subspaces: "%len(subsp_axis))
print(subsp_axis)
#%%
alltab = []
subsp_nm = ["PC23","PC2526","PC4950","RND12"]
for li in range(param_col_arr.shape[0]):
    for ui in range(param_col_arr.shape[1]):
        for si in range(param_col_arr.shape[2]):
            alltab.append([layers[li],ui,si,subsp_nm[si],stat_col_arr[li,ui,si]] \
                          + list(param_col_arr[li,ui,si,:]) + list(sigma_col_arr[li,ui,si,:]))
alltab = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2",\
            "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
#%%
rftab = []
layers_rf = layers[:5]
for li in range(param_col_arr_rf.shape[0]):
    for ui in range(param_col_arr_rf.shape[1]):
        for si in range(param_col_arr_rf.shape[2]):
            rftab.append([layers_rf[li],ui,si,subsp_nm[si],stat_col_arr_rf[li,ui,si]] \
                          + list(param_col_arr_rf[li,ui,si,:]) + list(sigma_col_arr_rf[li,ui,si,:]))
rftab = pd.DataFrame(rftab, columns=["Layer","unit","spacenum","spacename","R2",\
            "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
#%%
from scipy.stats import ttest_rel, ttest_ind
figdir = "E:\OneDrive - Washington University in St. Louis\Manuscript_Manifold\Figure3"
mskrf = (rftab.R2 > 0.5)*(rftab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=rftab[mskrf], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.savefig(join(figdir, "%s_kappaRF_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaRF_pur_violin.pdf"%netname))
plt.show()

msk = (alltab.R2 > 0.5)*(alltab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=alltab[msk], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.legend()
plt.savefig(join(figdir, "%s_kappa_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappa_pur_violin.pdf"%netname))
plt.show()

mskrf = (rftab.R2 > 0.5)*(rftab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=rftab[mskrf], name=layers, dodge=True,
            inner="point", meanline_visible=True)
msk = (alltab.R2 > 0.5)*(alltab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=alltab[msk], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.savefig(join(figdir, "%s_kappaRF_cmb_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaRF_cmb_violin.pdf"%netname))
plt.show()
ttest_ind(alltab.kappa[msk*alltab.Layer=="fc7"], alltab.kappa[msk*alltab.Layer=="conv1"],nan_policy='omit')