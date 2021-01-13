"""
Experiment code is this one and associated bash
    insilico_manifold_vgg16.sh
    insilico_ResizeManifold_script.py
    insilico_manifold_resize.sh
    insilico_manifold_resize_vgg16.sh
"""
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
sumdir = "E:\OneDrive - Washington University in St. Louis\Artiphysiology\Manifold\summary"
subsp_nm = ["PC23","PC2526","PC4950","RND12"]
def array2table(param_col_arr, sigma_col_arr, stat_col_arr, layers, subsp_nm, param_name):
    alltab = []
    for li in range(param_col_arr.shape[0]):
        for ui in range(param_col_arr.shape[1]):
            for si in range(param_col_arr.shape[2]):
                alltab.append([layers[li],ui,si,subsp_nm[si],stat_col_arr[li,ui,si]] \
                              + list(param_col_arr[li,ui,si,:]) + list(sigma_col_arr[li,ui,si,:]))
    param_names = list(param_name)
    param_std_names = [p+"_std" for p in param_names]
    alltab = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2", ] + \
        param_names + param_std_names)
    return alltab

def add_regcurve(ax, slope, intercept, **kwargs):
    XLIM = ax.get_xlim()
    ax.plot(XLIM, np.array(XLIM) * slope + intercept, **kwargs)

netname = "vgg16"
with np.load(join(sumdir, "%s_KentFit_bsl_orig.npz"%netname)) as data:
    param_col_arr = data["param_col"]  # ["theta", "phi", "psi", "kappa", "beta", "A"]
    sigma_col_arr = data["sigma_col"]  # ["theta", "phi", "psi", "kappa", "beta", "A"]
    stat_col_arr = data["stat_col"]  # "r2"
    layers = data["layers"]
    subsp_axis = data["subsp_axis"]
    param_name = data["param_name"]
    fulltab = array2table(param_col_arr, sigma_col_arr, stat_col_arr, layers, subsp_nm, param_name)

with np.load(join(sumdir, "%s_KentFit_bsl_rf_fit.npz" % netname)) as data:
    param_col_arr_rf = data["param_col"]
    sigma_col_arr_rf = data["sigma_col"]
    stat_col_arr_rf = data["stat_col"]
    rftab = array2table(param_col_arr_rf, sigma_col_arr_rf, stat_col_arr_rf, layers, subsp_nm, param_name)

with np.load(join(sumdir, "%s_KentFit_bsl.npz" % netname)) as data:
    param_col_arr_fc = data["param_col"]
    sigma_col_arr_fc = data["sigma_col"]
    stat_col_arr_fc = data["stat_col"]
    layers_fc = data["layers"]
    fulltab_fc = array2table(param_col_arr_fc, sigma_col_arr_fc, stat_col_arr_fc, layers_fc, subsp_nm, param_name)

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
param_names = list(param_name)
param_std_names = [p+"_std" for p in param_names]
# alltab = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2", \
#             "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
alltab_bsl = pd.DataFrame(alltab, columns=["Layer","unit","spacenum","spacename","R2", ] + param_names +
                                          param_std_names)

rftab = []
layers_rf = [L for L in layers if "conv" in L]
for li in range(param_col_arr_rf.shape[0]):
    for ui in range(param_col_arr_rf.shape[1]):
        for si in range(param_col_arr_rf.shape[2]):
            rftab.append([layers_rf[li],ui,si,subsp_nm[si],stat_col_arr_rf[li,ui,si]] \
                          + list(param_col_arr_rf[li,ui,si,:]) + list(sigma_col_arr_rf[li,ui,si,:]))
# rftab = pd.DataFrame(rftab, columns=["Layer","unit","spacenum","spacename","R2",\
#             "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
rftab_bsl = pd.DataFrame(rftab, columns=["Layer","unit","spacenum","spacename","R2",]+param_names+param_std_names)
# "theta", "phi", "psi", "kappa", "beta", "A", "theta_std", "phi_std", "psi_std", "kappa_std", "beta_std", "A_std"])
#%%
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import linregress
figdir = "E:\OneDrive - Washington University in St. Louis\Manuscript_Manifold\Figure3"
#%%
#%%
mskrf = (rftab.R2 > 0.5)*(rftab.A > 1E-3)*(rftab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=rftab[mskrf], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.savefig(join(figdir, "%s_kappaRF_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaRF_bsl_pur_violin.pdf"%netname))
plt.show()

msk = (fulltab.R2 > 0.5)*(fulltab.A > 1E-3)*(fulltab.spacenum==0)
ax = sns.violinplot(x="Layer", y="kappa",
            data=fulltab[msk], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.savefig(join(figdir, "%s_kappaFull_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaFull_bsl_pur_violin.pdf"%netname))
plt.show()

fcmsk = (fulltab_fc.R2 > 0.5)*(fulltab_fc.A > 1E-3)*(fulltab_fc.spacenum==0)*(fulltab_fc.Layer.str.contains("fc"))
ax = sns.violinplot(x="Layer", y="kappa",
            data=fulltab_fc[fcmsk], name=layers, dodge=True,
            inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
plt.savefig(join(figdir, "%s_kappaFull_fc_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaFull_fc_bsl_pur_violin.pdf"%netname))
plt.show()
#%% Concatenate fc table and rf matched table.
mskrf = (rftab.R2 > 0.5)*(rftab.A > 1E-3)*(rftab.spacenum == 0)
fcmsk = (fulltab_fc.R2 > 0.5)*(fulltab_fc.A > 1E-3)*(fulltab_fc.spacenum == 0)*(fulltab_fc.Layer.str.contains("fc"))
layercattab = pd.concat((rftab[mskrf], fulltab_fc[fcmsk]), axis=0)
laynames = layercattab.Layer.unique()
layermap = {nm: i for i, nm in enumerate(laynames)}
layermap_inall = {'conv1':0,'conv2':1,'conv3':2,'conv4':3,'conv5':4,'conv6':5,'conv7':6,'conv8':7,'conv9':8,\
                  'conv10':9,'conv11':10,'conv12':11,'conv13':12,'fc1':13,'fc2':14,'fc3':15}
#%%
ax = sns.violinplot(x="Layer", y="kappa", name=layers, dodge=True,
            data=layercattab, inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
slope, intercept, r_val, p_val, stderr = linregress(layercattab["Layer"].map(layermap), layercattab.kappa)
statstr = "All layers Kappa value vs layer num:\nkappa = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(layercattab))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Kappa Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin.png"%netname))
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin.pdf"%netname))
plt.show()
#%%
ax = sns.violinplot(x="Layer", y="kappa", name=layers, dodge=True,
            data=layercattab, inner="box", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
slope, intercept, r_val, p_val, stderr = linregress(layercattab["Layer"].map(layermap), layercattab.kappa)
statstr = "All layers Kappa value vs layer num:\nkappa = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(layercattab))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Kappa Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin_box.png"%netname))
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin_box.pdf"%netname))
plt.show()
#%%
layernum = layercattab["Layer"].map(layermap_inall)
ax = sns.violinplot(x=layernum, y="kappa", name=layers, dodge=True,
            data=layercattab, inner="point", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
slope, intercept, r_val, p_val, stderr = linregress(layernum, layercattab.kappa)
statstr = "All layers Kappa value vs layer num:\nkappa = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(layercattab))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Kappa Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin_LayNumX.png"%netname))
plt.savefig(join(figdir, "%s_kappaFull_cmb_bsl_pur_violin_LayNumX.pdf"%netname))
plt.show()
#%% Try to see if there is any trend with 'beta'
ax = sns.violinplot(x="Layer", y="beta", name=layers, dodge=True,
            data=layercattab, inner="box", meanline_visible=True)
for violin in zip(ax.collections[::2]):
    violin[0].set_alpha(0.3)
slope, intercept, r_val, p_val, stderr = linregress(layercattab["Layer"].map(layermap), layercattab.beta)
statstr = "All layers Kappa value vs layer num:\nbeta = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
          "p=%.1e N=%d" % (slope, intercept, stderr, r_val, p_val, len(layercattab))
add_regcurve(ax, slope, intercept, alpha=0.5)
plt.title("CNN %s Manifold Exp Beta Progression\n"%netname+statstr)
plt.savefig(join(figdir, "%s_Beta_cmb_bsl_pur_violin_box.png"%netname))
plt.savefig(join(figdir, "%s_Beta_cmb_bsl_pur_violin_box.pdf"%netname))
plt.show()
#%% Get rid of outliers.
# plot the regression curve.
#%% Test the progression by linear regression.
laynames = layercattab.Layer.unique()
layermap = {nm: i for i, nm in enumerate(laynames)}
layernum = layercattab["Layer"].map(layermap)
print("Layer numbering in all computed:\n%s"%layermap)
slope, intercept, r_val, p_val, stderr = linregress(layernum[layernum<9], layercattab.kappa[layernum<9])
print("Conv layers Kappa value vs layer num: kappa = layerN * %.3f + %.3f (slope ste=%.3f) R2=%.3f slope!=0 p=%.1e N=%d" % \
      (slope, intercept, stderr, r_val, p_val, sum(layernum<9)))
slope, intercept, r_val, p_val, stderr = linregress(layernum, layercattab.kappa)
print("All layers Kappa value vs layer num: kappa = layerN * %.3f + %.3f (slope ste=%.3f) R2=%.3f slope!=0 p=%.1e N=%d" % \
      (slope, intercept, stderr, r_val, p_val, len(layernum)))
#%%
layerinvmap_inall = {0:'conv1',1:'conv2',2:'conv3',3:'conv4',4:'conv5',5:'conv6',6:'conv7',7:'conv8',8:'conv9',\
       9:'conv10',10:'conv11',11:'conv12',12:'conv13',13:'fc1',14:'fc2',15:'fc3'}
layermap_inall = {'conv1':0,'conv2':1,'conv3':2,'conv4':3,'conv5':4,'conv6':5,'conv7':6,'conv8':7,'conv9':8,\
                  'conv10':9,'conv11':10,'conv12':11,'conv13':12,'fc1':13,'fc2':14,'fc3':15}
layernum_inall = layercattab["Layer"].map(layermap_inall)
print("Layer numbering in all conv / fc layers:\n%s"%layermap_inall)
slope, intercept, r_val, p_val, stderr = linregress(layernum_inall[layernum_inall<13], layercattab.kappa[layernum_inall<13])
print("Conv layers Kappa value vs layer num: kappa = layerN * %.3f + %.3f (slope ste=%.3f) R2=%.3f slope!=0 p=%.1e N=%d" % \
      (slope, intercept, stderr, r_val, p_val, sum(layernum_inall<13)))
slope, intercept, r_val, p_val, stderr = linregress(layernum_inall, layercattab.kappa)
print("All layers Kappa value vs layer num: kappa = layerN * %.3f + %.3f (slope ste=%.3f) R2=%.3f slope!=0 p=%.1e N=%d" % \
      (slope, intercept, stderr, r_val, p_val, len(layernum_inall)))

#%% Obsolete....
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