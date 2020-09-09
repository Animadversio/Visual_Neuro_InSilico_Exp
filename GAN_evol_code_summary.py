"""
This script models BigGAN Evol summary code to analyze the newest version of in silico evolutions.
It dedicates to understand the distribution of evolved codes in the space, esp. in the frame of Hessian eigenvectors.

"""

import os
import re
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import tqdm
from time import time
from os.path import join
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
#%%
rootdir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
summarydir = join(rootdir, "summary")
os.makedirs(summarydir, exist_ok=True)
#%%
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
        npzname = join(unit_str, "scores%s_%05d.npz" % (parts[0], int(parts[1]))) #
        # scoresHessCMA500_1_fc6_98508.npz
        entry = (unit_str, *unit_tup, parts[0], GANname, int(parts[1]), float(parts[2]), npzname)
        rec_col.append(entry)

exprec_tab = pd.DataFrame(rec_col, columns=["unitstr", 'net', 'layer', 'unit', 'suffix', "optimizer", "GAN", "RND",
                                            "score", "npzname"])
#%%
fc6idx = (exprec_tab.GAN=="fc6").nonzero()[0]#%%
FC6_exprec_tab = exprec_tab.iloc[fc6idx].copy()
FC6_exprec_tab = FC6_exprec_tab.reset_index()
#%%
T0 = time()
meancode_col = []
meanscore_col = []
for rowi in fc6idx:
    data = np.load(join(rootdir, exprec_tab.npzname[rowi]))
    mask = (data["generations"] == data["generations"].max())
    meanscore = np.mean(data['scores_all'][mask])
    meanscore_col.append(meanscore)
    if 'codes_all' in data:
        meancode = np.mean(data['codes_all'][mask, :], axis=0)
        meancode_col.append(meancode)
    elif 'codes_fin' in data:
        meancode = np.mean(data['codes_fin'][-40:, :], axis=0)
        meancode_col.append(meancode)
print("%.1f sec"%(time()-T0))
#%%
meanscores = np.array(meanscore_col)
meancodes = np.array(meancode_col)
np.savez(join(summarydir, "FC6_evol_mean_code.npz"), meancodes=meancodes, meanscores=meanscores, rowidx=fc6idx,
         exptab=exprec_tab)
#%%
with np.load(join(summarydir, "FC6_evol_mean_code.npz")) as data:
    meancodes = data['meancodes']
    meanscores = data['meanscores']
    rowidx = data["rowidx"]
#%%
Hessdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
data = np.load(join(Hessdir, "Texture_Avg_Hess.npz"))#,eigv_avg=eigv_avg, eigvect_avg=eigvect_avg, H_avg=H_avg)
eigval = data["eigv_avg"].copy()
eigvec = data["eigvect_avg"].copy()
# eigval = eigval[::-1]
# eigvec = eigvec[:, ::-1]
#%%
CholRel_idx = (FC6_exprec_tab.optimizer=='CholCMA_fc6').nonzero()[0]
#%%
shufcodes = np.array([meancodes[i, np.random.permutation(4096)].copy() for i in range(meancodes.shape[0])])
#%%
evol_proj_cc = meancodes[CholRel_idx,:] @ eigvec
shuf_proj_cc = shufcodes[CholRel_idx,:] @ eigvec
evol_proj_cc = np.flip(evol_proj_cc, axis=1)
shuf_proj_cc = np.flip(shuf_proj_cc, axis=1)
#%%

plt.figure()
plt.hist(evol_proj_cc[:, -1], 50, alpha=0.5)
plt.hist(shuf_proj_cc[:, -1], 50, alpha=0.5)
plt.show()
#%%
from scipy.stats import ks_2samp, kstest, ttest_rel, ttest_ind
eigid = 1
stat_col = []
for eigid in tqdm(range(4096)):
    ksstat = ks_2samp(np.abs(evol_proj_cc[:, eigid]), np.abs(shuf_proj_cc[:, eigid]))
    tstat = ttest_ind(evol_proj_cc[:, eigid], shuf_proj_cc[:, eigid])
    tstat_amp = ttest_ind(np.abs(evol_proj_cc[:, eigid]), np.abs(shuf_proj_cc[:, eigid]))
    stat_col.append((ksstat.statistic, ksstat.pvalue, tstat.statistic, tstat.pvalue, tstat_amp.statistic,
                     tstat_amp.pvalue))
#%%
stat_tab = pd.DataFrame(stat_col, columns=["KS", "KS_p", "T", "T_p","T_abs","T_abs_p"])
stat_tab.to_csv(join(summarydir, "Eigvect_Proj_cc_cmp.csv"))
#%%
plt.figure()
plt.scatter(range(4096), stat_tab['T'], 9, alpha=0.3, label="Ttest(coef_Evol, coef_Shuf)")
plt.scatter(range(4096), stat_tab['T_abs'], 9, alpha=0.3, label="Ttest(ampl_Evol, ampl_Shuf)")
plt.ylabel("T statistics")
plt.xlabel("eigen index")
plt.title("Comparison of Projection Coefficients on Hessian Eigen Vectors")
plt.legend()
plt.savefig(join(summarydir, "eigvect_proj_cmp_Ttest.png"))
plt.savefig(join(summarydir, "eigvect_proj_cmp_Ttest.pdf"))
plt.show()
#%%
plt.figure()
plt.scatter(range(4096), stat_tab['KS'], 9, alpha=0.3, label="KS(coef_Evol, coef_Shuf)")
plt.ylabel("Kolmo-Smirov statistics")
plt.xlabel("eigen index")
plt.title("Comparison of Projection Coefficients on Hessian Eigen Vectors")
plt.legend()
plt.savefig(join(summarydir, "eigvect_proj_cmp_KStest.png"))
plt.savefig(join(summarydir, "eigvect_proj_cmp_KStest.pdf"))
plt.show()
#%%
xxtick = np.repeat(np.arange(4096)[np.newaxis, :], evol_proj_cc.shape[0], axis=0)
plt.figure()
plt.scatter(xxtick.flatten(), evol_proj_cc.flatten(), 5, alpha=0.01)
plt.scatter(xxtick.flatten(), shuf_proj_cc.flatten(), 5, alpha=0.01)
plt.show()
#%%
evol_proj_men = np.mean(evol_proj_cc, axis=0)
evol_proj_std = np.std(evol_proj_cc, axis=0)
evol_proj_sem = evol_proj_std / np.sqrt(evol_proj_cc.shape[0])
shuf_proj_men = np.mean(shuf_proj_cc, axis=0)
shuf_proj_std = np.std(shuf_proj_cc, axis=0)
shuf_proj_sem = shuf_proj_std / np.sqrt(evol_proj_cc.shape[0])
plt.figure()
plt.scatter(range(4096), evol_proj_men,7,alpha=0.4)
plt.scatter(range(4096), shuf_proj_men,7,alpha=0.4)
plt.fill_between(range(4096), evol_proj_men-2*evol_proj_sem, evol_proj_men+2*evol_proj_sem)#, alpha=0.2)
plt.fill_between(range(4096), shuf_proj_men-2*shuf_proj_sem, shuf_proj_men+2*shuf_proj_sem)#, alpha=0.2)
plt.show()
#%%
plt.figure()
plt.errorbar(range(4096), evol_proj_men,evol_proj_sem, alpha=0.4, fmt='.', label="evolved coef")
plt.errorbar(range(4096), shuf_proj_men,shuf_proj_sem, alpha=0.4, fmt='.', label="shuffle coef")
plt.title("Comparison of Projection Coefficients on Hessian Eigen Vectors\n Point=Mean, Errorbar=SEM")
plt.ylabel("Project Coefficient")
plt.legend()
plt.savefig(join(summarydir, "eigvect_proj_errorbar.png"))
plt.savefig(join(summarydir, "eigvect_proj_errorbar.pdf"))
plt.show()
#%%
"""Apply this analysis on the BigGAN"""
#%%
bganidx = (exprec_tab.GAN=="BigGAN").nonzero()[0]
BGAN_exprec_tab = exprec_tab.iloc[bganidx].copy()
BGAN_exprec_tab = BGAN_exprec_tab.reset_index()
#%%
T0 = time()
bgancode_col = []
bganscore_col = []
for rowi in tqdm(bganidx):
    # rowi = bganidx[i]
    data = np.load(join(rootdir, exprec_tab.npzname[rowi]))
    mask = (data["generations"] == data["generations"].max())
    meanscore = np.mean(data['scores_all'][mask])
    bganscore_col.append(meanscore)
    if 'codes_all' in data:
        meancode = np.mean(data['codes_all'][mask, :], axis=0)
        bgancode_col.append(meancode)
    elif 'codes_fin' in data:
        meancode = np.mean(data['codes_fin'][-40:, :], axis=0)
        bgancode_col.append(meancode)
print("%.1f sec"%(time()-T0))

bganscores = np.array(bganscore_col)
bgancodes = np.array(bgancode_col)
np.savez(join(summarydir, "BigGAN_evol_mean_code.npz"), bgancodes=bgancodes, bganscores=bganscores, rowidx=bganidx,
         exptab=BGAN_exprec_tab)
#%%

Hessdir2 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
data = np.load(join(Hessdir2, "H_avg_1000cls.npz"))#,eigv_avg=eigv_avg, eigvect_avg=eigvect_avg, H_avg=H_avg)
eigval_B = data["eigvals_avg"].copy()
eigvec_B = data["eigvects_avg"].copy()
# eigval = eigval[::-1]
# eigvec = eigvec[:, ::-1]
#%%
shufcodes_B = np.array([bgancodes[i, np.random.permutation(256)].copy() for i in range(bgancodes.shape[0])])
shufcodes_sep_B = np.array([bgancodes[i, np.hstack((np.random.permutation(128), 128+np.random.permutation(
    128)))].copy()  for i in range(bgancodes.shape[0])])
#%%
CMArows = (BGAN_exprec_tab.optimizer == "CholCMA").nonzero()[0]
#%%
evol_proj_cc_B = bgancodes[CMArows,:] @ eigvec_B
shuf_proj_cc_B = shufcodes_B[CMArows,:] @ eigvec_B
shuf_proj_cc_sep_B = shufcodes_sep_B[CMArows,:] @ eigvec_B
evol_proj_cc_B = np.flip(evol_proj_cc_B, axis=1)
shuf_proj_cc_B = np.flip(shuf_proj_cc_B, axis=1)
shuf_proj_cc_sep_B = np.flip(shuf_proj_cc_sep_B, axis=1)
evol_proj_men_B = np.mean(evol_proj_cc_B,0)
evol_proj_sem_B = np.std(evol_proj_cc_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
shuf_proj_men_B = np.mean(shuf_proj_cc_B,0)
shuf_proj_sem_B = np.std(shuf_proj_cc_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
shuf_proj_men_sep_B = np.mean(shuf_proj_cc_sep_B,0)
shuf_proj_sem_sep_B = np.std(shuf_proj_cc_sep_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
#%%
plt.figure()
plt.errorbar(range(256), evol_proj_men_B,evol_proj_sem_B, alpha=0.4, fmt='.', label="evolved coef")
plt.errorbar(range(256), shuf_proj_men_B,shuf_proj_sem_B, alpha=0.4, fmt='.', label="shuffle coef")
plt.errorbar(range(256), shuf_proj_men_sep_B,shuf_proj_sem_sep_B, alpha=0.4, fmt='.', label="shuffle coef (separate)")
plt.title("Comparison of Projection Coefficients on Hessian Eigen Vectors\n Point=Mean, Errorbar=SEM")
plt.ylabel("Project Coefficient")
plt.legend()
plt.savefig(join(summarydir, "BigGAN_eigvect_proj_errorbar.png"))
plt.savefig(join(summarydir, "BigGAN_eigvect_proj_errorbar.pdf"))
plt.show()
#%%
CMArows = (BGAN_exprec_tab.optimizer == "CholCMA_class").nonzero()[0]
#%%
evol_proj_cc_B = bgancodes[CMArows,:] @ eigvec_B
shuf_proj_cc_B = shufcodes_B[CMArows,:] @ eigvec_B
shuf_proj_cc_sep_B = shufcodes_sep_B[CMArows,:] @ eigvec_B
evol_proj_cc_B = np.flip(evol_proj_cc_B, axis=1)
shuf_proj_cc_B = np.flip(shuf_proj_cc_B, axis=1)
shuf_proj_cc_sep_B = np.flip(shuf_proj_cc_sep_B, axis=1)
evol_proj_men_B = np.mean(evol_proj_cc_B,0)
evol_proj_sem_B = np.std(evol_proj_cc_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
shuf_proj_men_B = np.mean(shuf_proj_cc_B,0)
shuf_proj_sem_B = np.std(shuf_proj_cc_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
shuf_proj_men_sep_B = np.mean(shuf_proj_cc_sep_B,0)
shuf_proj_sem_sep_B = np.std(shuf_proj_cc_sep_B,0) / np.sqrt(evol_proj_cc_B.shape[0])
#%%
plt.figure()
plt.errorbar(range(256), evol_proj_men_B,evol_proj_sem_B, alpha=0.4, fmt='.', label="evolved coef")
plt.errorbar(range(256), shuf_proj_men_B,shuf_proj_sem_B, alpha=0.4, fmt='.', label="shuffle coef")
plt.errorbar(range(256), shuf_proj_men_sep_B,shuf_proj_sem_sep_B, alpha=0.4, fmt='.', label="shuffle coef (separate)")
plt.title("Comparison of Projection Coefficients on Hessian Eigen Vectors\n Point=Mean, Errorbar=SEM")
plt.ylabel("Project Coefficient")
plt.legend()
plt.savefig(join(summarydir, "BigGAN_class_eigvect_proj_errorbar.png"))
plt.savefig(join(summarydir, "BigGAN_class_eigvect_proj_errorbar.pdf"))
plt.show()

