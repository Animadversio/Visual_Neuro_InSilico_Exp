""" Plot a few exemplar traj and plot summary for Convergence Speed"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from time import time
from os.path import join
from scipy.stats import linregress, ttest_ind, ttest_rel
from scipy.stats import spearmanr
from torchvision.transforms import ToPILImage
from GAN_utils import upconvGAN
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune"
summarydir = join(rootdir, "summary")
tau_full_tab = pd.read_csv(join(summarydir, "optim_Timescale_tab_robust.csv"),index_col=0)
#%%
traj_dir = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new\summary\example_traj"
outdir = r"E:\OneDrive - Washington University in St. Louis\Manuscript_Manifold\FigureS3"
#%%
gen_mask = ((tau_full_tab.suffix != "") | ((tau_full_tab.suffix == "") & tau_full_tab.layer.str.contains("fc"))) & \
            (tau_full_tab.GAN == "fc6") & (tau_full_tab.score > 2) & (tau_full_tab.optimizer == 'CholCMA_fc6')
figh = plt.figure(figsize=[6,6])
axs = sns.violinplot(x="layer", y="tau_e", data=tau_full_tab[gen_mask], saturation=0.6)
for viol, in zip(axs.collections[::2]):
    viol.set_alpha(0.6)
figh.savefig(join(traj_dir, "ConvgSpeed_summary_tau632.pdf"))
figh.savefig(join(traj_dir, "ConvgSpeed_summary_tau632.png"))
figh.savefig(join(outdir, "ConvgSpeed_summary_tau632.pdf"))
figh.savefig(join(outdir, "ConvgSpeed_summary_tau632.png"))
plt.show()
#%%

#%% Summarize the progression and the mean and sem of each layer.
def testProgression(valvec, labvec, valname="Y", labname="X"):
    cval, pval = spearmanr(labvec, valvec)
    statstr = "Spearman Correlation of %s - %s %.3f (%.1e)\n" % (valname, labname, cval, pval)
    slope, intercept, r_val, p_val, stderr = linregress(labvec, valvec)
    statstr += "%s value vs %s:\ntau_e = layerN * %.3f + %.3f (slope ste=%.3f)\nR2=%.3f slope!=0 " \
              "p=%.1e N=%d" % (valname, labname, slope, intercept, stderr, r_val, p_val, len(labvec))
    print(statstr)
    return statstr

layermap = {nm: i for i, nm in enumerate(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'])}
statstr = testProgression(tau_full_tab[gen_mask].tau_e, tau_full_tab[gen_mask].layer.map(layermap), "tau_e",
                             "layer num")
#%% Summarize mean and std
avg_arr = tau_full_tab[gen_mask].groupby("layer").mean()["tau_e"]
std_arr = tau_full_tab[gen_mask].groupby("layer").std()["tau_e"]
print("%s value per %s mean+-std"%("tau_e", "layer"))
for layer, val in avg_arr.items():
    print("%s %.2f+-%.2f"%(layer,avg_arr[layer], std_arr[layer]))

#%%
# Mask for all the succeeded resized evolutions in FC6 GAN.
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

#%%
G = upconvGAN("fc6")
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
#%% Summary plot violin 
tau_e