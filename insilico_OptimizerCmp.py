"""Code for experiment and figure of comparing Optimizers"""
from os.path import join
import os
from sys import platform
import numpy as np
from insilico_Exp import ExperimentEvolve
from Optimizer import Genetic, CholeskyCMAES, Optimizer
import matplotlib.pylab as plt
import utils_old
#%% Decide the result storage place based on the computer the code is running
if platform == "linux": # cluster
    recorddir = "/scratch/binxu/CNN_data/"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
        initcodedir = r"D:\Generator_DB_Windows\stimuli\texture006"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        recorddir = r"E:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
        initcodedir = r"E:\Monkey_Data\Generator_DB_Windows\stimuli\texture006"
#%
unit_arr = [ #('caffe-net', 'conv1', 5, 10, 10),
             #('caffe-net', 'conv2', 5, 10, 10),
             #('caffe-net', 'conv3', 5, 10, 10),
             #('caffe-net', 'conv4', 5, 10, 10),
             #('caffe-net', 'conv5', 5, 10, 10),
             ('caffe-net', 'fc6', 1),
             #('caffe-net', 'fc7', 1),
             #('caffe-net', 'fc8', 1),
            ]
Optim_arr = ["Genetic", "CholCMA"]
#%% Genetic Algorithm Parameters AND CMA-ES Parameters
population_size = 40
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75
# Genetic(population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
#          parental_skew=0.5, n_conserve=0)
code_length = 4096
init_sigma = 3
Aupdate_freq = 10
# use the mean code of the texture patterns as the initcode
codes, _ = utils_old.load_codes2(initcodedir, 40)
initcode = np.mean(codes, axis=0, keepdims=True)
# CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
#                   Aupdate_freq=Aupdate_freq, init_code=np.zeros([1, code_length]))
#%%
from time import time
for unit in unit_arr[:]:
    savedir = join(recorddir, "optim_cmp", "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    os.makedirs(savedir, exist_ok=True)
    for Optim_str in Optim_arr:
        for triali in range(10):
            t0 = time()
            if Optim_str == "Genetic":
                optim = Genetic(population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
                            parental_skew=0.5, n_conserve=0)
            elif Optim_str == "CholCMA":
                optim = CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
                            Aupdate_freq=Aupdate_freq, init_code=initcode) # np.zeros([1, code_length])
            experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
            experiment.run()
            fig0 = experiment.visualize_best(show=False)
            fig0.savefig(join(savedir, "%s_BestImgTrial%03d.png" % (Optim_str, triali,)))
            fig = experiment.visualize_trajectory(show=False)
            fig.savefig(join(savedir, "%s_ScoreTrajTrial%03d.png" % (Optim_str, triali,)))
            # fig2 = experiment.visualize_exp(show=False)
            # fig2.savefig(join(savedir, "EvolveTrial%03d.png" % (triali)))
            plt.close('all')
            np.savez(join(savedir, "%s_scores_trial%03d.npz" % (Optim_str, triali)),
                     generations=experiment.generations,
                     scores_all=experiment.scores_all)
            print("Optimization with %s took %.1f s" % (Optim_str, time() - t0))
            # lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
            #         range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
            # best_scores_col.append(lastgen_max)


#%% Visualization part. 
#%% Plot the comparison figure
import matplotlib as mpl
import matplotlib.pylab as plt
recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
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
unit = ('caffe-net', 'fc6', 1)
savedir = join(recorddir, "optim_cmp", "%s_%s_%d" % (unit[0], unit[1], unit[2]))
figh = plt.figure()
for Optim_str in Optim_arr:
    for triali in range(1):
        with np.load(join(savedir, "%s_scores_trial%03d.npz" % (Optim_str, triali))) as data:
            generations = data["generations"]
            scores_all = data["scores_all"]
        gen_slice = np.arange(min(generations), max(generations) + 1)
        AvgScore = np.zeros_like(gen_slice)
        MaxScore = np.zeros_like(gen_slice)
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(scores_all[generations == geni])
            MaxScore[i] = np.max(scores_all[generations == geni])
        plt.scatter(generations, scores_all, s=16, alpha=0.4, label=Optim_str + " %d" % triali)
        plt.plot(gen_slice, AvgScore, color='black')
        plt.plot(gen_slice, MaxScore, color='red')
plt.axis("tight")
plt.xlabel("generation", fontsize=16)
plt.ylabel("artificial \"neuron\" activation", fontsize=16) # ("CNN unit score")
# plt.title("Optimization Trajectory of Score\n")  # + title_str)
plt.legend()
plt.savefig("Optim_cmp.pdf", transparent=True)
#plt.show()

#%%
figdir = r"D:\Generator_DB_Windows\data\with_CNN\optim_cmp\summary"
outdir = r"O:\ThesisProposal\GA_CMA"
unit_arr = [ ('caffe-net', 'conv1', 5, 10, 10),
             ('caffe-net', 'conv2', 5, 10, 10),
             ('caffe-net', 'conv3', 5, 10, 10),
             ('caffe-net', 'conv4', 5, 10, 10),
             ('caffe-net', 'conv5', 5, 10, 10),
             ('caffe-net', 'fc6', 1),
             ('caffe-net', 'fc7', 1),
             ('caffe-net', 'fc8', 1), ]

color_arr = [u'#1f77b4', u'#ff7f0e']
Optim_arr = ["Genetic", "CholCMA"]

for unit in unit_arr[:]:
    figh = plt.figure(figsize=(5,5))
    savedir = join(recorddir, "optim_cmp", "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    for triali in range(10):
        for Optim_str, color in zip(Optim_arr,color_arr):
            with np.load(join(savedir, "%s_scores_trial%03d.npz" % (Optim_str, triali))) as data:
                generations = data["generations"]
                scores_all = data["scores_all"]
            score_m, score_s, blockarr = summary_by_block(scores_all, generations, sem=True)
            plt.plot(blockarr, score_m, color=color, alpha=0.8, label=Optim_str)
            plt.fill_between(blockarr, score_m - score_s, score_m + score_s, alpha=0.15, color=color)
    plt.axis("tight")
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("artificial \"neuron\" activation", fontsize=16) # ("CNN unit score")
    plt.title("Optimization Trajectory of Score\n%s %s Ch%d"%unit[:3])  # + title_str)
    plt.legend(Optim_arr)
    # plt.savefig("Optim_cmp.pdf", transparent=True)
    # plt.savefig()
    saveallforms(figdir, "optim_curv_cmp_%s_%s_Ch%d"%unit[:3])
    plt.show()

#%% Statitics
import pandas as pd
from easydict import EasyDict
score_col = [] 
for unit in unit_arr[:]:
    savedir = join(recorddir, "optim_cmp", "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    for triali in range(10):
        score_struct = EasyDict()
        score_struct.layer = unit[1]
        score_struct.chan = unit[2]
        score_struct.trial = triali
        for i, Optim_str in enumerate(Optim_arr):
            with np.load(join(savedir, "%s_scores_trial%03d.npz" % (Optim_str, triali))) as data:
                generations = data["generations"]
                scores_all = data["scores_all"]
            score_m, score_s, blockarr = summary_by_block(scores_all, generations, sem=True)
            score_struct[Optim_str] = score_m[98:].mean()
        score_col.append(score_struct)

scoretab = pd.DataFrame(score_col)
#%%
from contextlib import redirect_stdout
from scipy.stats import ttest_rel, ttest_ind
figdir = r"D:\Generator_DB_Windows\data\with_CNN\optim_cmp\summary"
outdir = r"O:\ThesisProposal\GA_CMA"
f1 = open(join(figdir, 'stat_summary.txt'), 'w')
for layer in scoretab.layer.unique():
    msk = scoretab.layer==layer
    tval, pval = ttest_rel(scoretab.CholCMA[msk], scoretab.Genetic[msk])
    act_m = [scoretab.CholCMA[msk].mean(), scoretab.Genetic[msk].mean()]
    act_s = [scoretab.CholCMA[msk].sem(), scoretab.Genetic[msk].sem()]
    print("Layer %s, CMA %.2f+-%.2f vs GA %.2f+-%.2f, R=%.3f; t=%.3f, P=%.1e"%(layer, act_m[0], act_s[0], act_m[1], act_s[1],
        act_m[1]/act_m[0], tval, pval, ))
    with redirect_stdout(f1):
        print("Layer %s, CMA %.2f+-%.2f vs GA %.2f+-%.2f, R=%.3f; t=%.3f, P=%.1e"%(layer, act_m[0], act_s[0], act_m[1], act_s[1],
        act_m[1]/act_m[0], tval, pval, ))

f1.close()
