"""Code for making the figure of comparing Optimizers"""
from os.path import join
import os
from sys import platform
import numpy as np
from insilico_Exp import ExperimentEvolve
from Optimizer import Genetic, CholeskyCMAES, Optimizer
import utils
#%% Decide the result storage place based on the computer the code is running
if platform == "linux": # cluster
    recorddir = "/scratch/binxu/CNN_data/"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
        initcodedir = r"D:\Generator_DB_Windows\stimuli\texture006"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        recorddir = r"D:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
#%
unit_arr = [ ('caffe-net', 'conv1', 5, 10, 10),
             ('caffe-net', 'conv2', 5, 10, 10),
             ('caffe-net', 'conv3', 5, 10, 10),
             ('caffe-net', 'conv4', 5, 10, 10),
             ('caffe-net', 'conv5', 5, 10, 10),
             ('caffe-net', 'fc6', 1),
             ('caffe-net', 'fc7', 1),
             ('caffe-net', 'fc8', 1), ]
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
codes, _ = utils.load_codes2(initcodedir, 40)
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

#%% Plot the comparison figure
import matplotlib
import matplotlib.pylab as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
