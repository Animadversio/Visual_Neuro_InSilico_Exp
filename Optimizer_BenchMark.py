from insilico_Exp import *
import time
import sys
import os
from os import makedirs
from os.path import join
import utils
import numpy as np
from numpy.random import randint
from numpy.linalg import norm
from numpy import sqrt, zeros, abs, floor, log, log2, eye
from numpy.random import randn
orig_stdout = sys.stdout
from ZO_HessAware_Optimizers import CholeskyCMAES, HessAware_Gauss_Cylind, HessAware_Gauss_Spherical, HessAware_Gauss_Cylind_DC

# %% HessAware_Gauss_Spherical Testing
unit = ('caffe-net', 'fc8', 1)
savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
expdir = join(savedir, "%s_%s_%d_BenchMark" % unit)
makedirs(expdir, exist_ok=True)
# lr=0.25; mu=0.01; Lambda=0.99; trial_i=0
# UF = 200
# lr_list = [1,2] # lr_list = [0.1, 0.05, 0.5, 0.25, 0.01]
# mu_list = [0.01, 0.005] # mu_list = [0.01, 0.02, 0.005, 0.04, 0.002, 0.001]
# Lambda_list = [1]
# for i, Lambda in enumerate(Lambda_list):
#     for j, lr in enumerate(lr_list):
#         for k, mu in enumerate(mu_list):
#             idxno = k + j * len(mu_list) + i * len(lr_list) * len(mu_list)
#             for trial_i in range(3):
trial_i = 0
# optim = HessAware_Gauss_Spherical(4096, population_size=40, lr=lr, mu=mu, Lambda=Lambda, Hupdate_freq=UF, sphere_norm=300, maximize=True)
optim = HessAware_Gauss_Cylind(4096, population_size=40, lr_norm=5, mu_norm=10, lr_sph=2, mu_sph=0.005,
                Lambda=1, Hupdate_freq=201, max_norm=300, maximize=True, rankweight=True)
# optim = CholeskyCMAES(4096, init_sigma=3.0, Aupdate_freq=10)
optim_name = str(optim.__class__).split(".")[1].split("'")[0]
fn_str = "%s_lrnorm%.1f_munorm%.4f_lrsph%.1f_musph%.4f_Lambda%.2f_%s_tr%03d" % (optim_name,
   optim.lr_norm, optim.mu_norm, optim.lr_sph, optim.mu_sph, optim.Lambda, "rank" if optim.rankweight else "grad", randint(1000))
# fn_str = "%s_initsgm%.1f_UF%d_tr%d" % (optim_name, optim.sigma, optim.update_crit, trial_i)
f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
# sys.stdout = f
experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
experiment.run(init_code=3*np.random.randn(1, 4096))
param_str = "%s lr_norm=%.1f, mu_norm=%.2f, \nlr_sph=%.1f, mu_sph=%.4f, Lambda=%.2f. %s" % (optim_name,
   optim.lr_norm, optim.mu_norm, optim.lr_sph, optim.mu_sph, optim.Lambda, "rank" if optim.rankweight else "grad")
# param_str = "%s init sigma=%.1f, UpdateCriterion=%.2f" % (optim_name, optim.sigma, optim.update_crit)
fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
fig2 = experiment.visualize_best(show=False)# , title_str=param_str)
fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
fig3 = experiment.visualize_exp(show=False, title_str=param_str)
fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
time.sleep(5)
plt.close('all')
# sys.stdout = orig_stdout
f.close()
np.savez(join(expdir, fn_str+"_data.npz"), codes_all=experiment.codes_all, scores_all=experiment.scores_all, generations=experiment.generations)
#%%
unit = ('caffe-net', 'fc8', 1)
savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
expdir = join(savedir, "%s_%s_%d_BigGAN_BenchMark" % unit)
makedirs(expdir, exist_ok=True)
# optim = HessAware_Gauss_Spherical(4096, population_size=40, lr=lr, mu=mu, Lambda=Lambda, Hupdate_freq=UF, sphere_norm=300, maximize=True)
# optim = HessAware_Gauss_Cylind(256, population_size=40, lr_norm=5, mu_norm=10, lr_sph=2, mu_sph=0.005,
#                 Lambda=1, Hupdate_freq=201, max_norm=300, maximize=True, rankweight=True)
optim = CholeskyCMAES(128, init_sigma=0.1, Aupdate_freq=10)
optim_name = str(optim.__class__).split(".")[1].split("'")[0]
# fn_str = "%s_lrnorm%.1f_munorm%.4f_lrsph%.1f_musph%.4f_Lambda%.2f_%s_tr%03d" % (optim_name,
#    optim.lr_norm, optim.mu_norm, optim.lr_sph, optim.mu_sph, optim.Lambda, "rank" if optim.rankweight else "grad", randint(1000))
fn_str = "%s_initsgm%.3f_UF%d_128_tr%03d" % (optim_name, optim.sigma, optim.update_crit, randint(1000))
f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
# sys.stdout = f
experiment = ExperimentEvolve(unit, max_step=150, optimizer=optim, GAN="BigGAN")
experiment.run(init_code=zeros((1, 128)))
# param_str = "%s lr_norm=%.1f, mu_norm=%.2f, \nlr_sph=%.1f, mu_sph=%.4f, Lambda=%.2f. %s" % (optim_name,
#    optim.lr_norm, optim.mu_norm, optim.lr_sph, optim.mu_sph, optim.Lambda, "rank" if optim.rankweight else "grad")
param_str = "%s init sigma=%.3f, UpdateCriterion=%.2f" % (optim_name, optim.sigma, optim.update_crit)
fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
fig2 = experiment.visualize_best(show=False)# , title_str=param_str)
fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
fig3 = experiment.visualize_exp(show=False, title_str=param_str)
fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
time.sleep(5)
plt.close('all')
# sys.stdout = orig_stdout
f.close()
np.savez(join(expdir, fn_str+"_data.npz"), codes_all=experiment.codes_all, scores_all=experiment.scores_all, generations=experiment.generations)
