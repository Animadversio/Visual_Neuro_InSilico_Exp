import numpy as np
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid
from insilico_Exp import ExperimentEvolve
#%%
mode_dict = ["inv","lin","exp"]
def optim_result(param):
    pop_size = int(param[0,0])
    select_rate = param[0,1]
    select_size = int(pop_size * select_rate)
    lr = param[0,2]
    mu_init = param[0,3]
    mu_rate = param[0,4]
    mode = int(param[0,5])
    n_gen = 4000 // pop_size # 4000 is the total evaluation budget
    optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=pop_size, select_size=select_size, lr=lr)
    optimizer.lr_schedule(n_gen=n_gen, lim=(mu_init, mu_rate*mu_init), mode=mode_dict[mode])
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
    return np.percentile(Exp.scores_all, 99.5)

#%%
def optim_result_short(param):
    pop_size = int(param[0,0])
    select_rate = param[0,1]
    select_size = int(pop_size * select_rate)
    lr = param[0,2]
    mu_init = param[0,3]
    mu_rate = param[0,4]
    mode = int(param[0,5])
    n_gen = 2000 // pop_size # 4000 is the total evaluation budget
    optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=pop_size, select_size=select_size, lr=lr)
    optimizer.lr_schedule(n_gen=n_gen, lim=(mu_init, mu_rate*mu_init), mode=mode_dict[mode])
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
    return np.percentile(Exp.scores_all, 99.5)
#%%
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
mixed_domain =[{'name': 'pop_size', 'type': 'continuous', 'domain': (20, 50),'dimensionality': 1},
                {'name': 'select_rate', 'type': 'continuous', 'domain': (0.1, 0.9),'dimensionality': 1},
                {'name': 'lr', 'type': 'continuous', 'domain': (0.5, 2),'dimensionality': 1},
               {'name': 'mu_init', 'type': 'continuous', 'domain': (40, 80),'dimensionality': 1},
               {'name': 'mu_rate', 'type': 'continuous', 'domain': (0.15, 0.9),'dimensionality': 1},
               {'name': 'mode', 'type': 'categorical', 'domain': (0,1,2),'dimensionality': 1},]
myBopt = BayesianOptimization(f=optim_result,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 10,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='EI',        # Expected Improvement
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=True)
#%% 8 hour 20 min 19 sec 811 iterations
%%time
max_iter = 900       ## maximum number of iterations
max_time = 30000      ## maximum allowed time
eps      = 1e-3     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=0, verbosity=True)

#%%
from os.path import join
savedir = "E:\OneDrive - Washington University in St. Louis\Optimizer_Tuning\BO_autotune";
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_tune.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best, domain=mixed_domain, mode_dict=mode_dict)
#%%
import pandas as pd
scores_tab = pd.DataFrame(np.append(myBopt.X,-myBopt.Y,1), columns=["pop_size","select_ratio","lr","mu_init","mu_rate","mode","scores"])
scores_tab.to_csv(join(savedir, "ZOHA_Sphere_lr_euclid_BO_tune.csv"))
#%%
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_obj.npz"), BOobj=myBopt) # not successful since there some local functions in BO object.
#%%
%%time
myBopt_short = BayesianOptimization(f=optim_result_short,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 20,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='LCB',        # Expected Improvement # "LCB" seems to create more variablity than "EI"
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=True)
#%%  8 hour 20 min 19 sec 811 iterations
max_iter = 400       ## maximum number of iterations
max_time = 10000      ## maximum allowed time
eps      = 1e-2     ## tolerance, max distance between consicutive evaluations.
myBopt_short.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
#
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_short_tune.npz"), X=myBopt_short.X, Y=myBopt_short.Y, Y_best=myBopt_short.Y_best, domain=mixed_domain, mode_dict=mode_dict)
scores_short_tab = pd.DataFrame(np.append(myBopt_short.X,-myBopt_short.Y,1), columns=["pop_size","select_ratio","lr","mu_init","mu_rate","mode","scores"])
scores_short_tab.to_csv(join(savedir, "ZOHA_Sphere_lr_euclid_BO_short_tune.csv"))