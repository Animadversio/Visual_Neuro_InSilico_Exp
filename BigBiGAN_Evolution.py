from insilico_Exp import ExperimentEvolve
from ZO_HessAware_Optimizers import CholeskyCMAES
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid
import numpy as np
#%%
from BigBiGAN import BigBiGAN_render
from build_montages import build_montages
from geometry_utils import SLERP, LERP
from PIL import Image
imgs = BigBiGAN_render(0.7*np.random.randn(20, 120), 255.0)
mtg = build_montages(imgs, (128, 128), (5, 5))[0]
Img = Image.fromarray(np.uint8(mtg))
Img.show()
#%% Measure how fast image change in the space.
sphere_norm = 9
Vecs = 0.7*np.random.randn(2, 120)
Q, R = np.linalg.qr(Vecs.T)
Vec1 = Q.T[0:1, :] * sphere_norm
Vec2 = Q.T[1:2, :] * sphere_norm
#%%
ticks = 21
imgs = BigBiGAN_render(SLERP(Vec1, Vec2, ticks, (0, 1)), 255.0)
mtg = build_montages(imgs, (128, 128), (ticks, 1))[0]
Img = Image.fromarray(np.uint8(mtg))
Img.show()
#%%
optimizer = CholeskyCMAES(120, population_size=None, init_sigma=0.25, init_code=0.8*np.random.randn(1, 120),
                          Aupdate_freq=10, maximize=True, random_seed=None, optim_params={})
experiment = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizer, GAN="BigBiGAN",
                       verbose=True)
experiment.run()
#%%
optimizerZH = ZOHA_Sphere_lr_euclid(120, population_size=40, select_size=15, lr=1.0,
                 maximize=True, rankweight=True, rankbasis=True, sphere_norm=9)
optimizerZH.lr_schedule(100, lim=(10, 6), mode="lin") # the space is much tighter than the fc6 space, so step size
# should be tuned for that
experiment = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizerZH, GAN="BigBiGAN",
                       verbose=True)
experiment.run()
experiment.visualize_trajectory(True)
experiment.visualize_exp(True)
experiment.visualize_best(True)
#%%
mode_dict = ["lin","exp"] #"inv",
def optim_result(param):
    pop_size = int(param[0,0])
    select_rate = param[0,1]
    select_size = int(pop_size * select_rate)
    lr = param[0,2]
    mu_init = param[0,3]
    mu_rate = param[0,4]
    mode = int(param[0,5])
    sphere_norm = int(param[0,6])
    n_gen = 3000 // pop_size # 4000 is the total evaluation budget
    optimizer = ZOHA_Sphere_lr_euclid(120, population_size=pop_size, select_size=select_size, lr=lr,
                        maximize=True, rankweight=True, rankbasis=True, sphere_norm=sphere_norm)
    optimizer.lr_schedule(n_gen, lim=(mu_init, mu_rate*mu_init), mode=mode_dict[mode])  # the space is much tighter than the fc6 space, so step size
    # should be tuned for that
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=n_gen, backend="torch", optimizer=optimizer,
                                  GAN="BigBiGAN", verbose=False)
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    Exp.run()
    return np.percentile(Exp.scores_all[Exp.generations > Exp.generations.max()-5], 99.5)
#%
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
mixed_domain =[{'name': 'pop_size', 'type': 'continuous', 'domain': (15, 40), 'dimensionality': 1},
               {'name': 'select_rate', 'type': 'continuous', 'domain': (0.1, 0.9), 'dimensionality': 1},
               {'name': 'lr', 'type': 'continuous', 'domain': (0.2, 2), 'dimensionality': 1},
               {'name': 'mu_init', 'type': 'continuous', 'domain': (8, 20), 'dimensionality': 1},
               {'name': 'mu_rate', 'type': 'continuous', 'domain': (0.15, 0.9), 'dimensionality': 1},
               {'name': 'mode', 'type': 'categorical', 'domain': (0, 1), 'dimensionality': 1},
               {'name': 'sphere_norm', 'type': 'continuous', 'domain': (7, 12), 'dimensionality': 1}]
#%%
myBopt = BayesianOptimization(f=optim_result,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata=20,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='LCB',        # Expected Improvement # "LCB" seems to create more variablity than "EI"
                             exact_feval=False,         # True evaluations, no sample noise
                             maximize=True)
#%
from os.path import join
import pandas as pd
savedir = r"E:\OneDrive - Washington University in St. Louis\BigBiGAN"
max_iter = 400       ## maximum number of iterations
max_time = 15000      ## maximum allowed time
eps      = 1e-2     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False)
#
np.savez(join(savedir, "BBGAN_ZOHA_BO_tune.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best,
         domain=mixed_domain, mode_dict=mode_dict)
scores_short_tab = pd.DataFrame(np.append(myBopt.X, -myBopt.Y, 1), columns=["pop_size","select_ratio","lr","mu_init",
                                                                           "mu_rate","mode","norm","scores"])
scores_short_tab.to_csv(join(savedir, "BBGAN_ZOHA_BO_tune.csv"))
#%%
# from BigBiGAN import to_image, make_grid, G_u
# with torch.no_grad():
#     imgs = G_u(0.4 * torch.randn(20,120).cuda())
# imgs_np = to_image(make_grid(imgs, 7))
# imgs_np.show()
#%%
from GAN_utils import visualize_np
from geometry_utils import SLERP
