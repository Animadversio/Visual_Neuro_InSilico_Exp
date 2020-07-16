from insilico_Exp import ExperimentEvolve
from ZO_HessAware_Optimizers import CholeskyCMAES
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid
import numpy as np
#%%
optimizer = CholeskyCMAES(120, population_size=None, init_sigma=0.25, init_code=0.8*np.random.randn(1, 120),
                          Aupdate_freq=10, maximize=True, random_seed=None, optim_params={})
experiment = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizer, GAN="BigBiGAN",
                       verbose=True)
experiment.run()
#%%
optimizerZH = ZOHA_Sphere_lr_euclid(120, population_size=40, select_size=15, lr=0.8,
                 maximize=True, rankweight=True, rankbasis=True, sphere_norm=9)
optimizerZH.lr_schedule(100, lim=(20, 7), mode="exp")
experiment = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizerZH, GAN="BigBiGAN",
                       verbose=True)
experiment.run()
experiment.visualize_trajectory(True)
experiment.visualize_exp(True)
experiment.visualize_best(True)
#%%
from BigBiGAN import BigBiGAN_render
imgs = BigBiGAN_render(0.7*np.random.randn(1,120),255.0)[0]