"""Try To use Bayes Optim for the Black box optimization in GAN space."""

import GPyOpt
import GPy
from GPyOpt.methods import BayesianOptimization
import numpy as np
#%%
def f(x): return (6*x-2)**2*np.sin(12*x-4)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=30)
myBopt.plot_acquisition()
#%%
import torch
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
#%%
G.cuda()
def img_contrast(code):
    with torch.no_grad():
        img = G.visualize(torch.from_numpy(code).cuda().float())
    out = img.std(dim=(1,2,3)).cpu().numpy()
    return out[:, np.newaxis]
domain = [{'name': 'code', 'type': 'continuous', 'domain': (-3,3), 'dimensionality': 4096}]
# Maximize will add a negative sign to the optimization.
GANBopt = BayesianOptimization(f=img_contrast, domain=domain, batch_size=1, acquisition_optimizer_type='lbfgs', verbosity=True, maximize=True, acquisition_type="LCB")
GANBopt.run_optimization(max_iter=600, max_time=600, verbosity=True)
GANBopt.plot_convergence()
#%%
import cma
def img_contrast(code):
    with torch.no_grad():
        img = G.visualize(torch.from_numpy(code).cuda().float())
    out = img.std(dim=(1,2,3)).cpu().numpy()
    return -out[:, np.newaxis]

def img_contrast_batch(codes):

    with torch.no_grad():
        img = G.visualize(torch.from_numpy(np.array(codes)).cuda().float())
    out = img.std(dim=(1,2,3)).cpu().numpy()
    return (-out).tolist()

options = {'maxiter': 200, 'CMA_diagonal': True, 'CMA_cmean': 0.6, 'maxstd':5,  'minstd':0.5}
res = cma.evolution_strategy.fmin(img_contrast, np.random.randn(4096), 2, options, parallel_objective=img_contrast_batch)