from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

#%%
BGAN = BigGAN.from_pretrained("biggan-deep-256")
# BGAN.generator()
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
#%%
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
#%
def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean()
#%%
from imageio import imread
target = imread("block042_thread000_gen_gen041_001030.bmp")
target_tsr = torch.from_numpy(target / 255.0).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
# ToPILImage()(target_tsr[0, :].float()).show()
#%%
from torch.optim import SGD, Adam
noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
class_init = 0.06 * torch.randn(1, 128).cuda()
alpha = 5
class_vec = class_init.detach().clone().cuda().requires_grad_(True)
noise_vec = noise_init.detach().clone().cuda().requires_grad_(True)
optim1 = SGD([noise_vec], lr=0.01, weight_decay=0, )
optim2 = SGD([class_vec], lr=0.01, weight_decay=0.005, )
for step in range(200):
    optim1.zero_grad()
    optim2.zero_grad()
    fitimg = BGAN.generator(torch.cat((noise_vec, class_vec), dim=1), 0.7)
    dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)#
    dsim.backward()
    optim1.step()
    optim2.step()
    if (step + 1) % 10 ==0:
        print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))
#%
imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
imcmp.show()
#%%
# current best setting
#  [-1.5, -3, -3, -2.5, 0.8, 0.6]
#  [-1.5, -2.5, -3, -3, 0.8, 0.5]
def optim_BigGAN(param):
    lr1 = 10**param[0,0]
    lr2 = 10**param[0,1]
    wd1 = 10**param[0,2]
    wd2 = 10**param[0,3]
    mom1 = param[0,4]
    mom2 = param[0,5]
    noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
    class_init = 0.06 * torch.randn(1, 128).cuda()
    alpha = 5
    class_vec = class_init.detach().clone().cuda().requires_grad_(True)
    noise_vec = noise_init.detach().clone().cuda().requires_grad_(True)
    optim1 = SGD([noise_vec], lr=lr1, weight_decay=wd1, momentum=mom1)
    optim2 = SGD([class_vec], lr=lr2, weight_decay=wd2, momentum=mom2)
    for step in range(300):
        optim1.zero_grad()
        optim2.zero_grad()
        fitimg = BGAN.generator(torch.cat((noise_vec, class_vec), dim=1), 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        dsim.backward()
        optim1.step()
        optim2.step()
        classnorm = class_vec.norm()
        noisenorm = noise_vec.norm()
        if classnorm > 1.25:
            class_vec = (class_vec / classnorm * 0.7).detach().clone()
            optim2 = SGD([class_vec], lr=lr2, weight_decay=wd2, momentum=mom2)
            print("Class space renormalize")
        if noisenorm > 13:
            noise_vec = (noise_vec / noisenorm * 10).detach().clone()
            optim1 = SGD([noise_vec], lr=lr1, weight_decay=wd1, momentum=mom1)
            print("Noise space renormalize")
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), classnorm, noisenorm))
    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    imcmp.save(join(savedir, "%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
    if torch.isnan(dsim):
        return 1E6
    else:
        return dsim.item()
#%%
%%time
import numpy as np
import pandas as pd
from os.path import join
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
mixed_domain =[{'name': 'lr1', 'type': 'continuous', 'domain': (-4, -1.5),'dimensionality': 1},
               {'name': 'lr2', 'type': 'continuous', 'domain': (-4, -1.5),'dimensionality': 1},
               {'name': 'wd1', 'type': 'continuous', 'domain': (-4, -1.5),'dimensionality': 1},
               {'name': 'wd2', 'type': 'continuous', 'domain': (-4, -1.5),'dimensionality': 1},
               {'name': 'momentum1', 'type': 'continuous', 'domain': (0, 0.9),'dimensionality': 1},
               {'name': 'momentum2', 'type': 'continuous', 'domain': (0, 0.9),'dimensionality': 1},]
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert"
#%%
%%time
myBopt = BayesianOptimization(f=optim_BigGAN,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 0,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='LCB',        # Expected Improvement ‘EI’
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=False)
#%%
myBopt.X = np.array([[-1.5, -2.5, -3, -3, 0.8, 0.5], [-1.5, -2.5, -3.5, -3.5, 0.8, 0.5]])
myBopt.Y = np.array([[1.666],[1.790]])
#%% 40 mins for 48 iterations
max_iter = 900       ## maximum number of iterations
max_time = 30000      ## maximum allowed time
eps      = 1e-4     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
np.savez(join(savedir, "BigGAN_SGD_optim_BO_tune300.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best, domain=mixed_domain)
scores_short_tab = pd.DataFrame(np.append(myBopt.X, myBopt.Y, 1), columns=["lr1","lr2","wd1","wd2","mom1","mom2","scores"])
scores_short_tab.to_csv(join(savedir, "BigGAN_SGD_optim_BO_tune300.csv"))