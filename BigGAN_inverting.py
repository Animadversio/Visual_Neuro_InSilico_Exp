from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import numpy as np
import pandas as pd
from os.path import join
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
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
#%% Load up Hessian data for BigGAN
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\Hessian"
data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz")
#%
evc_clas = torch.from_numpy(data['eigvects_clas_avg']).cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg']).cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()
#%%
from imageio import imread
target = imread("block042_thread000_gen_gen041_001030.bmp")
target_tsr = torch.from_numpy(target / 255.0).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
# ToPILImage()(target_tsr[0, :].float()).show()
#%% Try direct optimizaion using Adam or SGD
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
    fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
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
    return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
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


# %%
#%% Using Hessian to Pre-condition the latent space

#%%
noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
class_init = 0.06 * torch.randn(1, 128).cuda()
#%%
alpha = 5
noise_coef = (noise_init @ evc_nois).detach().clone().requires_grad_(True)
class_coef = (class_init @ evc_clas).detach().clone().requires_grad_(True)
# optim1 = SGD([noise_coef], lr=0.01, weight_decay=0.005, momentum=0.8)
# optim2 = SGD([class_coef], lr=0.01, weight_decay=0.05, momentum=0.8)
# This is really good and consistent.
optim1 = Adam([noise_coef], lr=0.05, weight_decay=0.005, betas=(0.8, 0.999))
optim2 = Adam([class_coef], lr=0.05, weight_decay=0.05, betas=(0.8, 0.999))
for step in range(300):
    optim1.zero_grad()
    optim2.zero_grad()
    class_vec = class_coef @ evc_clas.T
    noise_vec = noise_coef @ evc_nois.T
    fitimg = BGAN.generator(torch.cat((noise_vec, class_vec), dim=1), 0.7)
    fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
    dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)#
    dsim.backward()
    optim1.step()
    optim2.step()
    if (step + 1) % 10 ==0:
        print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))
#%
imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
imcmp.show()
imcmp.save(join(savedir, "%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
#%%
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR
# This is really good and consistent.
latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
optim = Adam([latent_coef], lr=0.03, weight_decay=0.01, betas=(0.9, 0.999))
scheduler = StepLR(optim, 50, gamma=1.0)
# scheduler = CosineAnnealingLR(optim, T_max=300, eta_min=1E-2)
# optim = SGD([latent_coef], lr=0.05, weight_decay=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.08, div_factor=2, final_div_factor=4, total_steps=300, steps_per_epoch=1, epochs=300)
# scheduler = CyclicLR(optim, base_lr=1E-2, max_lr=10E-2, step_size_up=25, step_size_down=75)
for step in range(300):
    optim.zero_grad()
    latent_code = latent_coef @ evc_all.T
    noise_vec = latent_code[:,:128]
    class_vec = latent_code[:,128:]
    fitimg = BGAN.generator(latent_code, 0.7)
    fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
    dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)#
    dsim.backward()
    optim.step()
    # scheduler.step()
    if (step + 1) % 10 ==0:
        print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))

imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
imcmp.show()
imcmp.save(join(savedir, "Hall_lr%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))

#%% Define the Experiment function
def Hess_all_BigGAN_optim(param):
    lr = 10 ** param[0, 0]
    wd = 10 ** param[0, 1]
    beta1 = 1 - 10 ** param[0, 2]  # param[2] = log10(1 - beta1)
    beta2 = 1 - 10 ** param[0, 3]  # param[3] = log10(1 - beta2)
    noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
    class_init = 0.06 * torch.randn(1, 128).cuda()
    latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
    optim = Adam([latent_coef], lr=lr, weight_decay=wd, betas=(beta1, beta2))
    # torch.optim.lr_scheduler
    scores_all = []
    for step in range(300):
        optim.zero_grad()
        latent_code = latent_coef @ evc_all.T
        noise_vec = latent_code[:, :128]
        class_vec = latent_code[:, 128:]
        fitimg = BGAN.generator(latent_code, 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        dsim.backward()
        optim.step()
        scores_all.append(dsim.item())
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))

    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    # imcmp.show()
    imcmp.save(join(savedir, "Hall%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))

    plt.figure()
    plt.plot(scores_all)
    plt.title("lr %.E wd %.E beta1 %.3f beta2 %.3f"%(lr,wd,beta1,beta2))
    plt.savefig(join(savedir, "traj_Hall%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
    return dsim.item() if not torch.isnan(dsim) else 1E6
#%%

mixed_domain =[{'name': 'lr', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'wd', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'beta1', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'beta2', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},]
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\Hessian"

myBopt = BayesianOptimization(f=Hess_all_BigGAN_optim,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 0,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='LCB',        # Expected Improvement ‘EI’
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=False)
#%%
myBopt.X = np.array([[-1.5, -2, -1, -3], [-1.5, -3, -1, -3], [-1.5, -2, -2, -3]])
myBopt.Y = np.array([[1.11],[0.90],[1.29]])
#%% 40 mins for 48 iterations
max_iter = 100       ## maximum number of iterations
max_time = 3600      ## maximum allowed time
eps      = 1e-2     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
np.savez(join(savedir, "BigGAN_Hess_Adam_optim_BO_tune300.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best, domain=mixed_domain)
scores_short_tab = pd.DataFrame(np.append(myBopt.X, myBopt.Y, 1), columns=["lr","wd","beta1","beta2","scores"])
scores_short_tab.to_csv(join(savedir, "BigGAN_Hess_Adam_optim_BO_tune300.csv"))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
alpha = 5
def Hess_sep_BigGAN_optim(param):
    lr1 = 10 ** param[0, 0]
    wd1 = 10 ** param[0, 1]
    lr2 = 10 ** param[0, 2]
    wd2 = 10 ** param[0, 3]
    noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
    class_init = 0.06 * torch.randn(1, 128).cuda()
    noise_coef = (noise_init @ evc_nois).detach().clone().requires_grad_(True)
    class_coef = (class_init @ evc_clas).detach().clone().requires_grad_(True)
    optim1 = Adam([noise_coef], lr=lr1, weight_decay=wd1, betas=(0.9, 0.999))
    optim2 = Adam([class_coef], lr=lr2, weight_decay=wd2, betas=(0.9, 0.999))
    # torch.optim.lr_scheduler
    for step in range(300):
        optim1.zero_grad()
        optim2.zero_grad()
        class_vec = class_coef @ evc_clas.T
        noise_vec = noise_coef @ evc_nois.T
        fitimg = BGAN.generator(torch.cat((noise_vec, class_vec), dim=1), 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        dsim.backward()
        optim1.step()
        optim2.step()
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))

    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    imcmp.show()
    imcmp.save(join(savedir, "Hsep%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
    return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
mixed_domain =[{'name': 'lr1', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'wd1', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'lr2', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'wd2', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},]
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\Hessian"
myBopt2 = BayesianOptimization(f=Hess_sep_BigGAN_optim,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 0,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='CMA',
                             acquisition_type='LCB',        # Expected Improvement ‘EI’
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=False)
#%
myBopt2.X = np.array([[-1.5,-3,-1.6,-3], [-1.5,-3,-1.6,-3], [-1.8,-3,-1.8,-3]])
myBopt2.Y = np.array([[1.12],[0.98],[1.13]])
#% 40 mins for 48 iterations
max_iter = 200       ## maximum number of iterations
max_time = 10800      ## maximum allowed time
eps      = 1e-4      ## tolerance, max distance between consicutive evaluations.
myBopt2.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
np.savez(join(savedir, "BigGAN_Hess_sep_Adam_optim_BO_tune300.npz"), X=myBopt2.X, Y=myBopt2.Y, Y_best=myBopt2.Y_best, domain=mixed_domain)
scores_short_tab = pd.DataFrame(np.append(myBopt2.X, myBopt2.Y, 1), columns=["lr1","wd1","lr2","wd2","scores"])
scores_short_tab.to_csv(join(savedir, "BigGAN_Hess_sep_Adam_optim_BO_tune300.csv"))

#%% Use an natural image to fit!
import matplotlib.pylab as plt
from skimage.transform import resize
from skimage.io import imread
dogimg = imread(r"E:\Github_Projects\GAN-Transform-and-Project\examples\very-cute-doggo.jpg")
dogrsz = resize(dogimg,(256,256),anti_aliasing=True,)
plt.imshow(dogrsz)
plt.show()
if dogrsz.max() > 1.1:
    dogrsz = dogrsz / 255.0
target_tsr = torch.from_numpy(dogrsz).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
#%%
Hess_sep_BigGAN_optim(np.array([[-1.37,-3,-1.5,-3]]))
#%%
Hess_all_BigGAN_optim(np.array([[-1.5,-3,-0.8,-2.5]]))
#%%
alpha = 5
def Hess_all_reg_BigGAN_optim(param):
    lr = 10 ** param[0, 0]
    beta1 = 1 - 10 ** param[0, 1]  # param[2] = log10(1 - beta1)
    beta2 = 1 - 10 ** param[0, 2]  # param[3] = log10(1 - beta2)
    reg_w1 = 10 ** param[0, 3]  # param[2] = log10(1 - beta1)
    reg_w2 = 10 ** param[0, 4]  # param[3] = log10(1 - beta2)
    noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
    class_init = 0.06 * torch.randn(1, 128).cuda()
    latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
    optim = Adam([latent_coef], lr=lr, weight_decay=0, betas=(beta1, beta2))
    # torch.optim.lr_scheduler
    scores_all = []
    for step in range(300):
        optim.zero_grad()
        latent_code = latent_coef @ evc_all.T
        noise_vec = latent_code[:, :128]
        class_vec = latent_code[:, 128:]
        fitimg = BGAN.generator(latent_code, 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        loss = dsim + reg_w1 * noise_vec.pow(2).sum() + reg_w2 * class_vec.pow(2).sum()
        loss.backward()
        # L2reg.backward()
        optim.step()
        scores_all.append(dsim.item())
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))

    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    # imcmp.show()
    imcmp.save(join(savedir, "Hallreg%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))

    plt.figure()
    plt.plot(scores_all)
    plt.title("lr %.E beta1 %.3f beta2 %.3f wd_nos %.E wd_cls %.E "%(lr,beta1,beta2,reg_w1,reg_w2))
    plt.savefig(join(savedir, "traj_Hallreg%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
    return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
mixed_domain =[{'name': 'lr', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'beta1', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'beta2', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'reg_w1', 'type': 'continuous', 'domain': (-5, -0.5), 'dimensionality': 1},
               {'name': 'reg_w2', 'type': 'continuous', 'domain': (-5, -0.5), 'dimensionality': 1},]
#%%
doggoBopt = BayesianOptimization(f=Hess_all_reg_BigGAN_optim,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 0,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='EI',        # Expected Improvement ‘EI’
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=False)
#%
doggoBopt.X = np.array([[-1.5,-1,-3,-2,-2], [-1.5,-1,-3,-4,-2], [-1.5,-1,-3,-4,-1]])
doggoBopt.Y = np.array([[1.70],[1.86],[2.055]])
#% 40 mins for 48 iterations
max_iter = 200       ## maximum number of iterations
max_time = 7200      ## maximum allowed time
eps      = 1e-4     ## tolerance, max distance between consicutive evaluations.
doggoBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
np.savez(join(savedir, "BigGAN_Hess_Adam_L2reg_optim_BO_tune300_dog.npz"), X=doggoBopt.X, Y=doggoBopt.Y, Y_best=doggoBopt.Y_best, domain=mixed_domain)
scores_short_tab = pd.DataFrame(np.append(doggoBopt.X, doggoBopt.Y, 1), columns=["lr","beta1","beta2","reg_w1","reg_w2","scores"])
scores_short_tab.to_csv(join(savedir, "BigGAN_Hess_Adam_L2reg_optim_BO_tune300_dog.csv"))
#%% Try LBFGS on the problem not successful....
#   It's super easy to explode in the middle
#   Obsolete...... July. 30th

from torch.optim import LBFGS
alpha = 5
reg_w1 = 10**-2.5
reg_w2 = 10**-2.5
noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
class_init = 0.06 * torch.randn(1, 128).cuda()
latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
optim = LBFGS([latent_coef], lr=0.15, max_iter=30, history_size=100)

# torch.optim.lr_scheduler
scores_all = []
for step in range(50):
    def closure():
        optim.zero_grad()
        latent_code = latent_coef @ evc_all.T
        noise_vec = latent_code[:, :128]
        class_vec = latent_code[:, 128:]
        fitimg = BGAN.generator(latent_code, 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        loss = dsim + reg_w1 * noise_vec.pow(2).sum() + reg_w2 * class_vec.pow(2).sum()
        loss.backward()
        scores_all.append(dsim.item())
        return loss
    # L2reg.backward()
    optim.step(closure)
    # if (step + 1) % 10 == 0:
    print("step%d loss %.2f norm: code: %.2f" % (step, scores_all[-1], latent_coef.norm())) #class_vec.norm(), noise_vec.norm()))
latent_code = latent_coef @ evc_all.T
noise_vec = latent_code[:, :128]
class_vec = latent_code[:, 128:]
fitimg = BGAN.generator(latent_code, 0.7)
fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)
imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
# imcmp.show()
imcmp.save(join(savedir, "Halllbfgs%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))

plt.figure()
plt.plot(scores_all)
# plt.title("lr %.E beta1 %.3f beta2 %.3f wd_nos %.E wd_cls %.E "%(lr,beta1,beta2,reg_w1,reg_w2))
plt.savefig(join(savedir, "traj_Halllbfgs%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
# return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
