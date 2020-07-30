from pytorch_pretrained_biggan import BigGAN, BigGANConfig, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import numpy as np
import pandas as pd
from imageio import imread
import matplotlib.pylab as plt
import sys
import os
from os.path import join
from time import time
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
#%%
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN

if sys.platform == "linux":
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    BGAN = get_BigGAN()
    Hpath = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
else:
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    Hpath = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"

BGAN.cuda().eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
#%% Set up loss 
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])

def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean()

alpha = 5  # relative weight
#%% Load hessian Matrix
data = np.load(Hpath)
evc_clas = torch.from_numpy(data['eigvects_clas_avg']).cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg']).cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()

#%%
from argparse import ArgumentParser
parser = ArgumentParser()
# def Hess_all_BigGAN_optim(param):
#     lr = 10 ** param[0, 0]
#     wd = 10 ** param[0, 1]
#     beta1 = 1 - 10 ** param[0, 2]  # param[2] = log10(1 - beta1)
#     beta2 = 1 - 10 ** param[0, 3]  # param[3] = log10(1 - beta2)
#     noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
#     class_init = 0.06 * torch.randn(1, 128).cuda()
#     latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
#     optim = Adam([latent_coef], lr=lr, weight_decay=wd, betas=(beta1, beta2))
#     # torch.optim.lr_scheduler
#     scores_all = []
#     for step in range(300):
#         optim.zero_grad()
#         latent_code = latent_coef @ evc_all.T
#         noise_vec = latent_code[:, :128]
#         class_vec = latent_code[:, 128:]
#         fitimg = BGAN.generator(latent_code, 0.7)
#         fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
#         dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
#         dsim.backward()
#         optim.step()
#         scores_all.append(dsim.item())
#         if (step + 1) % 10 == 0:
#             print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, dsim.item(), class_vec.norm(), noise_vec.norm()))
#
#     imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
#     # imcmp.show()
#     imcmp.save(join(savedir, "Hall%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
#
#     plt.figure()
#     plt.plot(scores_all)
#     plt.title("lr %.E wd %.E beta1 %.3f beta2 %.3f"%(lr,wd,beta1,beta2))
#     plt.savefig(join(savedir, "traj_Hall%06d_%.3f.jpg" % (np.random.randint(1000000), dsim.item())))
#     return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
def Hess_all_reg_BigGAN_optim(param):
    lr = 10 ** param[0, 0]
    beta1 = 1 - 10 ** param[0, 1]  # param[2] = log10(1 - beta1)
    beta2 = 1 - 10 ** param[0, 2]  # param[3] = log10(1 - beta2)
    reg_w1 = 10 ** param[0, 3]  # param[2] = log10(1 - beta1)
    reg_w2 = 10 ** param[0, 4]  # param[3] = log10(1 - beta2)
    sched_gamma = param[0, 5]
    noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
    class_init = 0.06 * torch.randn(1, 128).cuda()
    latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
    optim = Adam([latent_coef], lr=lr, weight_decay=0, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=sched_gamma)
    RNDid = np.random.randint(1000000)
    scores_all = []
    nos_norm = []
    cls_norm = []
    for step in range(maxstep):
        optim.zero_grad()
        latent_code = latent_coef @ evc_all.T
        noise_vec = latent_code[:, :128]
        class_vec = latent_code[:, 128:]
        fitimg = BGAN.generator(latent_code, 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        loss = dsim + reg_w1 * noise_vec.pow(2).sum() + reg_w2 * class_vec.pow(2).sum()
        loss.backward()
        optim.step()
        scheduler.step()
        scores_all.append(dsim.item())
        nos_norm.append(noise_vec.norm().item())
        cls_norm.append(class_vec.norm().item())
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, scores_all[-1], cls_norm[-1], nos_norm[-1]))
        if (step + 1) in ckpt_steps:
            imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
            imcmp.save(join(savedir, "Hallreg%06d_%.3f_s%d.jpg" % (RNDid, dsim.item(), step + 1)))

    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    imcmp.save(join(savedir, "Hallreg%06d_%.3f_final.jpg" % (RNDid, dsim.item())))
    # imcmp.show()
    fig, ax = plt.subplots()
    ax.plot(scores_all, label="Loss")
    ax.set_ylabel("Image Dissimilarity", color="blue", fontsize=14)
    plt.legend()
    ax2 = ax.twinx()
    ax2.plot(nos_norm, color="orange", label="noise")
    ax2.plot(cls_norm, color="magenta", label="class")
    ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
    plt.legend()
    plt.title("lr %.E beta1 %.3f beta2 %.3f wd_nos %.E wd_cls %.E gamma %.1f"%(lr,beta1,beta2,reg_w1,reg_w2,sched_gamma))
    plt.savefig(join(savedir, "traj_Hallreg%06d_%.3f.jpg" % (RNDid, dsim.item())))
    np.savez(join(savedir, "code_Hallreg%06d.jpg" % (RNDid,)), dsim=dsim.item(), scores_all=np.array(scores_all),
             nos_norm=np.array(nos_norm),cls_norm=np.array(cls_norm), code=latent_code.detach().cpu().numpy())
    return dsim.item() if not torch.isnan(dsim) else 1E6
#%%
imgroot  = r"/scratch/binxu/GAN_invert/img2fit" if sys.platform == "linux" else \
    r"E:\Cluster_Backup\BigGAN_invert\img2fit"
saveroot = r"/scratch/binxu/GAN_invert" if sys.platform == "linux" else r"E:\Cluster_Backup\BigGAN_invert"
imgnm = "block079_thread000_gen_gen078_003146.bmp"
foldernm = imgnm.split("/")[-1].split(".")[0]
savedir = join(saveroot, foldernm)
os.makedirs(savedir, exist_ok=True)
alpha = 5
maxstep = 600
ckpt_steps = [50, 100, 200, 300, 400, 500]
target = imread(join(imgroot, imgnm))
target_tsr = torch.from_numpy(target / 255.0).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
t0 = time()
dsim = Hess_all_reg_BigGAN_optim(np.array([[-1.3, -0.5, -2.34, -4, -3, 0.7]]))
dsim2 = Hess_all_reg_BigGAN_optim(np.array([[-1.3, -0.5, -2.34, -4, -3, 1.0]]))
print(time() - t0)  # 113 sec for 600 steps
#%%
hess_all_domain =[{'name': 'lr', 'type': 'continuous', 'domain': (-3, -1), 'dimensionality': 1},
               {'name': 'wd', 'type': 'continuous', 'domain': (-4, -1), 'dimensionality': 1},
               {'name': 'beta1', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'beta2', 'type': 'continuous', 'domain': (-3, -0.5), 'dimensionality': 1},]
hess_all_reg_domain =[{'name': 'lr', 'type': 'continuous', 'domain': (-3, -1), 'dimensionality': 1},
               {'name': 'beta1', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'beta2', 'type': 'continuous', 'domain': (-4, -0.5), 'dimensionality': 1},
               {'name': 'reg_w1', 'type': 'continuous', 'domain': (-5, -0.5), 'dimensionality': 1},
               {'name': 'reg_w2', 'type': 'continuous', 'domain': (-5, -0.5), 'dimensionality': 1},
               {'name': 'gamma', 'type': 'continuous', 'domain': (0.1, 1.5), 'dimensionality': 1},]

myBopt = BayesianOptimization(f=Hess_all_reg_BigGAN_optim,                     # Objective function
                             domain=hess_all_reg_domain,          # Box-constraints of the problem
                             initial_design_numdata=5,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='EI',        # Expected Improvement ‘EI’
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=False)
#%%
# myBopt.X = np.array([[-1.5, -2, -1, -3], [-1.5, -3, -1, -3], [-1.5, -2, -2, -3]])
# myBopt.Y = np.array([[1.11],[0.90],[1.29]])
#%% 40 mins for 48 iterations
max_iter = 1000       ## maximum number of iterations
max_time = 36000      ## maximum allowed time
eps      = 1e-4     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
np.savez(join(savedir, "BigGAN_Hess_Adam_optim_BO_tune600.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best,
         domain=hess_all_reg_domain)
scores_short_tab = pd.DataFrame(np.append(myBopt.X, myBopt.Y, 1), columns=[param['name'] for param in hess_all_reg_domain] + ['score'])
scores_short_tab.to_csv(join(savedir, "BigGAN_Hess_Adam_optim_BO_tune600.csv"))
#%% Test the double y axis plot
# fig, ax = plt.subplots()
# ax.plot(range(1, 10), label="Loss")
# ax.set_ylabel("Image Dissimilarity", fontsize=14)
# plt.legend()
# ax2 = ax.twinx()
# ax2.plot(range(10,20),color="orange",label="noise")
# ax2.plot(range(5,20),color="magenta",label="class")
# plt.legend()
# ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
# fig.savefig("tmp.png")
# plt.show()
