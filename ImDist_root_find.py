import tqdm
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import os
import sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from os.path import join
from imageio import imread
from scipy.linalg import block_diag
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, GANForwardMetricHVPOperator, compute_hessian_eigenthings, get_full_hessian
#%%
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
for param in ImDist.parameters():
    param.requires_grad_(False)

def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean(axis=1)
#%%
BGAN = BigGAN.from_pretrained("biggan-deep-256")
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
EmbedMat = BGAN.embeddings.weight
#%%
noise_vec = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
class_vec = EmbedMat[:, 373:374].T
final_latent = torch.cat((noise_vec, class_vec), dim=1).cuda()
fit_img = BGAN.generator(final_latent, 0.7)
fit_img = (fit_img + 1.0) / 2.0
#%%
mov_latent = final_latent.detach().clone().requires_grad_(True)
mov_img = BGAN.generator(mov_latent, 0.7)
mov_img = (mov_img + 1.0) / 2.0
dsim = ImDist(fit_img, mov_img)
H = get_full_hessian(dsim, mov_latent)
eigval, eigvec = np.linalg.eigh(H)
del dsim
torch.cuda.empty_cache()
#%%
eigi = -1
tan_vec = torch.from_numpy(eigvec[:, eigi]).unsqueeze(0).float().cuda()
stepsize = (0.5 / eigval[eigi])**(1/2)
#%%
ticks = np.arange(1,5) * 2 * stepsize

step_latents = torch.tensor(ticks).unsqueeze(1).float().cuda() @ tan_vec + final_latent
with torch.no_grad():
    step_imgs = BGAN.generator(step_latents, 0.7)
    step_imgs = (step_imgs + 1.0) / 2.0
    dist_steps = ImDist(step_imgs, fit_img).squeeze()
#%%
ticks = np.arange(1,5) * 2 * stepsize
target_dist = torch.tensor([0.1, 0.2, 0.3, 0.4, ]).cuda()
tick_tsr = torch.tensor(ticks).unsqueeze(1).float().cuda().requires_grad_(True)
optimizer = Adam([tick_tsr], lr=0.1)
#%%
for i in range(50):
    optimizer.zero_grad()
    step_latents = tick_tsr @ tan_vec + final_latent
    step_imgs = BGAN.generator(step_latents, 0.7)
    step_imgs = (step_imgs + 1.0) / 2.0
    dist_steps = ImDist(step_imgs, fit_img).squeeze()
    loss = (target_dist - dist_steps).pow(2).mean()
    loss.backward()
    optimizer.step()
    if (i) % 10 == 0:
        print("step %d dsim %.3f" % (i, loss.item(), ))
#%%
from scipy.optimize import root_scalar

def dist_step(tick):
    step_latents = tick * tan_vec + final_latent
    with torch.no_grad():
        step_imgs = BGAN.generator(step_latents, 0.7)
        step_imgs = (step_imgs + 1.0) / 2.0
        dist_steps = ImDist(step_imgs, fit_img).squeeze()
    return dist_steps.item()
#%%
from time import time
t0 = time()
pos_ticks_target = []
neg_ticks_target = []
for dist_target in [0.1, 0.2, 0.3, 0.4, 0.5]:
    xsol = root_scalar(lambda tick: dist_step(tick)-dist_target, bracket=(0, 1), xtol=1E-4)
    pos_ticks_target.append(xsol.root)
    xsol2 = root_scalar(lambda tick: dist_step(tick) - dist_target, bracket=(-1, 0), xtol=1E-4)
    neg_ticks_target.append(xsol2.root)
print(time() - t0)