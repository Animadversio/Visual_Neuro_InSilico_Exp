from GAN_utils import BigBiGAN_wrapper, loadBigBiGAN
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper, loadPGGAN, PGGAN_wrapper, loadBigBiGAN, BigBiGAN_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from time import time
from os.path import join
import os
import sys
import lpips
from GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\App\Inversion"
os.makedirs(rootdir, exist_ok=True)
#%%
data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN\H_avg_BigBiGAN.npz")
evc, eva = data["evc_avg"], data["eva_avg"],
evc_tsr = torch.from_numpy(evc).cuda().float()
eva_tsr = torch.from_numpy(eva).cuda().float()
#%%
from torch.optim import Adam
BBGAN = loadBigBiGAN()
G = BigBiGAN_wrapper(BBGAN)
#%%
refvec = torch.randn(1, 120).cuda()
refimg = G.visualize(refvec)
#%%
def L1loss(im1, im2):
    return  (im1 - im2).abs().mean(axis=[1,2,3])

fitvec = torch.randn(1, 120).cuda()
fitvec.requires_grad_(True)
optimizer = Adam([fitvec], lr=1e-3, )
#%%
for step in range(250):
    fitimg = G.visualize(fitvec)
    dsim = ImDist(fitimg, refimg)
    L1 = L1loss(fitimg, refimg)
    loss = L1 + dsim
    loss.backward()
    optimizer.step()
    if step%10==0:
        print("step %d L1 %.3f dsim %.3f"%(step, L1.item(), dsim.item()))

imgcmp = torch.cat((refimg, fitimg))
ToPILImage()(make_grid(imgcmp).cpu()).show()
#%%

basis = evc_tsr
# basis = torch.eye(120).float().cuda()
#%%
fitvec = torch.randn(4, 120).cuda()
fitproj = fitvec @ basis
fitproj.requires_grad_(True)
optimizer = Adam([fitproj], lr=5e-3, )
for step in range(500):
    fitimg = G.visualize(fitproj @ basis.T)
    dsim = ImDist(fitimg, refimg)
    L1 = L1loss(fitimg, refimg)
    loss = L1.sum() + dsim.sum()
    loss.backward()
    optimizer.step()
    if step%10==0:
        print("step %d L1 %s dsim %s"%(step, ["%.2f"%d for d in L1.tolist()], ["%.2f"%d for d in dsim.squeeze(
             ).tolist()]))

imgcmp = torch.cat((refimg, fitimg))
ToPILImage()(make_grid(imgcmp).cpu()).show()
# Preliminary data
# HBasis
# step 240 L1 ['0.20', '0.10', '0.19', '0.13'] dsim ['0.28', '0.27', '0.29', '0.24']
# step 240 L1 ['0.14', '0.11', '0.19', '0.16'] dsim ['0.30', '0.27', '0.22', '0.23']
# step 490 L1 ['0.13', '0.16', '0.14', '0.14'] dsim ['0.27', '0.27', '0.26', '0.27']
# None basis
# step 240 L1 ['0.14', '0.13', '0.14', '0.10'] dsim ['0.27', '0.34', '0.30', '0.32']
# step 240 L1 ['0.18', '0.14', '0.14', '0.10'] dsim ['0.36', '0.26', '0.32', '0.22']
# step 490 L1 ['0.21', '0.10', '0.11', '0.14'] dsim ['0.29', '0.25', '0.26', '0.31']