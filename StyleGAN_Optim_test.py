from GAN_utils import loadBigGAN, loadStyleGAN2, loadStyleGAN, StyleGAN_wrapper, BigGAN_wrapper, StyleGAN2_wrapper, \
    loadPGGAN,  PGGAN_wrapper,  loadBigBiGAN, BigBiGAN_wrapper
from Hessian.hessian_analysis_tools import plot_spectra, compute_hess_corr
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
from Hessian.GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
ImDist.cuda()
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\App\Inversion"
os.makedirs(rootdir, exist_ok=True)
#%%
data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN\H_avg_StyleGAN.npz")
evc, eva = data["evc_avg"], data["eva_avg"],
evc_tsr = torch.from_numpy(evc).cuda().float()
eva_tsr = torch.from_numpy(eva).cuda().float()
#%%
from torch.optim import Adam, SGD
SGAN = loadStyleGAN()
G = StyleGAN_wrapper(SGAN)
#%%
refvec = torch.randn(1, 512).cuda()
refimg = G.visualize(refvec)
#%%
def L1loss(im1, im2):
    return  (im1 - im2).abs().mean(axis=[1,2,3])

fitvec = torch.randn(1, 512).cuda()
fitvec.requires_grad_(True)
optimizer = Adam([fitvec], lr=1e-3, )
#%
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

# basis = evc_tsr
basis = torch.eye(120).float().cuda()
#%%
fitvec = torch.randn(4, 512).cuda()
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
# step 490 L1 ['0.09', '0.09', '0.10', '0.13'] dsim ['0.11', '0.11', '0.12', '0.15']
# None basis
# step 490 L1 ['0.11', '0.12', '0.11', '0.12'] dsim ['0.14', '0.17', '0.14', '0.17']
#%%
"""Try out W space """
step = 6
img = G.StyleGAN.generator([0.1*torch.randn(1, 512).cuda() for i in range(5)],
                           [0.5*torch.randn(5, 1, 4*2**i, 4*2**i, device="cuda") for i in range(step+1)], step=step)
ToPILImage()(make_grid(torch.clamp(img+1/2,0,1).cpu())).show()
#%%
"""Try out W space """
step = 6
img = G.StyleGAN.generator([0.1*torch.randn(1, 512).cuda(), 0.2*torch.randn(1, 512).cuda(),],
                           [0.5*torch.randn(2, 1, 4*2**i, 4*2**i, device="cuda") for i in range(step+1)], step=step)
ToPILImage()(make_grid(torch.clamp(img+1/2,0,1).cpu())).show()
#%%
tmpimg = G.visualize(0.15*torch.randn(5, 512).cuda(), wspace=True)
ToPILImage()(make_grid(tmpimg).cpu()).show()
#%%
# basis = evc_tsr
basis = torch.eye(512).float().cuda()
#%%
fitvec = 0.15*torch.randn(4, 512).cuda()
fitproj = fitvec @ basis
fitproj.requires_grad_(True)
optimizer = SGD([fitproj], lr=2.5e-3, )
for step in range(150):
    fitimg = G.visualize(fitproj @ basis.T, wspace=True)
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
# step 140 L1 ['0.16', '0.15', '0.16', '0.17'] dsim ['0.27', '0.23', '0.23', '0.24']
