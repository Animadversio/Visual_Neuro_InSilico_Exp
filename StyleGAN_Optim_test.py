from GAN_utils import BigBiGAN_wrapper, loadBigBiGAN
from GAN_utils import loadBigGAN, loadStyleGAN2, loadStyleGAN, StyleGAN_wrapper, BigGAN_wrapper, StyleGAN2_wrapper, \
    loadPGGAN, \
    PGGAN_wrapper, \
    loadBigBiGAN, BigBiGAN_wrapper
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
ImDist.cuda()
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\App\Inversion"
os.makedirs(rootdir, exist_ok=True)
#%%
data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN\H_avg_StyleGAN.npz")
evc, eva = data["evc_avg"], data["eva_avg"],
evc_tsr = torch.from_numpy(evc).cuda().float()
eva_tsr = torch.from_numpy(eva).cuda().float()
#%%
from torch.optim import Adam
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
step = 6
img = G.StyleGAN.generator(0.2*torch.randn(1, 1, 512).cuda(), [torch.randn(1, 1, 4*2**i, 4*2**i, device="cuda") for
                                                                i in range(step+1)], step=step)
ToPILImage()(torch.clamp(img[0,:].cpu()+1/2,0,1)).show()
#%%
class StyleGAN_wrapper():  # nn.Module
    def __init__(self, StyleGAN, resolution=256):
        sys.path.append(StyleGAN1_root)
        from generate import get_mean_style
        self.StyleGAN = StyleGAN
        self.mean_style = get_mean_style(StyleGAN, "cuda")
        self.step = int(math.log(resolution, 2)) - 2

    def visualize(self, code, scale=1.0, resolution=256, mean_style=None, wspace=False, noise=None):
        # if step is None: step = self.step
        step = int(math.log(resolution, 2)) - 2
        if not wspace:
            if mean_style is None: mean_style = self.mean_style
            imgs = self.StyleGAN(
                code, step=step, alpha=1,
                mean_style=mean_style, style_weight=0.7,
            )
        else: # code ~ 0.2 * torch.randn(1, 1, 512)
            if noise is None:
                noise = [torch.randn(code.shape[0], 1, 4 * 2 ** i, 4 * 2 ** i, device="cuda") for i in range(step + 1)]
            G.StyleGAN.generator(code.unsqueeze(1), noise, step=step)
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, resolution=256, mean_style=None, B=15, wspace=False, noise=None):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                       resolution=resolution, mean_style=mean_style, wspace=wspace, noise=noise).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
            # clear_output(wait=True)
            # progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, resolution=256, mean_style=None, B=15, wspace=False, noise=None):
        img_tsr = self.visualize_batch_np(codes_all_arr, resolution=resolution, mean_style=mean_style, B=B,
                                          wspace=wspace, noise=noise)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]