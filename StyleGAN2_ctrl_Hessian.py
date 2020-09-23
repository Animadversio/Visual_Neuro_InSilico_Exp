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
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
#%%
SGAN = loadStyleGAN2('ffhq-512-avg-tpurun1.pt')
SD = SGAN.state_dict()
#%%
shuffled_SD = {}
for name, Weight in SD.items():
    idx = torch.randperm(Weight.numel())
    W_shuf = Weight.view(-1)[idx].view(Weight.shape)
    shuffled_SD[name] = W_shuf
#%%
torch.save(shuffled_SD, join(datadir, "StyleGAN2_ffhq-512-avg-tpurun1_shuffle.pt"))
    # print(name, Weight.shape, Weight.mean().item(), Weight.std().item())
#%%
SGAN_sf = loadStyleGAN2('ffhq-512-avg-tpurun1.pt')
SGAN_sf.load_state_dict(torch.load(join(datadir, "StyleGAN2_ffhq-512-avg-tpurun1_shuffle.pt")))
G_sf = StyleGAN2_wrapper(SGAN_sf)
#%%
img = G_sf.visualize(torch.randn(1,512).cuda()).cpu()
ToPILImage()(img[0,:].cpu()).show()
#%%
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None

triali = 0
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2\ctrl_Hessians"
os.makedirs(savedir, exist_ok=True)
for triali in tqdm(range(42, 50)):
    feat = torch.randn(1,512).cuda()
    eigvals, eigvects, H = hessian_compute(G_sf, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    feat.requires_grad_(True)
    for blocki in [0, 3, 5, 8, 10, 12]:
        L2dist_col = []
        torch.cuda.empty_cache()
        H1 = SGAN_sf.convs[blocki].register_forward_hook(Hess_hook)
        img = SGAN_sf([feat], truncation=1)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col[0], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
                 feat=feat.cpu().detach().numpy())
