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
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper, loadStyleGAN, StyleGAN_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
#%%
from GAN_utils import StyleGAN_wrapper, loadStyleGAN
StyleGAN = loadStyleGAN()
SG = StyleGAN_wrapper(StyleGAN)
SG.use_wspace(True)
fixed = SG.fix_noise()
#%%
SG.random = False
vec = torch.randn(1,512).cuda()
img1 = SG.visualize(vec)
img2 = SG.visualize(vec)
L1max = (img1-img2).abs().max()
dsim = ImDist(img1,img2)
print("Maximal difference between trial L1max %.3f Dsim %.3f"%(L1max, dsim))
# random Maximal difference between trial L1max 0.610 Dsim 0.069
# fixednoise  Maximal difference between trial L1max 0.000 Dsim 0.000

#%% Effect of allowing noise in Z space spectra
SG.use_wspace(False)
SG.random = False
feat = torch.randn(1, 512)
eva_BP, evc_BP, H_BP = hessian_compute(SG, feat, ImDist, hessian_method="BP")
SG.random = True
eva_BP_rnd, evc_BP_rnd, H_BP_rnd = hessian_compute(SG, feat, ImDist, hessian_method="BP")
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN"
plt.figure(figsize=[8,4])
plt.subplot(121)
plt.plot(eva_BP)
plt.plot(eva_BP_rnd)
plt.ylabel("eig")
plt.subplot(122)
plt.plot(np.log10(np.abs(eva_BP)),label="fixednoise")
plt.plot(np.log10(np.abs(eva_BP_rnd)),label="random")
plt.legend()
plt.ylabel("log(eig)")
plt.suptitle("Randomness affects spectrum computed ")
plt.savefig(join(figdir, "StyleGAN_rndnoise_spectrum_effect.png"))
plt.show()
#%%
#%% Effect of allowing noise in w space spectra
SG.use_wspace(True)
SG.random = False
feat_w = SG.StyleGAN.style(feat.cuda())
eva_BP_w, evc_BP_w, H_BP_w = hessian_compute(SG, feat_w, ImDist, hessian_method="BP")
SG.random = True
eva_BP_w_rnd, evc_BP_w_rnd, H_BP_w_rnd = hessian_compute(SG, feat_w, ImDist, hessian_method="BP")
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN"
plt.figure(figsize=[8,4])
plt.subplot(121)
plt.plot(eva_BP_w)
plt.plot(eva_BP_w_rnd)
plt.ylabel("eig")
plt.subplot(122)
plt.plot(np.log10(np.abs(eva_BP_w)),label="fixednoise")
plt.plot(np.log10(np.abs(eva_BP_w_rnd)),label="random")
plt.legend()
plt.ylabel("log(eig)")
plt.suptitle("Randomness affects spectrum computed ")
plt.savefig(join(figdir, "StyleGAN_wspace_rndnoise_spectrum_effect.png"))
plt.show()
#%%
""" 
Note for StyleGAN: 
Allowing noise to be floating around instead of fixed will affect the lower part of spectra and can create negative 
eigenvalues. 
This affects Z space significantly strong as there are many small eigenvalues. Fix the noise will reveal structure in 
the lower part of the spectra, and W space mildly. 
However this doesn't change major conclusions of the top part of the spectra.
Similar rules should apply to StyleGAN2.  
"""

