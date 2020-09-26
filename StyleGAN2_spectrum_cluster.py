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
from GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, BigGAN_wrapper, loadStyleGAN2, StyleGAN2_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
from hessian_analysis_tools import scan_hess_npz, average_H, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
if sys.platform == "linux":
	saveroot = r"/scratch/binxu/GAN_hessian/StyleGAN2"

modelname = "model.ckpt-533504"  # 109 sec
label = modelname + ("_W" if wspace else "") 
				  + ("_fix" if fixed else "") 
				  + ("_ctrl" if shuffled else "")

SGAN = loadStyleGAN2(modelname+".pt")
G = StyleGAN2_wrapper(SGAN)
if wspace: G.use_wspace(True)
if fixed: G.

savedir = join(rootdir, label)
os.makedirs(savedir, exist_ok=True)
for triali in range(0, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())




