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
from hessian_analysis_tools import scan_hess_npz, average_H, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example, compute_vector_hess_corr, compute_hess_corr
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
if sys.platform == "linux":
	saveroot = r"/scratch/binxu/GAN_hessian/StyleGAN2"
else:
	saveroot = r"E:\Cluster_Backup\StyleGAN2"

from argparse import ArgumentParser
parser = ArgumentParser(description='Computing Hessian at different part of the code space in StyleGAN2')
parser.add_argument('--modelname', type=str, default="model.ckpt-533504", help='checkpoint name')
parser.add_argument('--method', type=str, default="BP", help='Method of computing Hessian can be `BP` or '
                                                             '`ForwardIter` `BackwardIter` ')
parser.add_argument('--wspace', type=bool, default=False, help='resolution of generated image')
parser.add_argument('--fixed', type=bool, default=False, help='number of repititions')
parser.add_argument('--shuffled', type=bool, default=False, )#nargs="+"
args = parser.parse_args()#['--modelname', "ffhq-256-config-e-003810", "--fixed", "True"])

modelname = args.modelname  # "model.ckpt-533504"  # 109 sec
label = modelname + ("_W" if args.wspace else "") \
				  + ("_fix" if args.fixed else "") \
				  + ("_ctrl" if args.shuffled else "")

SGAN = loadStyleGAN2(modelname+".pt")
G = StyleGAN2_wrapper(SGAN)
if args.wspace: G.use_wspace(True)
if args.fixed: G.random = False

savedir = join(saveroot, label)
os.makedirs(savedir, exist_ok=True)
print(savedir)
for triali in range(2, 80):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())


figdir = join(savedir, "summary", label)
os.makedirs(figdir, exist_ok=True)
# modelnm = "StyleGAN_wspace_shuffle"
# Load the Hessian NPZ
eva_ctrl, evc_ctrl, feat_ctrl, meta = scan_hess_npz(savedir, "Hess_BP_(\d*).npz", evakey='eva_BP', evckey='evc_BP', featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_ctrl, evc_ctrl)
np.savez(join(figdir, "H_avg_%s.npz"%label), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum"%label, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%label), eigval_col=eva_ctrl, )
# compute and plot the correlation between hessian at different points
corr_mat_log_ctrl, corr_mat_lin_ctrl = compute_hess_corr(eva_ctrl, evc_ctrl, figdir=figdir, use_cuda=True, savelabel=label)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%label, savelabel=label)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%label, savelabel=label)
fig3 = plot_consistency_example(eva_ctrl, evc_ctrl, figdir=figdir, nsamp=5, titstr="%s"%label, savelabel=label)
fig3 = plot_consistency_example(eva_ctrl, evc_ctrl, figdir=figdir, nsamp=3, titstr="%s"%label, savelabel=label)

