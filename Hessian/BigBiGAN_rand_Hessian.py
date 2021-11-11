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
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper, StyleGAN2_wrapper, loadPGGAN, PGGAN_wrapper, loadBigBiGAN, BigBiGAN_wrapper
from Hessian.hessian_analysis_tools import plot_spectra, compute_hess_corr
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigBiGAN"
os.makedirs(datadir, exist_ok=True)
#%%
# BBGAN_sf = loadBigBiGAN()
# BBGAN_sf.big_gan.init_weights()
# #%%
# randinitd_SD = {}
# for name, Weight in SD.items():
#     idx = torch.randperm(Weight.numel())
#     W_shuf = Weight.view(-1)[idx].view(Weight.shape)
#     randinitd_SD[name] = W_shuf
#%%
BBGAN_sf = loadBigBiGAN()
BBGAN_sf.big_gan.init_weights()
#%%
torch.save(BBGAN_sf.state_dict(), join(datadir, "BigBiGAN_randinit.pt"))
#%%
BBGAN_sf = loadBigBiGAN()
BBGAN_sf.load_state_dict(torch.load(join(datadir, "BigBiGAN_randinit.pt")))
G_sf = BigBiGAN_wrapper(BBGAN_sf)
#%%
img = G_sf.visualize(torch.randn(1,120).cuda()).cpu()
ToPILImage()(img[0,:].cpu()).show()
#%%
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None

savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigBiGAN\rand_Hessians" # join(datadir, "ctrl_Hessians")
os.makedirs(savedir, exist_ok=True)
for triali in tqdm(range(0, 1)):
    feat = torch.randn(1, 120).cuda()
    eigvals, eigvects, H = hessian_compute(G_sf, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    # feat.requires_grad_(True)
    # for blocki in [0, 3, 5, 8, 10, 12]:
    #     L2dist_col = []
    #     torch.cuda.empty_cache()
    #     H1 = SGAN_sf.convs[blocki].register_forward_hook(Hess_hook)
    #     img = SGAN_sf([feat], truncation=1)
    #     H1.remove()
    #     T0 = time()
    #     H00 = get_full_hessian(L2dist_col[0], feat)
    #     eva00, evc00 = np.linalg.eigh(H00)
    #     print("Spent %.2f sec computing" % (time() - T0))
    #     np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
    #              feat=feat.cpu().detach().numpy())
#%%
from Hessian.hessian_analysis_tools import scan_hess_npz, average_H, compute_vector_hess_corr, plot_consistentcy_mat, \
    plot_consistency_hist,  plot_consistency_example, plot_spectra
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigBiGAN\rand_Hessians"
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigBiGAN"
realfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"

modelnm = "BigBiGAN_randinit"
# Load the Hessian NPZ
eva_ctrl, evc_ctrl, feat_ctrl, meta = scan_hess_npz(savedir, "eig_full_trial(\d*).npz", evakey='eva', evckey='evc', featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_ctrl, evc_ctrl)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum"%modelnm, figdir=figdir)
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum_med"%modelnm, figdir=figdir, median=True)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_ctrl, )
# compute and plot the correlation between hessian at different points
"""Note the spectra of randinitd BigBiGAN doesn't make sense we use randomly initialized one instead"""
#%%
corr_mat_log_ctrl, corr_mat_lin_ctrl = compute_hess_corr(eva_ctrl+1E-7, evc_ctrl, figdir=figdir, use_cuda=True,
                                                         savelabel=modelnm+"shift1E-7")
corr_mat_vec_ctrl = compute_vector_hess_corr(eva_ctrl+1E-7, evc_ctrl, figdir=figdir, use_cuda=True,
                                                         savelabel=modelnm+"shift1E-7")
fig1, fig2 = plot_consistentcy_mat(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm+"shift1E-7",
                                   savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm+"shift1E-7",
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_ctrl+1E-7, evc_ctrl, figdir=figdir, nsamp=5, titstr="%s"%modelnm+"shift1E-7", savelabel=modelnm)
#%%
with np.load(join(figdir, "spectra_col_%s.npz"%modelnm)) as data:
    eva_ctrl = data["eigval_col"]
with np.load(join(realfigdir, "spectra_col.npz")) as data:
    eva_real = data["eigval_col"]
fig0 = plot_spectra(eva_real, savename="BigBiGAN_randinit_spectrum_cmp", figdir=figdir, abs=True,
            titstr="BigBiGAN cmp", label="real", fig=None)
fig0 = plot_spectra(eva_ctrl, savename="BigBiGAN_randinit_spectrum_cmp", figdir=figdir, abs=True,
            titstr="BigBiGAN cmp", label="randinitd", fig=fig0)
#%%
with np.load(join(realfigdir, "Hess__corr_mat.npz")) as data:
    corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%"real",
                                    savelabel="BigBiGAN_randinit_cmp")
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%"randinit",
                                    savelabel="BigBiGAN_randinit_cmp", figs=(fig11, fig22))
#%%
# eva_col, evc_col, meta = scan_hess_npz(r"E:\Cluster_Backup\BigBiGAN")
# np.savez(join(realfigdir, "spectra_col_%s.npz"%"BigBiGAN"), eigval_col=eva_col)
#%%
