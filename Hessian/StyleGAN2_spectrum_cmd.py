"""
Commandline tool to compute Spectrum of StyleGAN2 
Esp. in a period that I cannot compile the StyleGAN2.
"""
import sys
import os
from os.path import join
from time import time
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
sys.path.append("E:\Github_Projects\Visual_Neuro_InSilico_Exp")
sys.path.append("D:\Github\Visual_Neuro_InSilico_Exp")
import lpips
try:
    ImDist = lpips.LPIPS(net="squeeze").cuda()
except:
    ImDist = lpips.PerceptualLoss(net="squeeze").cuda()
from Hessian.GAN_hessian_compute import hessian_compute
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper
rootdir = r"E:\Cluster_Backup\StyleGAN2"
#%% Configurations for different checkpoints
modelname = "stylegan2-cat-config-f"
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=2)  #
modelname = "ffhq-256-config-e-003810"  # 109 sec
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=1)  # 491 sec per BP
modelname = "ffhq-512-avg-tpurun1"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
modelname = "stylegan2-ffhq-config-f"
SGAN = loadStyleGAN2(modelname+".pt", size=1024, channel_multiplier=2)  # 491 sec per BP
modelname = "2020-01-11-skylion-stylegan2-animeportraits"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
modelname = "stylegan2-car-config-f"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
modelname = "model.ckpt-533504"  # 109 sec
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
#%%
# for triali in range(1,16):
#     feat = 0.5 * torch.randn(1, 512).detach().clone().cuda()
#     T0 = time()
#     eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#     print("%.2f sec" % (time() - T0))  # 2135.00 sec
#     H_col = []
#     for EPS in [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]:
#         T0 = time()
#         eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
#         print("%.2f sec" % (time() - T0))  # 325.83 sec
#         print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
#         H_col.append((eva_FI, evc_FI, H_FI))
#     T0 = time()
#     eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
#     print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % (np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]))
#     print("%.2f sec" % (time() - T0))  # 2132.44 sec
#
#     np.savez(join("E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2", "Hess_cmp_%d.npz"%triali), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
#                                         eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
#                                         eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#     print("Save finished")
#%%
"""Compute spectrum for different models through CMD interface. """

modelname = "ffhq-256-config-e-003810"  # 109 sec
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=1)  # 491 sec per BP
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(150):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())

modelname = "stylegan2-cat-config-f"
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(150):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())

modelname = "model.ckpt-533504"  # 109 sec
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(50, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())

modelname = "2020-01-11-skylion-stylegan2-animeportraits"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(50, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#%% "ffhq-512-avg-tpurun1"
modelname = "ffhq-512-avg-tpurun1"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(0, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#%% "stylegan2-ffhq-config-f"
modelname = "stylegan2-ffhq-config-f"
SGAN = loadStyleGAN2(modelname+".pt", size=1024, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(0, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#%%
#%% "stylegan2-ffhq-config-f"
modelname = "stylegan2-car-config-f"
modelsnm = "Car512"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
savedir = join(rootdir, modelname)
os.makedirs(savedir, exist_ok=True)
for triali in range(0, 100):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                   preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())

#%%
"""Hessian spectra of StyleGAN2 from wspace"""
from Hessian.hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, compute_vector_hess_corr, plot_spectra,\
    plot_consistentcy_mat,plot_consistency_example,plot_consistency_hist
#%% "ffhq-512-avg-tpurun1"
modelname = "ffhq-512-avg-tpurun1"
modelnm = "ffhq-512-avg-tpurun1"+"_wspace"
modelsnm = "Face512"
SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
G.use_wspace(True)
savedir = join(rootdir, modelname+"_wspace")
os.makedirs(savedir, exist_ok=True)
#%%
img = G.visualize(G.StyleGAN.get_latent(torch.randn(5, 512).cuda()))
ToPILImage()(make_grid(img).cpu()).show()
#%%
for triali in range(50, 100):
    feat_z = torch.randn(1, 512).cuda()
    feat = G.StyleGAN.get_latent(feat_z)
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                    preprocess=lambda img:img)
                   #preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%03d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
             feat=feat.detach().cpu().numpy(), feat_z=feat_z.detach().cpu().numpy())
#%%
#%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2_wspace"
os.makedirs(figdir, exist_ok=True)
# Load the Hessian NPZ
eva_col, evc_col, feat_col, meta = scan_hess_npz(savedir, "Hess_BP_(\d*).npz", featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col, )
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_col, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=3, titstr="%s"%modelnm, savelabel=modelnm)

#%% "stylegan2-cat-config-f"
modelname = "stylegan2-cat-config-f"
modelnm = "stylegan2-cat-config-f"+"_wspace"
modelsnm = "Cat256"
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=2)
G = StyleGAN2_wrapper(SGAN)
G.use_wspace(True)
savedir = join(rootdir, modelname+"_wspace")
os.makedirs(savedir, exist_ok=True)
#%%
img = G.visualize(G.StyleGAN.get_latent(torch.randn(5, 512).cuda()))
ToPILImage()(make_grid(img).cpu()).show()
#%%
for triali in range(1, 100):
    feat_z = torch.randn(1, 512).cuda()
    feat = G.StyleGAN.get_latent(feat_z)
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                    preprocess=lambda img:img)
                   # preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
             feat=feat.detach().cpu().numpy(), feat_z=feat_z.detach().cpu().numpy())
#%%
modelname = "stylegan2-cat-config-f"
modelnm = "stylegan2-cat-config-f"+"_wspace"
savedir = join(rootdir, modelname+"_wspace")
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2_wspace"
# Load the Hessian NPZ
eva_col, evc_col, feat_col, meta = scan_hess_npz(savedir, "Hess_BP_(\d*).npz", featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col, )
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_col, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=3, titstr="%s"%modelnm, savelabel=modelnm)

#%%
#% "ffhq-256-config-e-003810" Face 256
modelname = "ffhq-256-config-e-003810"
modelnm = modelname+"_wspace"
modelsnm = "Face256"
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=1)  # 491 sec per BP
G = StyleGAN2_wrapper(SGAN)
G.use_wspace(True)
img = G.visualize(G.StyleGAN.get_latent(torch.randn(5, 512).cuda()))
ToPILImage()(make_grid(img).cpu()).show()

savedir = join(rootdir, modelname+"_wspace")
os.makedirs(savedir, exist_ok=True)
for triali in range(0, 100):
    feat_z = torch.randn(1, 512).cuda()
    feat = G.StyleGAN.get_latent(feat_z)
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP",
                    preprocess=lambda img:img)
                   # preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 109 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
             feat=feat.detach().cpu().numpy(), feat_z=feat_z.detach().cpu().numpy())
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2_wspace"
# Load the Hessian NPZ
eva_col, evc_col, feat_col, meta = scan_hess_npz(savedir, "Hess_BP_(\d*).npz", featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col, )
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_col, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=3, titstr="%s"%modelnm, savelabel=modelnm)
#%%
