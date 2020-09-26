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
from hessian_analysis_tools import scan_hess_npz, average_H, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
def shuffle_state_dict(SD, mask=lambda name:False):
    shuffled_SD = {}
    for name, Weight in SD.items():
        if mask(name):
            print("skip shuffling %s"%name)
            shuffled_SD[name] = Weight
            continue
        else:
            idx = torch.randperm(Weight.numel())
            W_shuf = Weight.view(-1)[idx].view(Weight.shape)
            shuffled_SD[name] = W_shuf
    return shuffled_SD
#%%
SGAN = loadStyleGAN2('ffhq-512-avg-tpurun1.pt')

#%%
shuffled_SD = shuffle_state_dict(SGAN.state_dict())
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
#%%
from hessian_analysis_tools import scan_hess_npz, average_H, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2\ctrl_Hessians"
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"

modelnm = "StyleGAN2_Face512_shuffle"
# Load the Hessian NPZ
eva_ctrl, evc_ctrl, feat_ctrl, meta = scan_hess_npz(savedir, "eig_full_trial(\d*).npz", evakey='eva', evckey='evc', featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_ctrl, evc_ctrl)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_ctrl, )
# compute and plot the correlation between hessian at different points
corr_mat_log_ctrl, corr_mat_lin_ctrl = compute_hess_corr(eva_ctrl, evc_ctrl, figdir=figdir, use_cuda=True, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eva_ctrl, evc_ctrl, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
#%% Comparison plot with real GANs: spectra
realfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
with np.load(join(figdir, "spectra_col_%s.npz"%"StyleGAN2_Face512_shuffle")) as data:
    eva_ctrl = data["eigval_col"]
with np.load(join(realfigdir, "spectra_col_%s.npz"%"ffhq-512-avg-tpurun1")) as data:
    eva_real = data["eigval_col"]
fig0 = plot_spectra(eva_real, savename="StyleGAN2_Face512_shuffle_spectrum_cmp", figdir=figdir, abs=True,
            titstr="StyleGAN2_Face512 cmp", label="real", fig=None)
fig0 = plot_spectra(eva_ctrl, savename="StyleGAN2_Face512_shuffle_spectrum_cmp", figdir=figdir, abs=True,
            titstr="StyleGAN2_Face512 cmp", label="shuffled", fig=fig0)
#%% Comparison plot with real GANs: correlation
with np.load(join(realfigdir, "Hess_ffhq-512-avg-tpurun1_corr_mat.npz")) as data:
    corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]
with np.load(join(figdir, "Hess_%s_corr_mat.npz"%"StyleGAN2_Face512_shuffle")) as data:
    corr_mat_log_ctrl, corr_mat_lin_ctrl = data["corr_mat_log"], data["corr_mat_lin"]
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%"real",
                                    savelabel="StyleGAN2_Face512_shuffle_cmp")
fig11, fig22 = plot_consistency_hist(corr_mat_log_ctrl, corr_mat_lin_ctrl, figdir=figdir, titstr="%s"%"shuffle",
                                    savelabel="StyleGAN2_Face512_shuffle_cmp", figs=(fig11, fig22))
#%%
eva_col, evc_col, meta = scan_hess_npz(r"E:\Cluster_Backup\StyleGAN2\ffhq-512-avg-tpurun1")
np.savez(join(realfigdir, "spectra_col_%s.npz"%"ffhq-512-avg-tpurun1"), eigval_col=eva_col)
#%%
SGAN = loadStyleGAN2('ffhq-512-avg-tpurun1.pt')
G = StyleGAN2_wrapper(SGAN)
#%%
G.random = False
feat = torch.randn(1, 512).cuda()
img1 = G.visualize(feat)
img2 = G.visualize(feat)
print((img1-img2).abs().max())
#%%
"""Precise control of shuffling and its effect on the image"""
from os.path import join
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, BigGAN_wrapper, loadStyleGAN2, StyleGAN2_wrapper, ckpt_root
# modelnm = "ffhq-512-avg-tpurun1"
#%%
maskfun = lambda name: False#"style." in name or "convs." in name
for modelnm in ["ffhq-512-avg-tpurun1", "ffhq-256-config-e-003810", "stylegan2-cat-config-f", "model.ckpt-533504"]:
    SGAN = loadStyleGAN2(modelnm+'.pt')
    shuf_SD = shuffle_state_dict(SGAN.state_dict(), maskfun)
    torch.save(shuf_SD, join(ckpt_root, modelnm+"_shuffle.pt"), )
    feat = torch.randn(5, 512).cuda()
    G = StyleGAN2_wrapper(SGAN)
    img = G.visualize(feat)
    G.StyleGAN.load_state_dict(shuf_SD)
    img_sf = G.visualize(feat)
    mtg = ToPILImage()(make_grid(torch.cat((img, img_sf)),nrow=5).cpu())
    mtg.show()
    mtg.save(join(ckpt_root, modelnm+"_shuffle.png"))
#%%
