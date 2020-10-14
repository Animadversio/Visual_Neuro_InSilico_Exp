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
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
from lpips import LPIPS
#%%
ImDist = LPIPS(net="squeeze")
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None
#%%
BGAN = loadBigGAN()
SD = BGAN.state_dict()
#%%
shuffled_SD = {}
for name, Weight in SD.items():
    idx = torch.randperm(Weight.numel())
    W_shuf = Weight.view(-1)[idx].view(Weight.shape)
    shuffled_SD[name] = W_shuf
#%%
torch.save(shuffled_SD, join(datadir, "BigGAN_shuffle.pt"))
    # print(name, Weight.shape, Weight.mean().item(), Weight.std().item())
#%%
BGAN_sf = loadBigGAN()
BGAN_sf.load_state_dict(torch.load(join(datadir, "BigGAN_shuffle.pt")))
G_sf = BigGAN_wrapper(BGAN_sf)
#%%
img = BGAN_sf.generator(torch.randn(1, 256).cuda()*0.05, 0.7).cpu()
ToPILImage()((1+img[0])/2).show()
#%%
triali = 0
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\ctrl_Hessians"
for triali in tqdm(range(1, 100)):
    feat = torch.cat((torch.randn(128).cuda(), BGAN_sf.embeddings.weight[:, triali].clone()), dim=0)
    eigvals, eigvects, H = hessian_compute(G_sf, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    feat.requires_grad_(True)
    for blocki in [0, 3, 5, 8, 10, 12]:
        L2dist_col = []
        torch.cuda.empty_cache()
        H1 = BGAN_sf.generator.layers[blocki].register_forward_hook(Hess_hook)
        img = BGAN_sf.generator(feat, 0.7)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col[0], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
                 feat=feat.cpu().detach().numpy())
#%%
""" Compute a full path from genz to GAN image manifold """

savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\ctrl_Hessians"
for triali in tqdm(range(100, 101)):
    feat = torch.cat((torch.randn(128).cuda(), BGAN_sf.embeddings.weight[:, triali].clone()), dim=0)
    eigvals, eigvects, H = hessian_compute(G_sf, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    feat.requires_grad_(True)

    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = BGAN_sf.generator.gen_z.register_forward_hook(Hess_hook)
    img = BGAN_sf.generator(feat, 0.7)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col[0], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(savedir, "eig_gen_z_trial%d.npz"%triali), H=H00, eva=eva00, evc=evc00,
             feat=feat.cpu().detach().numpy())
    for blocki in list(range(13)):
        L2dist_col = []
        torch.cuda.empty_cache()
        H1 = BGAN_sf.generator.layers[blocki].register_forward_hook(Hess_hook)
        img = BGAN_sf.generator(feat, 0.7)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col[0], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
                 feat=feat.cpu().detach().numpy())
#%%
from hessian_analysis_tools import scan_hess_npz, plot_spectra, compute_hess_corr, plot_consistentcy_mat, plot_consistency_example, plot_consistency_hist, average_H
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\ctrl_Hessians"
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
modelnm = "BigGAN_shuffle"
# Load the Hessian NPZ
eva_ctrl, evc_ctrl, feat_ctrl, meta = scan_hess_npz(savedir, "eig_full_trial(\d*).npz", evakey='eva', evckey='evc', featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_ctrl, evc_ctrl)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_ctrl, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_ctrl, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_ctrl, evc_ctrl, figdir=figdir, use_cuda=False, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_ctrl, evc_ctrl, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)


#%%
"""Compute layer-wise Hessian for real BigGANs"""
from pytorch_pretrained_biggan import truncated_noise_sample
BGAN = loadBigGAN()
G = BigGAN_wrapper(BGAN)
triali = 0
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\real_Hessians"
for triali in tqdm(range(50)):
    noisevec = torch.from_numpy(truncated_noise_sample(1, 128, 0.6)).cuda()
    feat = torch.cat((noisevec, BGAN.embeddings.weight[:, triali:triali+1].clone().T), dim=1)
    eigvals, eigvects, H = hessian_compute(G, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    feat.requires_grad_(True)
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = BGAN.generator.gen_z.register_forward_hook(Hess_hook)
    img = BGAN.generator(feat, 0.7)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col[0], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(datadir, "eig_gen_z_trial%d.npz"%triali), H=H00, eva=eva00, evc=evc00,feat=feat.cpu().detach().numpy())
    for blocki in [0, 3, 5, 8, 10, 12]:
        L2dist_col = []
        torch.cuda.empty_cache()
        H1 = BGAN.generator.layers[blocki].register_forward_hook(Hess_hook)
        img = BGAN.generator(feat, 0.7)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col[0], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
                 feat=feat.cpu().detach().numpy())


#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\real_Hessians"
modelnm = "BigGAN_real"
# Load the Hessian NPZ
eva_real, evc_real, feat_real, meta = scan_hess_npz(savedir, "eig_full_trial(\d*).npz", evakey='eva', evckey='evc', featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_real, evc_real)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_ctrl)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_real, savename="%s_spectrum"%modelnm, figdir=figdir)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_real, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_real, evc_real, figdir=figdir, use_cuda=False, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_real, evc_real, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
#%% Comparison plot
#  Spectrum comparison
fig0 = plot_spectra(eva_real, savename="BigGAN_spectrum_shuffle_cmp", figdir=figdir, abs=True,
            titstr="BigGAN cmp", label="real", fig=None)
fig0 = plot_spectra(eva_ctrl, savename="BigGAN_spectrum_shuffle_cmp", figdir=figdir, abs=True,
            titstr="BigGAN cmp", label="shuffled", fig=fig0)
#%%  Correlation of Hessian plot
with np.load(join(figdir, "Hess_BigGAN_real_corr_mat.npz")) as data:
    corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%"real",
                                    savelabel="BigGAN_shuffle_cmp")
with np.load(join(figdir, "Hess_BigGAN_shuffle_corr_mat.npz")) as data:
    corr_mat_log, corr_mat_lin = data["corr_mat_log"], data["corr_mat_lin"]

fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%"shuffle",
                                    savelabel="BigGAN_shuffle_cmp", figs=(fig11,fig22))