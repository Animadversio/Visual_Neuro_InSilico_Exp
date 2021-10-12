#%%
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
import os
from os.path import join
import sys
import lpips
from Hessian.GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from Hessian.hessian_analysis_tools import average_H, scan_hess_npz

use_gpu = True if torch.cuda.is_available() else False
ImDist = lpips.LPIPS(net='squeeze').cuda()
#%%
"""Torch Hub version, really heavy and cumbersum"""
model = torch.hub.load('ndahlquist/pytorch-hub-stylegan:0.0.1', 'style_gan', pretrained=True)
class StyleGAN_wrapper():  # nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN

    def visualize(self, code, scale=1):
        imgs = self.StyleGAN.forward(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale
G = StyleGAN_wrapper(model.cuda())
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN_hr"
#%%
data = np.load(join(savedir, "Hessian_EPS_BP.npz"))
H_BP = data["H_BP"]
feat = torch.tensor(data['feat']).detach().cuda()
#%%
# noise = torch.randn(1, 512)
# feat = noise.detach().clone().cuda()
# G.StyleGAN.cuda()
H_col = []
for EPS in [1E-6, 1E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, ]:
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS,
           preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
    print("%.2f sec" % (time() - T0))  # 260.5 sec
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
    EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
    H_col.append((eva_FI, evc_FI, H_FI))

np.savez(join(savedir, "Hessian_EPS_accuracy.npz"), H_col=H_col, feat=feat.detach().cpu().numpy())
print("Save Completed. ")
#%%
np.savez(join(savedir, "Hessian_EPS_BP.npz"), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#%%
G.StyleGAN.to("cpu")
feat.cpu()
T0 = time()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True), device="cpu")
print("%.2f sec" % (time() - T0))  # this will exceed gpu memory
np.savez(join(savedir, "Hessian_EPS_BI.npz"), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI, feat=feat.detach().cpu().numpy())
# print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1])
# print("Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" %
#       np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
# print("Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter %.3f"%
#       np.corrcoef(H_FI.flatten(), H_BI.flatten())[0, 1])
#%% Load the Hessian data and compute the correlation value
data_BP = np.load(join(savedir, "Hessian_EPS_BP.npz"))
data_FI = np.load(join(savedir, "Hessian_EPS_accuracy.npz"), allow_pickle=True)

#   correlation with or without taking absolute of the eigenvalues
H_BP, evc_BP, eva_BP = data_BP["H_BP"], data_BP["evc_BP"], data_BP["eva_BP"]
EPS_list = [1E-6, 1E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 3E-2, 1E-1, ]
for EPSi in range(data_FI['H_col'].shape[0]):
    EPS = EPS_list[EPSi]
    eva_FI, evc_FI, H_FI = data_FI['H_col'][EPSi, :]
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
        EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
    H_PSD = evc_FI@np.diag(np.abs(eva_FI)) @evc_FI.T
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter (AbsHess) %.3f" % (
        EPS, np.corrcoef(H_BP.flatten(), H_PSD.flatten())[0, 1]))
    # print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
    #     EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
#%%%
from Hessian.hessian_analysis_tools import compute_hess_corr, plot_consistentcy_mat, plot_consistency_example
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
#%%
def plot_spectra(eigval_col, savename="spectrum_onetrial.jpg", figdir=savedir, fig=None, label="BP"):
    """A local function to compute these figures for different subspaces. """
    eigmean = eigval_col.mean(axis=0)
    eiglim = np.percentile(eigval_col, [5, 95], axis=0)
    sortidx = np.argsort(-np.abs(eigmean))
    eigmean = np.abs(eigmean[sortidx])
    eiglim = eiglim[:, sortidx]
    eigN = len(eigmean)
    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    else:
        # plt.figure(fig.number)
        plt.figure(num=fig.number)
        axs = fig.axes
    plt.sca(axs[0])
    plt.plot(range(eigN), eigmean, alpha=0.6)
    plt.fill_between(range(eigN), eiglim[0, :], eiglim[1, :], alpha=0.3, label=label)
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.legend()
    plt.sca(axs[1])
    plt.plot(range(eigN), np.log10(eigmean), alpha=0.6)
    plt.fill_between(range(eigN), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label=label)
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of StyleGAN\n (error bar for [5,95] percentile among all samples)")
    plt.savefig(join(figdir, savename), bbox_extra_artists=[st]) # this is working.
    # fig.show()
    return fig


fig = plot_spectra(data_BP["eva_BP"][np.newaxis, :], label="BP", savename="spectrum_onetrial.jpg")
fig = plot_spectra(data_FI["H_col"][4, 0][np.newaxis, :], savename="spectrum_method_cmp.jpg", label="ForwardIter 1E-3", fig=fig)
fig = plot_spectra(data_FI["H_col"][5, 0][np.newaxis, :], savename="spectrum_method_cmp.jpg", label="ForwardIter 3E-3", fig=fig)
fig = plot_spectra(data_FI["H_col"][6, 0][np.newaxis, :], savename="spectrum_method_cmp.jpg", label="ForwardIter 1E-2", fig=fig)
plt.show()
#%%
"""
This is the smaller explicit version of StyleGAN. Very easy to work with
"""
#%%
sys.path.append("E:\Github_Projects\style-based-gan-pytorch")
sys.path.append("D:\Github\style-based-gan-pytorch")
from model import StyledGenerator
from generate import get_mean_style
import math
#%%
generator = StyledGenerator(512).to("cuda")
# generator.load_state_dict(torch.load(r"E:\Github_Projects\style-based-gan-pytorch\checkpoint\stylegan-256px-new.model")['g_running'])
generator.load_state_dict(torch.load(r"D:\Github\style-based-gan-pytorch\checkpoint\stylegan-256px-new.model")[
                              'g_running'])
generator.eval()
for param in generator.parameters():
    param.requires_grad_(False)
mean_style = get_mean_style(generator, "cuda")
step = int(math.log(256, 2)) - 2
#%%
feat = torch.randn(1, 512, requires_grad=False).to("cuda")
image = generator(
        feat,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
#%%
class StyleGAN_wrapper():  # nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN

    def visualize(self, code, scale=1, step=step, mean_style=mean_style):
        imgs = self.StyleGAN(
            code,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale
G = StyleGAN_wrapper(generator)
#%%
from Hessian.GAN_hessian_compute import hessian_compute

#%%
for triali in range(1, 15):
    feat = torch.randn(1, 512,).to("cuda")
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 120 sec
    feat = feat.detach().clone()
    T0 = time()
    eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("%.2f sec" % (time() - T0))  # 120 sec
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=1E-3)
    print("%.2f sec" % (time() - T0))  # 64 sec
    print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1])
    print("Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" %
          np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
    print("Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter %.3f"%
          np.corrcoef(H_FI.flatten(), H_BI.flatten())[0, 1])
    H_col = []
    for EPS in [1E-6, 1E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]:
        T0 = time()
        eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
        H_PSD = evc_FI @ np.diag(np.abs(eva_FI)) @ evc_FI.T
        print("%.2f sec" % (time() - T0))  # 325.83 sec
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter (AbsHess) %.3f" % (
            EPS, np.corrcoef(H_BP.flatten(), H_PSD.flatten())[0, 1]))
        H_col.append((eva_FI, evc_FI, H_FI))

    np.savez(join(savedir, "Hess_accuracy_cmp_%d.npz" % triali), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                            eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
                                            eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
    print("Save finished")
#%%
datadir = r"E:\Cluster_Backup\StyleGAN"
for triali in tqdm(range(300)):
    feat = torch.randn(1, 512,).to("cuda")
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 120 sec
    np.savez(join(datadir, "Hessian_rand_%d.npz" % triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
        feat=feat.detach().cpu().numpy())
#%%
# eva_col = []
# evc_col = []
# for triali in tqdm(range(300)):
#     data = np.load(join(datadir, "Hessian_rand_%d.npz" % triali))
#     eva_col.append(data["eva_BP"])
#     evc_col.append(data["evc_BP"])
# #%%
# eva_col = np.array(eva_col)
datadir = r"E:\Cluster_Backup\StyleGAN"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN"

os.makedirs(figdir, exist_ok=True)
eva_col, evc_col, feat_col, meta = scan_hess_npz(datadir, "Hessian_rand_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col)
np.savez(join(figdir, "H_avg_%s.npz"%"StyleGAN"), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
#%%
fig = plot_spectra(eva_col, figdir=figdir, titstr="StyleGAN", )
#%%
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=True)
# without cuda 12:11 mins, with cuda 8:21
# corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False)
#%
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, posN=300, figdir=figdir, titstr="StyleGAN")
#%
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="StyleGAN",)
fig3.show()
#%%
#%% Accuracy plot
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN"
datadir = r"E:\Cluster_Data\StyleGAN"
EPS_list = [1E-6, 1E-5, 1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]
raw_corr_tab = []
PSD_corr_tab = []
for triali in range(15):
    print("Computation trial %d"%triali)
    data = np.load(join(savedir, "Hess_accuracy_cmp_%d.npz" % triali), allow_pickle=True)
    H_col = data["H_col"]
    eva_BP, evc_BP, H_BP = data["eva_BP"], data["evc_BP"], data["H_BP"]
    corr_vals = []
    PSD_corr_vals = []
    for EPSi, EPS in enumerate(EPS_list):
        eva_FI, evc_FI, H_FI = H_col[EPSi, :]
        H_PSD = evc_FI @ np.diag(np.abs(eva_FI)) @ evc_FI.T
        corr_vals.append(np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
        PSD_corr_vals.append(np.corrcoef(H_BP.flatten(), H_PSD.flatten())[0, 1])
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
            EPS, corr_vals[-1]))
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter (AbsHess) %.3f" % (
            EPS, PSD_corr_vals[-1]))
    raw_corr_tab.append(corr_vals)
    PSD_corr_tab.append(PSD_corr_vals)
raw_corr_tab = np.array(raw_corr_tab)
PSD_corr_tab = np.array(PSD_corr_tab)
np.savez(join(figdir, "accuracy_stats.npz"), raw_corr_tab=raw_corr_tab, PSD_corr_tab=PSD_corr_tab,
                                             EPS_list=EPS_list)
#%%
plt.plot(PSD_corr_tab.T)
plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
plt.ylabel("Correlation for Vectorized Hessian")
plt.xlabel("EPS for Forward Diff")
plt.title("StyleGAN BP vs ForwardIter Pos-Semi-Definite Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN_BP-FI-PSD-HessCorr.png"))
plt.show()

#%%
plt.plot(raw_corr_tab.T)
plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
plt.ylabel("Correlation for Vectorized Hessian")
plt.xlabel("EPS for Forward Diff")
plt.title("StyleGAN BP vs ForwardIter Raw Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN_BP-FI-raw-HessCorr.png"))
plt.show()

men = raw_corr_tab.mean(axis=0)
err = raw_corr_tab.std(axis=0)/np.sqrt(raw_corr_tab.shape[0])
plt.plot(men, )
plt.fill_between(range(len(men)), men-err, men+err, alpha=0.3, label="raw")
men = PSD_corr_tab.mean(axis=0)
err = PSD_corr_tab.std(axis=0)/np.sqrt(PSD_corr_tab.shape[0])
plt.plot(men, )
plt.fill_between(range(len(men)), men-err, men+err, alpha=0.3, label="PSD")
plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
plt.legend()
plt.ylabel("Correlation for Vectorized Hessian")
plt.xlabel("EPS for Forward Diff")
plt.title("StyleGAN BP vs ForwardIter Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN_BP-FI-HessCorr-cmp.png"))
plt.savefig(join(figdir, "StyleGAN_BP-FI-HessCorr-cmp.pdf"))
plt.show()
#%%
""" modern API. Analyze the W space geometry """
from Hessian.hessian_analysis_tools import scan_hess_npz, compute_hess_corr, compute_vector_hess_corr, average_H, \
    plot_consistentcy_mat, plot_consistency_hist, plot_spectra
from GAN_utils import loadStyleGAN, StyleGAN_wrapper
SGAN = loadStyleGAN()
SG = StyleGAN_wrapper(SGAN)
SG.wspace = True
#%%
imgs = SG.visualize(SG.mean_style + torch.randn(8, 512).cuda() * 0.15)
ToPILImage()(make_grid(imgs).cpu()).show()
#%%
imgs = SG.visualize(SG.mean_style + SG.StyleGAN.style(torch.randn(8, 512).cuda()) * 0.6)
ToPILImage()(make_grid(imgs).cpu()).show()

#%%
SG.wspace = True
mean_style = SG.mean_style
datadir = r"E:\Cluster_Backup\StyleGAN_wspace"
os.makedirs(datadir, exist_ok=True)
for triali in tqdm(range(80, 150)):
    feat_z = torch.randn(1, 512).cuda()
    feat = mean_style + 0.7 * SG.StyleGAN.style(feat_z) # torch.randn(1, 512,).to("cuda")
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(SG, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 120 sec
    np.savez(join(datadir, "Hessian_rand_0_7_%03d.npz" % triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
        feat=feat.detach().cpu().numpy(), feat_z=feat_z.detach().cpu().numpy())
#%%
datadir = r"E:\Cluster_Backup\StyleGAN_wspace"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN_wspace"
modelnm = "StyleGAN_Wspace"
eva_col, evc_col, feat_col, meta = scan_hess_npz(datadir, "Hessian_rand_0_7_(\d*).npz", featkey="feat")
# compute the Mean Hessian and save
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col)
np.savez(join(figdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
# compute and plot spectra
fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum"%modelnm, figdir=figdir)
fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum_med"%modelnm, figdir=figdir, median=True)
np.savez(join(figdir, "spectra_col_%s.npz"%modelnm), eigval_col=eva_col, )
# compute and plot the correlation between hessian at different points
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=True,
                                                         savelabel=modelnm)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=True,
                                                         savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                   savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="%s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=3, titstr="%s"%modelnm, savelabel=modelnm)
