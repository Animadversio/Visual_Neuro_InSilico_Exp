from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings, get_full_hessian
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
import torchvision.models as tv
#%%
from geometry_utils import LERP, SLERP, ExpMap
from PIL import Image
from skimage.io import imsave
from torchvision.utils import make_grid
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
def LExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis]
    interp_vects = steps @ tangvect + refvect
    return interp_vects

def SExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    refnorm, tannorm = np.linalg.norm(refvect), np.linalg.norm(tangvect)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis] * np.pi / 2
    interp_vects = (np.sin(steps) @ tangvect / tannorm + np.cos(steps) @ refvect / refnorm) * refnorm
    return interp_vects
#%%
import sys
sys.path.append(r"D:\Github\BigGANsAreWatching")
sys.path.append(r"E:\Github_Projects\BigGANsAreWatching")
from BigGAN.gan_load import UnconditionalBigGAN, make_big_gan
from BigGAN.model.BigGAN import Generator
BBGAN = make_big_gan(r"E:\Github_Projects\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth", resolution=128)
for param in BBGAN.parameters():
    param.requires_grad_(False)
BBGAN.eval()
# the model is on cuda from this.
class BigBiGAN_wrapper():#nn.Module
    def __init__(self, BigBiGAN, ):
        self.BigGAN = BigBiGAN

    def visualize(self, code, scale=1.0, resolution=256):
        imgs = self.BigGAN(code, )
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = BigBiGAN_wrapper(BBGAN)
#%%
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
#%%
T00 = time()
for triali in range(20):
    for trunc in [0.1, 1, 3, 6, 9, 10, 12, 15]:
        if trunc == 0.1:
            continue
        RND = np.random.randint(1000)
        noisevect = torch.randn(1, 120)
        noisevect = noisevect / noisevect.norm()
        ref_vect = trunc * noisevect.detach().clone().cuda()
        mov_vect = ref_vect.detach().clone().requires_grad_(True)
        imgs1 = G.visualize(ref_vect)
        imgs2 = G.visualize(mov_vect)
        dsim = ImDist(imgs1, imgs2)
        H = get_full_hessian(dsim, mov_vect)  # 77sec to compute a Hessian.
        # ToPILImage()(imgs[0,:,:,:].cpu()).show()
        eigvals, eigvects = np.linalg.eigh(H)
        plt.figure(figsize=[7,5])
        plt.subplot(1, 2, 1)
        plt.plot(eigvals)
        plt.ylabel("eigenvalue")
        plt.subplot(1, 2, 2)
        plt.plot(np.log10(eigvals))
        plt.ylabel("eigenvalue (log)")
        plt.suptitle("Hessian Spectrum Full Space")
        plt.savefig(join(savedir, "Hessian_norm%d_%03d.jpg" % (trunc, RND)))
        np.savez(join(savedir, "Hess_norm%d_%03d.npz" % (trunc, RND)), H=H, eigvals=eigvals, eigvects=eigvects, vect=ref_vect.cpu().numpy(),)
        #%
        img_all = None
        for eigi in range(50): #eigvects.shape[1]
            interp_codes = LExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 15, (-2.5, 2.5))
            with torch.no_grad():
                img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda(), resolution=128).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            clear_output(wait=True)
            progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))
        imggrid = make_grid(img_all, nrow=15)
        PILimg = ToPILImage()(imggrid)#.show()
        PILimg.save(join(savedir, "eigvect_lin_norm%d_%03d.jpg" % (trunc, RND)))
        #%
        img_all = None
        for eigi in range(50): #eigvects.shape[1]
            interp_codes = SExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 21, (-1, 1))
            with torch.no_grad():
                img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda(), resolution=128).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            clear_output(wait=True)
            progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))
        imggrid = make_grid(img_all, nrow=21)
        PILimg = ToPILImage()(imggrid)#.show()
        PILimg.save(join(savedir, "eigvect_sph_norm%d_%03d.jpg" % (trunc, RND)))
        print("Spent time %.1f sec"%(time() - T00))

#%%
from glob import glob
import re
import os
import pandas as pd
import tqdm
#%%
datadir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
npzlist = glob(join(datadir, "*.npz"))
npzpattern = re.compile("Hess_norm(\d.*)_(\d\d\d)")
npzlist = sorted(npzlist)
H_record = []
eigval_col = []
code_all = []
for idx in range(len(npzlist)):
    npzname = os.path.split(npzlist[idx])[1]
    parts = npzpattern.findall(npzname)[0]
    vecnorm = int(parts[0])
    RNDid = int(parts[1])
    data = np.load(npzlist[idx])
    eigvals = data['eigvals']
    code = data['vect']
    realnorm = np.linalg.norm(code)
    code_all.append(code)
    eigval_col.append(eigvals)
    H_record.append([npzname, vecnorm, realnorm, RNDid,
         eigvals[-1], eigvals[-5], eigvals[-10], eigvals[-20], eigvals[-40], eigvals[-60]])

H_rec_tab = pd.DataFrame(H_record, columns=["npz","norm","realnorm","RND",] + ["eig%d"%idx for idx in [1,5,10,20,40,60]])
#%%
H_rec_tab = H_rec_tab.sort_values("norm")
H_rec_tab = H_rec_tab.reset_index()
H_rec_tab.to_csv(join(figdir, "Hess_record.csv"))
#%%
eigval_arr = np.array(eigval_col)[:, ::-1]

eigmean = eigval_arr.mean(axis=0)
eigstd = eigval_arr.std(axis=0)
eiglim = np.percentile(eigval_arr, [5, 95], axis=0)
def plot_spectra(control=False, indiv_trace=False, savename="spectrum_stat_120.jpg", ):
    """A local function to compute these figures for different subspaces. """
    fig = plt.figure(figsize=[10, 5])
    dimen = len(eigmean)
    plt.subplot(1,2,1)
    plt.plot(range(dimen), eigmean, alpha=0.7)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(dimen), eiglim[0, :], eiglim[1, :], alpha=0.5, color="orange", label="all space")
    if indiv_trace:
        plt.plot(eigval_arr.T, color="pink", lw=1.5, alpha=0.5)
    if control:
        plt.plot(range(len(eigmean)), eigmean, alpha=0.7, color="green")  # , eigval_arr.std(axis=0)
        plt.fill_between(range(len(eigmean)), eiglim[0, :], eiglim[1, :], alpha=0.5,
                         color="purple", label="control")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.xlim([-10, 130])
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(range(dimen), np.log10(eigmean), alpha=0.7)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(dimen), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.5, color="orange", label="all space")
    if indiv_trace:
        plt.plot(np.log10(eigval_arr).T, color="pink", lw=1.5, alpha=0.5)
    if control:
        plt.plot(range(len(eigmean)), np.log10(eigmean), alpha=0.7, color="green")  # , eigval_arr.std(axis=0)
        plt.fill_between(range(len(eigmean)), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.5,
                         color="purple", label="control")

    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.xlim([-10, 130])
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of BigBiGAN in Different Spaces\n (error bar for [5,95] percentile)")
    plt.savefig(join(figdir, savename), bbox_extra_artists=[st]) # this is working.
    # plt.show()

figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
plot_spectra(control=False, savename="spectrum_stat_120_org.jpg", )
plot_spectra(indiv_trace=True, savename="spectrum_stat_120_traces.jpg", )
# plot_spectra(control=True, savename="spectrum_stat_cmp.jpg", )
#%%
def corr_torch(V1, V2):
    C1 = (V1 - V1.mean())
    C2 = (V2 - V2.mean())
    return torch.dot(C1, C2) / C1.norm() / C2.norm()

def corr_nan_torch(V1, V2):
    Msk = torch.isnan(V1) | torch.isnan(V2)
    return corr_torch(V1[~Msk], V2[~Msk])
#%%
evc_col = []
eva_col = []
code_all = []
for idx in range(len(npzlist)):
    data = np.load(join(datadir, H_rec_tab.npz[idx]))
    eigvals = data['eigvals']
    eigvects = data['eigvects']
    evc_col.append(eigvects)
    eva_col.append(eigvals)
    code_all.append(data['vect'])

code_all = np.array(code_all).squeeze()
# np.linalg.norm(code_all, axis=1)
#%%
T0 = time()
Hnums = eigval_arr.shape[0]
corr_mat_log = torch.zeros((Hnums, Hnums)).cuda()
corr_mat_lin = torch.zeros((Hnums, Hnums)).cuda()
for eigi in tqdm.trange(Hnums):
    eigval_i, eigvect_i = eva_col[eigi], evc_col[eigi]
    evc_i = torch.from_numpy(eigvect_i).cuda()
    eva_i = torch.from_numpy(eigval_i).cuda()
    for eigj in tqdm.trange(Hnums, disable=True):
        eigval_j, eigvect_j = eva_col[eigj], evc_col[eigj]
        evc_j = torch.from_numpy(eigvect_j).cuda()
        eva_j = torch.from_numpy(eigval_j).cuda()
        inpr = evc_i.T @ evc_j
        vHv_ij = torch.diag((inpr * eva_j.unsqueeze(0)) @ inpr.T)
        corr_mat_log[eigi, eigj] = corr_nan_torch(vHv_ij.log10(), eva_j.log10())
        corr_mat_lin[eigi, eigj] = corr_nan_torch(vHv_ij, eva_j)
        # vHv_ij = np.diag(eigvect_i.T @ eigvect_j @ np.diag(eigval_j) @ eigvect_j.T @ eigvect_i)
        # corr_mat_log[eigi, eigj] = np.corrcoef(np.log10(vHv_ij), np.log10(eigvect_j))[0, 1]
        # corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eigvect_j)[0, 1]

    print(time() - T0)
print(time() - T0)  # 107.9 sev
corr_mat_log = corr_mat_log.cpu().numpy()
corr_mat_lin = corr_mat_lin.cpu().numpy()
#%
np.savez(join(figdir, "evol_hess_corr_mat.npz"), corr_mat_log=corr_mat_log,
         corr_mat_lin=corr_mat_lin, code_all=code_all)
#%%
corr_mat_log_nodiag = corr_mat_log
corr_mat_lin_nodiag = corr_mat_lin
np.fill_diagonal(corr_mat_log_nodiag, np.nan)
np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
log_nodiag_mean = np.nanmean(corr_mat_log_nodiag)
lin_nodiag_mean = np.nanmean(corr_mat_lin_nodiag)
print("Log scale mean corr value %.3f"%np.nanmean(corr_mat_log_nodiag))  # 0.984
print("Linear scale mean corr value %.3f"%np.nanmean(corr_mat_lin_nodiag))  # 0.600

plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_log, fignum=0)
plt.title("BigBiGAN Hessian at random codes\nCorrelation Mat of log of vHv and eigenvalues"
          "\nNon-Diagonal mean %.3f"%log_nodiag_mean, fontsize=15)
plt.colorbar()
plt.subplots_adjust(top=0.85)
plt.savefig(join(figdir, "BigBiGAN_H_corrmat_log.jpg"))
# plt.show()
#%
fig = plt.figure(figsize=[10, 8])
plt.matshow(corr_mat_lin, fignum=0)
plt.title("FC6GAN Hessian at random codes\nCorrelation Mat of vHv and eigenvalues"
          "\nNon-Diagonal mean %.3f"%lin_nodiag_mean, fontsize=15)
plt.colorbar()
plt.subplots_adjust(top=0.85)
plt.savefig(join(figdir, "BigBiGAN_H_corrmat_lin.jpg"))
# plt.show()
#%% L2 and correlation matrix of the code vectors
from scipy.spatial.distance import pdist, squareform
code_corr = np.corrcoef(code_all)
np.fill_diagonal(code_corr, np.nan)
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_corr, fignum=0)
plt.title("Correlation Mat of code vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_corrmat.jpg"))
#%
vdist = pdist(code_all, metric='euclidean')
code_dist = squareform(vdist)
fig = plt.figure(figsize=[10, 8])
plt.matshow(code_dist, fignum=0)
plt.title("L2 Distance Mat of code vectors", fontsize=16)
plt.colorbar()
plt.savefig(join(figdir, "code_vec_distmat.jpg"))
# plt.show()
#%% Sampling 5 random points to see how they correlates
eiglist = sorted(np.random.choice(Hnums, 5, replace=False))  # range(5)
fig = plt.figure(figsize=[10, 10], constrained_layout=False)
spec = fig.add_gridspec(ncols=5, nrows=5, left=0.05, right=0.95, top=0.9, bottom=0.05)
for axi, eigi in enumerate(eiglist):
    eigval_i, eigvect_i = eva_col[eigi], evc_col[eigi]
    for axj, eigj in enumerate(eiglist):
        eigval_j, eigvect_j = eva_col[eigj], evc_col[eigj]
        vHv_ij = np.diag(eigvect_i.T @ eigvect_j @ np.diag(eigval_j) @ eigvect_j.T @ eigvect_i)
        ax = fig.add_subplot(spec[axi, axj])
        if axi == axj:
            ax.hist(np.log10(eigval_j), 20)
        else:
            ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
            ax.set_aspect(1, adjustable='datalim')
        if axi == 4:
            ax.set_xlabel("eigvals %d" % eigj)
        if axj == 0:
            ax.set_ylabel("vHv eigvects %d" % eigi)
ST = plt.suptitle("Consistency of Hessian Across Vectors\n"
                  "Cross scatter of EigenValues and vHv values for Hessian at 5 Random Vectors",
                  fontsize=18)
plt.savefig(join(figdir, "Hess_consistency_example_rnd%03d.jpg"%np.random.randint(1000)), bbox_extra_artists=[ST]) #
# this
# is
# working.
# plt.show()
