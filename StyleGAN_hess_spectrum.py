#%%
import torch
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
from os.path import join
import sys
import lpips
from GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
use_gpu = True if torch.cuda.is_available() else False
ImDist = lpips.LPIPS(net='squeeze').cuda()
#%%
model = torch.hub.load('ndahlquist/pytorch-hub-stylegan:0.0.1', 'style_gan', pretrained=True)
class StyleGAN_wrapper():  # nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN

    def visualize(self, code, scale=1):
        imgs = self.StyleGAN.forward(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale
G = StyleGAN_wrapper(model.cuda())
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN"
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

# T0 = time()
# eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=1E-3, preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True))
# print("%.2f sec" % (time() - T0))  # 252.28 sec
#
# G.StyleGAN.cpu()
# T0 = time()
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP", preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True), device="cpu") # this takes 16384 sec...
# print("%.2f sec" % (time() - T0))
#
# G.StyleGAN.cuda()
# T0 = time()
# eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP", preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)) # this will exceed gpu memory
# print("%.2f sec" % (time() - T0))
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