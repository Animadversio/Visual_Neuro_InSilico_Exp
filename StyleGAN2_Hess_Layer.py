import sys
from os.path import join
from time import time
import matplotlib.pylab as plt
from matplotlib import cm
import torch
import numpy as np
sys.path.append("E:\Github_Projects\Visual_Neuro_InSilico_Exp")
sys.path.append("D:\Github\Visual_Neuro_InSilico_Exp")
import os
# os.system(r"'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat'")
import lpips
try:
    ImDist = lpips.LPIPS(net="squeeze").cuda()
except:
    ImDist = lpips.PerceptualLoss(net="squeeze").cuda()
from GAN_hessian_compute import hessian_compute, get_full_hessian
from hessian_analysis_tools import compute_vector_hess_corr, compute_hess_corr, plot_layer_consistency_mat
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper
#%%
modelname = "ffhq-256-config-e-003810"  # 109 sec
SGAN = loadStyleGAN2(modelname+".pt", size=256, channel_multiplier=1)  # 491 sec per BP
#%%
L2dist_col = []
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None

feat = torch.randn(1, 512, requires_grad=True).cuda()
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
for blocki in range(len(SGAN.convs)):
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = SGAN.convs[blocki].register_forward_hook(Hess_hook)
    img = SGAN([feat], truncation=1)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col[0], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(datadir, "eig_genBlock%02d.npz"%blocki), H=H00, eva=eva00, evc=evc00)
    plt.plot(np.log10(eva00)[::-1])
    plt.title("GenBlock %d Spectra" % (blocki,))
    plt.xlim([0, len(evc00)])
    plt.savefig(join(datadir, "spectrum_genBlock%d.png" % (blocki)))
    # plt.show()
#%%
m_latent = SGAN.mean_latent(4096)
feat = torch.randn(1, 512, requires_grad=True).cuda()
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
plt.figure()
for blocki in range(len(SGAN.convs)):
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = SGAN.convs[blocki].register_forward_hook(Hess_hook)
    img = SGAN([feat], truncation=0.8, truncation_latent=m_latent, input_is_latent=True)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col[0], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(datadir, "eig_genBlock%02d_latent.npz"%blocki), H=H00, eva=eva00, evc=evc00)
    plt.plot(np.log10(eva00)[::-1])
    plt.title("GenBlock %d Spectra" % (blocki,))
    plt.xlim([0, len(evc00)])
    plt.savefig(join(datadir, "spectrum_genBlock%d_latent.png" % (blocki)))
    # plt.show()
#%%
def plot_layer_spectra(eva_col, layernames=None, titstr="GAN", namestr="all_layers", figdir=datadir,
                       normalize=False, cmap=cm.jet):
    Ln = len(eva_col)
    if layernames is None: layernames = ["Module %d"%i for i in range(Ln)]
    fig = plt.figure(figsize=[7, 8])
    for blocki, eva00 in enumerate(eva_col):
        norm = eva00.max() if normalize else 1
        plt.plot(np.log10(eva00 / norm)[::-1], color=cmap((blocki+1) / Ln),
                     label=layernames[blocki])
    plt.xlim([0, len(eva00)])
    plt.xlabel("eigenvalue rank")
    plt.ylabel("log(eig value)")
    plt.title("%s Hessian Spectra of Intermediate Layers Compared"%titstr)
    plt.subplots_adjust(top=0.9)
    plt.legend()
    plt.savefig(join(figdir, "spectrum_%s_cmp.png"%namestr))
    plt.savefig(join(figdir, "spectrum_%s_cmp.pdf"%namestr))
    plt.show()
    return fig

from hessian_analysis_tools import plot_consistentcy_mat, compute_hess_corr, compute_vector_hess_corr, plot_layer_consistency_mat
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
layernames = [("StyleBlock%02d" % blocki) for blocki in range(12)] # if blocki!=8 else "SelfAttention"
eva_col, evc_col = [], []
for blocki in range(12):
    data = np.load(join(datadir, "eig_genBlock%02d_latent.npz"%blocki))
    # data = np.load(join(datadir, "eig_ConvBlock%02d.npz"%blocki))
    eva_col.append(data["eva"])
    evc_col.append(data["evc"])
fig0 = plot_layer_spectra(eva_col, layernames=layernames, figdir=datadir, titstr="StyleGAN2",
                          namestr="wspace_all_block")
fig0 = plot_layer_spectra(eva_col, layernames=layernames, figdir=datadir, titstr="StyleGAN2", normalize=True,
                          namestr="wspace_all_block_norm")
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, savelabel="StyleGAN2_wspace", figdir=datadir)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, savelabel="StyleGAN2_wspace", figdir=datadir)
fig1, fig2, fig3 = plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="StyleGAN2_wspace",
                                      figdir=datadir, titstr="StyleGAN2", layernames=layernames)
#%%
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\StyleGAN2"
layernames = [("StyleBlock%02d" % blocki) for blocki in range(12)] # if blocki!=8 else "SelfAttention"
eva_col, evc_col = [], []
for blocki in range(12):
    # data = np.load(join(datadir, "eig_genBlock%02d_latent.npz"%blocki))
    data = np.load(join(datadir, "eig_genBlock%02d.npz"%blocki))
    eva_col.append(data["eva"])
    evc_col.append(data["evc"])
fig0 = plot_layer_spectra(eva_col, layernames=layernames, figdir=datadir, titstr="StyleGAN2",
                          namestr="zspace_all_block")
fig0 = plot_layer_spectra(eva_col, layernames=layernames, figdir=datadir, titstr="StyleGAN2", normalize=True,
                          namestr="zspace_all_block_norm")
from hessian_analysis_tools import plot_consistentcy_mat, compute_hess_corr, compute_vector_hess_corr, plot_layer_consistency_mat
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, savelabel="StyleGAN2_zspace", figdir=datadir)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, savelabel="StyleGAN2_zspace", figdir=datadir)
fig1, fig2, fig3 = plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="StyleGAN2_zspace",
                                      figdir=datadir, titstr="StyleGAN2", layernames=layernames)
#%%

