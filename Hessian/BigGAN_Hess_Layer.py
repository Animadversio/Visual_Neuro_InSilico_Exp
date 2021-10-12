
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from time import time
from os.path import join
import sys
import lpips
from GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
#%%
BGAN = loadBigGAN()
L2dist_col = []
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None
#%%
clas_vec = BGAN.embeddings.weight[:,120].detach().clone().unsqueeze(0)
feat = torch.cat((0.7*torch.randn(1, 128).cuda(), clas_vec), dim=1)
feat.requires_grad_(True)
#%%
"""Compute Hessian towards middle layers layer by layer"""
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
L2dist_col = []
torch.cuda.empty_cache()
H1 = BGAN.generator.gen_z.register_forward_hook(Hess_hook)
img = BGAN.generator(feat, 0.7)
H1.remove()
T0 = time()
H00 = get_full_hessian(L2dist_col[0], feat)
eva00, evc00 = np.linalg.eigh(H00)
print("Spent %.2f sec computing" % (time() - T0))
np.savez(join(datadir, "eig_gen_z.npz"), H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00)[::-1])
plt.title("gen_z Linear Layer Spectra" % ())
plt.xlim([0, len(evc00)])
plt.savefig(join(datadir, "spectrum_gen_z.png" ))
plt.show()
#%%
"""Compute Hessian towards each layer"""
for blocki in range(len(BGAN.generator.layers)):
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = BGAN.generator.layers[blocki].register_forward_hook(Hess_hook)
    img = BGAN.generator(feat, 0.7)
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
    plt.show()
#%%
plt.figure(figsize=[7,8])
Ln = len(BGAN.generator.layers)
eva00 = np.load(join(datadir, "eig_gen_z.npz"))["eva"] #, H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00)[::-1], label="gen_z")
for blocki in range(Ln):
    eva00 = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"]
    plt.plot(np.log10(eva00)[::-1], color=cm.jet((blocki+1) / Ln),
             label=("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention")
plt.xlim([0, len(evc00)])
plt.xlabel("eigenvalue rank")
plt.ylabel("log(eig value)")
plt.title("BigGAN Hessian Spectra of Intermediate Layers Compared")
plt.subplots_adjust(top=0.9)
plt.legend()
plt.savefig(join(datadir, "spectrum_all_Layers_cmp.png"))
plt.show()
#%% Normalized
plt.figure(figsize=[7,8])
Ln = len(BGAN.generator.layers)
eva00 = np.load(join(datadir, "eig_gen_z.npz"))["eva"] #, H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00/eva00.max())[::-1], label="gen_z")
for blocki in range(Ln):
    eva00 = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"]
    plt.plot(np.log10(eva00/eva00.max())[::-1], color=cm.jet((blocki+1) / Ln),
             label=("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention")
plt.xlim([0, len(evc00)])
plt.xlabel("eigenvalue rank")
plt.ylabel("log(eig value/max(eig))")
plt.title("BigGAN Hessian Normalized Spectra of Intermediate Layers Compared")
plt.subplots_adjust(top=0.9)
plt.legend()
plt.savefig(join(datadir, "spectrum_all_Layers_maxnorm.png"))
plt.show()
#%%
"""Loading and analyzing data of Normal GAN"""
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
Ln = len(BGAN.generator.layers) # 13
layernames = ["gen_z"] + [("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention" for blocki in range(Ln)] + [
    "Image"]
data = np.load(join(datadir, "eig_gen_z.npz"))
eva_col = [data["eva"]]
evc_col = [data["evc"]]
H_col   = [data["H"]]
for blocki in range(Ln):
    data = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))
    eva_col.append(data["eva"])
    evc_col.append(data["evc"])
    H_col.append(data["H"])

realfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
data = np.load(join(realfigdir, "H_avg_1000cls.npz"))
eva_Havg, evc_Havg, Havg = data['eigvals_avg'], data['eigvects_avg'], data['H_avg']
H_col.append(Havg)
eva_col.append(eva_Havg)
evc_col.append(evc_Havg)
#%% plot examples as A applied to B

#%% Compute Relationships between Hessian in different layers
Lnum = len(H_col)
corr_mat_lin = np.zeros((Lnum, Lnum))
corr_mat_log = np.zeros((Lnum, Lnum))
log_reg_slope = np.zeros((Lnum, Lnum))
log_reg_intcp = np.zeros((Lnum, Lnum))
for Li in range(len(H_col)):
    evc = evc_col[Li]
    eva = eva_col[Li]
    for Lj in range(len(H_col)):
        H = H_col[Lj]
        alphavec = np.diag(evc.T @ H @ evc)
        corr_mat_lin[Li, Lj] = np.corrcoef(alphavec, eva)[0,1]
        corr_mat_log[Li, Lj] = np.corrcoef(np.log10(alphavec), np.log10(eva))[0,1]
        slope, intercept = np.polyfit(np.log10(eva), np.log10(alphavec), 1)
        log_reg_slope[Li, Lj] = slope
        log_reg_intcp[Li, Lj] = intercept
#%%
fig = plot_layer_mat(corr_mat_lin, layernames=layernames, titstr="Linear Correlation of Amplification in BigGAN")
fig.savefig(join(figdir, "BigGAN_Layer_corr_lin_mat.pdf"))
fig = plot_layer_mat(corr_mat_log, layernames=layernames, titstr="Log scale Correlation of Amplification in BigGAN")
fig.savefig(join(figdir, "BigGAN_Layer_corr_log_mat.pdf"))
fig = plot_layer_mat(log_reg_slope, layernames=layernames, titstr="Log scale Slope of Amplification in BigGAN")
fig.savefig(join(figdir, "BigGAN_Layer_log_reg_slope.pdf"))
fig = plot_layer_mat(log_reg_intcp, layernames=layernames, titstr="Log scale intercept of Amplification in BigGAN")
fig.savefig(join(figdir, "BigGAN_Layer_log_reg_intercept.pdf"))
#%%
savestr = "BigGAN"
colorseq = [cm.jet(Li / (Lnum-1) ) for Li in range(Lnum)] # color for each curve.
for Li in range(Lnum):
    alphavec_col = []
    evc = evc_col[Li]
    eva = eva_col[Li]
    plt.figure(figsize=[5,4])
    for Lj in range(Lnum):
        H = H_col[Lj]
        alphavec = np.diag(evc.T @ H @ evc)
        alphavec_col.append(alphavec)
        plt.plot(np.log10(alphavec[::-1]), label=layernames[Lj], color=colorseq[Lj], lw=2)
    plt.xlabel("Rank of eigenvector (layer %d %s)"%(Li, layernames[Li]))
    plt.ylabel("Amplification") #  (layer %d %s)"%(Lj, layernames[Lj]
    plt.legend()
    plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d.png"%(savestr, Li)))
    plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d.pdf"%(savestr, Li)))
    plt.show()
#%%
def plot_layer_mat(layer_mat, layernames=None, titstr="Correlation of Amplification in BigGAN"):
    """Formatting function for ploting Layer by Layer matrix"""
    Lnum = layer_mat.shape[0]
    fig = plt.figure(figsize=[9, 8])
    plt.matshow(layer_mat, fignum=0)
    layermat_nan = layer_mat.copy()
    np.fill_diagonal(layermat_nan, np.nan)
    plt.title("%s across %d layers"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, Lnum, np.nanmean(layermat_nan), np.nanmedian(layermat_nan)), fontsize=15)
    fig.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(Lnum), layernames)
        plt.ylim(-0.5, Lnum - 0.5)
        plt.xticks(range(Lnum), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, Lnum - 0.5)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.show()
    return fig

def compute_plot_layer_corr_mat(eva_col, evc_col, H_col, layernames, titstr="BigGAN", savestr="BigGAN", figdir=""):
    Lnum = len(H_col)
    corr_mat_lin = np.zeros((Lnum, Lnum))
    corr_mat_log = np.zeros((Lnum, Lnum))
    log_reg_slope = np.zeros((Lnum, Lnum))
    log_reg_intcp = np.zeros((Lnum, Lnum))
    for Li in range(len(H_col)):
        evc = evc_col[Li]
        eva = eva_col[Li]
        for Lj in range(len(H_col)):
            H = H_col[Lj]
            alphavec = np.diag(evc.T @ H @ evc)
            log10alphavec = np.log10(alphavec)
            log10eva = np.log10(eva)
            corr_mat_lin[Li, Lj] = np.corrcoef(alphavec, eva)[0,1]
            corr_mat_log[Li, Lj] = np.corrcoef(log10alphavec, log10eva)[0,1]
            nanmask = (~np.isnan(log10alphavec)) * (~np.isnan(log10eva))
            slope, intercept = np.polyfit(log10eva[nanmask], log10alphavec[nanmask], 1)
            log_reg_slope[Li, Lj] = slope
            log_reg_intcp[Li, Lj] = intercept
    fig1 = plot_layer_mat(corr_mat_lin, layernames=layernames, titstr="Linear Correlation of Amplification in %s"%titstr)
    fig1.savefig(join(figdir, "%s_Layer_corr_lin_mat.pdf"%savestr))
    fig2 = plot_layer_mat(corr_mat_log, layernames=layernames, titstr="Log scale Correlation of Amplification in %s"%titstr)
    fig2.savefig(join(figdir, "%s_Layer_corr_log_mat.pdf"%savestr))
    fig3 = plot_layer_mat(log_reg_slope, layernames=layernames, titstr="Log scale Slope of Amplification in %s"%titstr)
    fig3.savefig(join(figdir, "%s_Layer_log_reg_slope.pdf"%savestr))
    fig4 = plot_layer_mat(log_reg_intcp, layernames=layernames, titstr="Log scale intercept of Amplification in %s"%titstr)
    fig4.savefig(join(figdir, "%s_Layer_log_reg_intercept.pdf"%savestr))
    return corr_mat_lin, corr_mat_log, log_reg_slope, log_reg_intcp, fig1, fig2, fig3, fig4, 

def plot_layer_amplif_curves(eva_col, evc_col, H_col, layernames, savestr="", figdir="",
                             maxnorm=False):
    Lnum = len(evc_col)
    colorseq = [cm.jet(Li / (Lnum - 1)) for Li in range(Lnum)]  # color for each curve.
    for Li in range(Lnum):  # source of eigenvector basis
        alphavec_col = []
        evc = evc_col[Li]
        eva = eva_col[Li]
        plt.figure(figsize=[5, 4])
        for Lj in range(Lnum):  # hessian target
            H = H_col[Lj]
            alphavec = np.diag(evc.T @ H @ evc)
            alphavec_col.append(alphavec)
            scaler = alphavec[-1] if maxnorm else 1
            plt.plot(np.log10(alphavec[::-1] / scaler), label=layernames[Lj], color=colorseq[Lj], lw=2, alpha=0.7)
        plt.xlabel("Rank of eigenvector (layer %d %s)" % (Li, layernames[Li]))
        plt.ylabel("Amplification (normalize max to 1)" if maxnorm else "Amplification")  # (layer %d %s)"%(Lj, layernames[Lj]
        plt.legend()
        plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d%s.png" % (savestr, Li, "_mxnorm" if maxnorm else "")))
        plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d%s.pdf" % (savestr, Li, "_mxnorm" if maxnorm else "")))
        plt.show()

def plot_layer_amplif_consistency(eigval_col, eigvec_col, layernames, layeridx=[0,1,-1], titstr="GAN", figdir="",
                                   savelabel=""):
    nsamp = len(layeridx)
    print("Plot hessian of layers : ", [layernames[idx] for idx in layeridx])
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, Li in enumerate(layeridx):
        eigval_i, eigvect_i = eigval_col[Li], eigvec_col[Li]
        for axj, Lj in enumerate(layeridx):
            eigval_j, eigvect_j = eigval_col[Lj], eigvec_col[Lj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_i), 20)
            else:
                ax.scatter(np.log10(eigval_i), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp - 1:
                ax.set_xlabel("H @ %s" % layernames[Lj])
            if axj == 0:
                ax.set_ylabel("eigvect @ %s" % layernames[Li])
    ST = plt.suptitle("Consistency of %s Amplification Factor Across Layers\n"
                      "Scatter of AmpFact for Hessian at %d Layers\n Source of EigVect on y axes, Source of Hessian "
                      "on x axes" % (titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Amplif_layer_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Amplif_layer_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig
#%%
plot_layer_amplif_curves(eva_col, evc_col, H_col, layernames=layernames, savestr="BigGAN")
plot_layer_amplif_curves(eva_col, evc_col, H_col, layernames=layernames, savestr="BigGAN", maxnorm=True)
#%%
plot_layer_amplif_consistency(eva_col, evc_col, layernames, layeridx=[0,1,2], titstr="BigGAN", figdir=figdir,
                                   savelabel="BigGAN")
plot_layer_amplif_consistency(eva_col, evc_col, layernames, layeridx=[1,9,12], titstr="BigGAN", figdir=figdir,
                                   savelabel="BigGAN")
#%%
"""Load and analyze Hessian for shuffled BigGAN"""
ctrl_datadir = join(datadir, r"ctrl_Hessians")
layernames_ctrl = [layernames[i+1] for i in [0, 3, 5, 8, 10, 12]] + ["Image"]
triali = 1
eva_col_ctrl = []
evc_col_ctrl = []
H_col_ctrl   = []
for blocki in [0, 3, 5, 8, 10, 12]:#range(Ln):
    data = np.load(join(ctrl_datadir, "eig_genBlock%02d_trial%d.npz" % (blocki,triali)))
    eva_col_ctrl.append(data["eva"])
    evc_col_ctrl.append(data["evc"])
    H_col_ctrl.append(data["H"])

data = np.load(join(ctrl_datadir, "eig_full_trial%d.npz"%triali))
eva_col_ctrl.append(data["eva"])
evc_col_ctrl.append(data["evc"])
H_col_ctrl.append(data["H"])
#%%
plot_layer_amplif_curves(eva_col_ctrl, evc_col_ctrl, H_col_ctrl, layernames=layernames_ctrl, savestr="BigGAN_shfl",
                         figdir=figdir)
plot_layer_amplif_curves(eva_col_ctrl, evc_col_ctrl, H_col_ctrl, layernames=layernames_ctrl, savestr="BigGAN_shfl",
                         figdir=figdir, maxnorm=True)
#%%
ctrl_datadir = join(datadir, r"ctrl_Hessians")
layernames_ctrl = layernames
triali = 100
data = np.load(join(ctrl_datadir, "eig_gen_z_trial%d.npz" % (triali)))
eva_col_ctrl = [data["eva"]]
evc_col_ctrl = [data["evc"]]
H_col_ctrl   = [data["H"]]
for blocki in range(Ln):
    data = np.load(join(ctrl_datadir, "eig_genBlock%02d_trial%d.npz" % (blocki,triali)))
    eva_col_ctrl.append(data["eva"])
    evc_col_ctrl.append(data["evc"])
    H_col_ctrl.append(data["H"])

data = np.load(join(ctrl_datadir, "eig_full_trial%d.npz"%triali))
eva_col_ctrl.append(data["eva"])
evc_col_ctrl.append(data["evc"])
H_col_ctrl.append(data["H"])
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
plot_layer_amplif_curves(eva_col_ctrl, evc_col_ctrl, H_col_ctrl, layernames=layernames_ctrl, savestr="BigGAN_shfl_all",
                         figdir=figdir)
plot_layer_amplif_curves(eva_col_ctrl, evc_col_ctrl, H_col_ctrl, layernames=layernames_ctrl, savestr="BigGAN_shfl_all",
                         figdir=figdir, maxnorm=True)
plot_layer_amplif_consistency(eva_col_ctrl, evc_col_ctrl, layernames, layeridx=[0,1,2], titstr="BigGAN_shfl_all",
                         figdir=figdir, savelabel="BigGAN")
plot_layer_amplif_consistency(eva_col_ctrl, evc_col_ctrl, layernames, layeridx=[1,9,12], titstr="BigGAN_shfl_all",
                         figdir=figdir, savelabel="BigGAN")
#%%
# corr_mat_lin, corr_mat_log, log_reg_slope, log_reg_intcp, _, _, _, _, = compute_plot_layer_corr_mat(eva_col, evc_col, H_col, layernames, titstr="BigGAN", savestr="BigGAN", figdir=figdir)
corr_mat_lin_ctrl, corr_mat_log_ctrl, log_reg_slope_ctrl, log_reg_intcp_ctrl, _, _, _, _, = compute_plot_layer_corr_mat(eva_col_ctrl, evc_col_ctrl, H_col_ctrl, layernames_ctrl, titstr="BigGAN_shfl", savestr="BigGAN_shfl", figdir=figdir)