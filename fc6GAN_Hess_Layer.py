import numpy as np
from GAN_utils import upconvGAN
import torch
from GAN_hvp_operator import compute_hessian_eigenthings,get_full_hessian,GANForwardHVPOperator,GANHVPOperator
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from os.path import join
# import seaborn as sns
#%%
G = upconvGAN()
G.G.requires_grad_(False)
layernames = [name for name, _ in G.G.named_children()]
#%%
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None

#%%
feat = torch.randn(4096, requires_grad=True)
archdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit"
#%%
layernames = [name for name, _ in G.G.named_children()]
eva_col = []
from time import time
for Li in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ]:#
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = G.G[Li].register_forward_hook(Hess_hook)
    img = G.visualize(feat)
    H1.remove()
    T0 = time()
    H10 = get_full_hessian(L2dist_col[0], feat)
    eva10, evc10 = np.linalg.eigh(H10)
    print("Layer %d, cost %.2f sec" % (Li, time() - T0))
    #%
    np.savez(join(archdir, "eig_Layer%d.npz" % (Li)), evc=evc10, eva=eva10)
    plt.plot(np.log10(eva10)[::-1])
    plt.title("Layer %d %s\n%s"%(Li, layernames[Li], G.G[Li].__repr__()))
    plt.xlim([0, 4096])
    plt.savefig(join(archdir, "spectrum_Layer%d.png" % (Li)))
    plt.show()
#%%
eva_col = []
for Li in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    with np.load(join(archdir, "eig_Layer%d.npz" % (Li))) as data:
        eva_col.append(data["eva"].copy())
#%%
np.savez(join(archdir, "spect_each_layer.npz"), eva_col=np.array(eva_col))
#%%
plt.figure(figsize=[7, 9])
for eva, name in zip(eva_col, layernames):
    plt.plot(np.log10(eva)[::-1], label=name)
plt.legend()
plt.xlim([0, 4096])
plt.title("Spectrum of Jacobian / Hessian Up to Each Layer")
plt.savefig(join(archdir, "spectrum_Layer_all.png"))
plt.savefig(join(archdir, "spectrum_Layer_all.pdf"))
plt.show()
#%%
archdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit"
from Hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_consistentcy_mat
# scan_hess_npz(archdir, npzpat="eig_Layer(\d).npz", evakey='eva_BP', evckey='evc_BP', )
eva_col = []
evc_col = []
for Li in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    with np.load(join(archdir, "eig_Layer%d.npz" % (Li))) as data:
        eva_col.append(data["eva"].copy())
        evc_col.append(data["evc"].copy())

#%%
from tqdm import tqdm
from time import time
def compute_vector_hess_corr(eigval_col, eigvec_col, savelabel="", figdir="", use_cuda=False):
    posN = len(eigval_col)
    T0 = time()
    # corr_mat_log = np.zeros((posN, posN))
    if use_cuda:
        corr_mat_lin = torch.zeros((posN, posN)).cuda()
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = torch.from_numpy(eigval_col[eigi]).cuda(), torch.from_numpy(eigvec_col[eigi]).cuda()
            H_i = (evc_i * eva_i.unsqueeze(0)) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = torch.from_numpy(eigval_col[eigj]).cuda(), torch.from_numpy(eigvec_col[eigj]).cuda()
                H_j = (evc_j * eva_j.unsqueeze(0)) @ evc_j.T
                corr_mat_lin[eigi, eigj] = corr_torch(H_i.flatten(), H_j.flatten())
        corr_mat_lin = corr_mat_lin.cpu().numpy()
    else:
        corr_mat_lin = np.zeros((posN, posN))
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi]
            H_i = (evc_i * eva_i[np.newaxis, :]) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj]
                H_j = (evc_j * eva_j[np.newaxis, :]) @ evc_j.T
                # corr_mat_log[eigi, eigj] = \
                # np.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(eva_j)))[0, 1]
                corr_mat_lin[eigi, eigj] = np.corrcoef(H_i.flatten(), H_j.flatten())[0, 1]
    print("%.1f sec" % (time() - T0))  #
    return corr_mat_lin

corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, use_cuda=True)
# 576 sec without cuda 92.7 sec with cuda
#%% Older version hess corr
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, use_cuda=True)
#%%
def plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="", figdir=archdir, titstr="GAN", layernames=None):
    posN = corr_mat_log.shape[0]
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    corr_mat_vec_nodiag = corr_mat_vec.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
    np.fill_diagonal(corr_mat_vec_nodiag, np.nan)
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    vec_nodiag_mean_cc = np.nanmean(corr_mat_vec_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    vec_nodiag_med_cc = np.nanmedian(corr_mat_vec_nodiag)
    print("Log scale corr non-diag mean value %.3f"%log_nodiag_mean_cc)
    print("Lin scale corr non-diag mean value %.3f"%lin_nodiag_mean_cc)
    print("Vec Hessian corr non-diag mean value %.3f"%vec_nodiag_mean_cc)
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    fig1.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN-0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.pdf"%savelabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    fig2.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.pdf"%savelabel))
    plt.show()

    fig3 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_vec, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vectorized Hessian Mat"
              "\nNon-Diagonal mean %.3f median %.3f" % (titstr, posN, vec_nodiag_mean_cc, vec_nodiag_med_cc),
              fontsize=15)
    plt.colorbar()
    fig3.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.jpg" % savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.pdf" % savelabel))
    plt.show()
    return fig1, fig2, fig3

Li_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="fc6GAN", titstr="fc6GAN", figdir=archdir, layernames=[layernames[Li] for Li in Li_list])
#%%
""" The matrix entry M_ij means the eigenvector of Hessian i applies to Hessian j. this matrix correlated with eigenvalues of matrix j forms the correlation M_ij """