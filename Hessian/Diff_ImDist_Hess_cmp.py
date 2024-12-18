
import os
from os.path import join
import pandas as pd
import numpy as np
import torch
from Hessian.GAN_hessian_compute import hessian_compute
# from hessian_analysis_tools import scan_hess_npz, plot_spectra, average_H, compute_hess_corr, plot_consistency_example
# from hessian_axis_visualize import vis_eigen_explore, vis_eigen_action, vis_eigen_action_row, vis_eigen_explore_row
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper, loadBigGAN, BigGAN_wrapper, loadPGGAN, PGGAN_wrapper
import matplotlib.pylab as plt
import matplotlib
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import lpips
ImDist = lpips.LPIPS(net="squeeze").cuda()
# SSIM
import pytorch_msssim
D = pytorch_msssim.SSIM()  # note SSIM, higher the score the more similar they are. So to confirm to the distance convention, we use 1 - SSIM as a proxy to distance.
# L2 / MSE
def MSE(im1, im2):
    return (im1 - im2).pow(2).mean(dim=[1,2,3])
# L1 / MAE
# def L1(im1, im2):
#     return (im1 - im2).abs().mean(dim=[1,2,3])
# Note L1 is less proper for this task, as it's farther away from a distance square function.
#%% Utility functions to quantify relationship between 2 eigvals or 2 Hessians.
def spectra_cmp(eigvals1, eigvals2, show=True):
    cc = np.corrcoef((eigvals1), (eigvals2))[0, 1]
    logcc = np.corrcoef(np.log10(np.abs(eigvals1)+1E-8), np.log10(np.abs(eigvals2)+1E-8))[0, 1]
    reg_coef = np.polyfit((eigvals1), (eigvals2), 1)
    logreg_coef = np.polyfit(np.log10(np.abs(eigvals1)+1E-8), np.log10(np.abs(eigvals2)+1E-8), 1)
    if show:
        print("Correlation %.3f (lin) %.3f (log). Regress Coef [%.2f, %.2f] (lin) [%.2f, %.2f] (log)"%
            (cc, logcc, *tuple(reg_coef), *tuple(logreg_coef)))
    return cc, logcc, reg_coef, logreg_coef

def Hessian_cmp(eigvals1, eigvecs1, H1, eigvals2, eigvecs2, H2, show=True):
    H_cc = np.corrcoef(H1.flatten(), H2.flatten())[0,1] 
    logH1 = eigvecs1 * np.log10(np.abs(eigvals1))[np.newaxis, :] @ eigvecs1.T
    logH2 = eigvecs2 * np.log10(np.abs(eigvals2))[np.newaxis, :] @ eigvecs2.T
    logH_cc = np.corrcoef(logH1.flatten(), logH2.flatten())[0, 1]
    if show:
        print("Entrywise Correlation Hessian %.3f log Hessian %.3f (log)"% (H_cc, logH_cc,))
    return H_cc, logH_cc

def top_eigvec_corr(eigvects1, eigvects2, eignum=10):
    cc_arr = []
    for eigi in range(eignum):
        cc = np.corrcoef(eigvects1[:, -eigi-1], eigvects2[:, -eigi-1])[0, 1]
        cc_arr.append(cc)
    return np.abs(cc_arr)

def eigvec_H_corr(eigvals1, eigvects1, H1, eigvals2, eigvects2, H2, show=True):
    vHv12 = np.diag(eigvects1.T @ H2 @ eigvects1)
    vHv21 = np.diag(eigvects2.T @ H1 @ eigvects2)
    cc_12 = np.corrcoef(vHv12, eigvals2)[0, 1]
    cclog_12 = np.corrcoef(np.log(np.abs(vHv12)+1E-8), np.log(np.abs(eigvals2+1E-8)))[0, 1]
    cc_21 = np.corrcoef(vHv21, eigvals1)[0, 1]
    cclog_21 = np.corrcoef(np.log(np.abs(vHv21)+1E-8), np.log(np.abs(eigvals1+1E-8)))[0, 1]
    if show:
        print("Applying eigvec 1->2: corr %.3f (lin) %.3f (log)"%(cc_12, cclog_12))
        print("Applying eigvec 2->1: corr %.3f (lin) %.3f (log)"%(cc_21, cclog_21))
    return cc_12, cclog_12, cc_21, cclog_21
#%%
saveroot = r"E:\Cluster_Backup"
#%%
BGAN = loadBigGAN()
G = BigGAN_wrapper(BGAN)
savedir = join(saveroot, "ImDist_cmp\\BigGAN")
os.makedirs(savedir, exist_ok=True)
SSIM_stat_col = []
MSE_stat_col = []
for idx in range(100):
    refvec = G.sample_vector(1, device="cuda")# 0.1 * torch.randn(1, 256)
    eigvals_PS, eigvects_PS, H_PS = hessian_compute(G, refvec, ImDist, hessian_method="BP")
    eigvals_SSIM, eigvects_SSIM, H_SSIM = hessian_compute(G, refvec, D, hessian_method="BP")
    eigvals_MSE, eigvects_MSE, H_MSE = hessian_compute(G, refvec, MSE, hessian_method="BP")
    #% eigvals_L1, eigvects_L1, H_L1 = hessian_compute(G, refvec, L1, hessian_method="BP")
    print("SSIM - LPIPS comparison")
    cc_SSIM, logcc_SSIM, reg_coef_SSIM, logreg_coef_SSIM = spectra_cmp(-eigvals_SSIM[::-1], eigvals_PS, show=True)
    H_cc_SSIM, logH_cc_SSIM = Hessian_cmp(-eigvals_SSIM[::-1], eigvects_SSIM, -H_SSIM, eigvals_PS, eigvects_PS, H_PS, show=True)
    SSIM_stat_col.append((idx, cc_SSIM, logcc_SSIM, *tuple(reg_coef_SSIM), *tuple(logreg_coef_SSIM), H_cc_SSIM, logH_cc_SSIM))
    print("MSE - LPIPS comparison")
    cc_MSE, logcc_MSE, reg_coef_MSE, logreg_coef_MSE = spectra_cmp(eigvals_MSE, eigvals_PS, show=True)
    H_cc_MSE, logH_cc_MSE = Hessian_cmp(eigvals_MSE, eigvects_MSE, H_MSE, eigvals_PS, eigvects_PS, H_PS, show=True)
    MSE_stat_col.append((idx, cc_MSE, logcc_MSE, *tuple(reg_coef_MSE), *tuple(logreg_coef_MSE), H_cc_MSE, logH_cc_MSE))
    np.savez(join(savedir,"Hess_cmp_%03d.npz"%idx), **{"eva_PS":eigvals_PS, "evc_PS":eigvects_PS, "H_PS":H_PS, 
                      "eva_SSIM":eigvals_SSIM, "evc_SSIM":eigvects_SSIM, "H_SSIM":H_SSIM, 
                      "eva_MSE":eigvals_MSE, "evc_MSE":eigvects_MSE, "H_MSE":H_MSE,})

np.savez(join(savedir, "H_cmp_stat.npz"), MSE_stat=MSE_stat_col, SSIM_stat=SSIM_stat_col)
MSE_stat_tab = pd.DataFrame(MSE_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
MSE_stat_tab.to_csv(join(savedir, "H_cmp_MSE_stat.csv"))
SSIM_stat_tab = pd.DataFrame(SSIM_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
SSIM_stat_tab.to_csv(join(savedir, "H_cmp_SSIM_stat.csv"))
del G, BGAN
torch.cuda.empty_cache()
#%%
PGGAN = loadPGGAN()
G = PGGAN_wrapper(PGGAN)
savedir = join(saveroot, "ImDist_cmp\\PGGAN")
os.makedirs(savedir, exist_ok=True)
SSIM_stat_col = []
MSE_stat_col = []
for idx in range(100):
    refvec = G.sample_vector(1, device="cuda")# 0.1 * torch.randn(1, 256)
    eigvals_PS, eigvects_PS, H_PS = hessian_compute(G, refvec, ImDist, hessian_method="BP")
    eigvals_SSIM, eigvects_SSIM, H_SSIM = hessian_compute(G, refvec, D, hessian_method="BP")
    eigvals_MSE, eigvects_MSE, H_MSE = hessian_compute(G, refvec, MSE, hessian_method="BP")
    #% eigvals_L1, eigvects_L1, H_L1 = hessian_compute(G, refvec, L1, hessian_method="BP")
    print("SSIM - LPIPS comparison")
    cc_SSIM, logcc_SSIM, reg_coef_SSIM, logreg_coef_SSIM = spectra_cmp(-eigvals_SSIM[::-1], eigvals_PS, show=True)
    H_cc_SSIM, logH_cc_SSIM = Hessian_cmp(-eigvals_SSIM[::-1], eigvects_SSIM, -H_SSIM, eigvals_PS, eigvects_PS, H_PS, show=True)
    SSIM_stat_col.append((idx, cc_SSIM, logcc_SSIM, *tuple(reg_coef_SSIM), *tuple(logreg_coef_SSIM), H_cc_SSIM, logH_cc_SSIM))
    print("MSE - LPIPS comparison")
    cc_MSE, logcc_MSE, reg_coef_MSE, logreg_coef_MSE = spectra_cmp(eigvals_MSE, eigvals_PS, show=True)
    H_cc_MSE, logH_cc_MSE = Hessian_cmp(eigvals_MSE, eigvects_MSE, H_MSE, eigvals_PS, eigvects_PS, H_PS, show=True)
    MSE_stat_col.append((idx, cc_MSE, logcc_MSE, *tuple(reg_coef_MSE), *tuple(logreg_coef_MSE), H_cc_MSE, logH_cc_MSE))
    np.savez(join(savedir,"Hess_cmp_%03d.npz"%idx), **{"eva_PS":eigvals_PS, "evc_PS":eigvects_PS, "H_PS":H_PS,
                      "eva_SSIM":eigvals_SSIM, "evc_SSIM":eigvects_SSIM, "H_SSIM":H_SSIM,
                      "eva_MSE":eigvals_MSE, "evc_MSE":eigvects_MSE, "H_MSE":H_MSE,})

np.savez(join(savedir, "H_cmp_stat.npz"), MSE_stat=MSE_stat_col, SSIM_stat=SSIM_stat_col)
MSE_stat_tab = pd.DataFrame(MSE_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
MSE_stat_tab.to_csv(join(savedir, "H_cmp_MSE_stat.csv"))
SSIM_stat_tab = pd.DataFrame(SSIM_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
SSIM_stat_tab.to_csv(join(savedir, "H_cmp_SSIM_stat.csv"))
#%%
modelsnm = "Face256"
SGAN = loadStyleGAN2("ffhq-256-config-e-003810.pt", size=256,)
G = StyleGAN2_wrapper(SGAN, )
savedir = join(saveroot, "ImDist_cmp\\StyleGAN2\\Face256")
os.makedirs(savedir, exist_ok=True)
SSIM_stat_col = []
MSE_stat_col = []
for idx in range(100):
    refvec = G.sample_vector(1, device="cuda")# 0.1 * torch.randn(1, 256)
    eigvals_PS, eigvects_PS, H_PS = hessian_compute(G, refvec, ImDist, hessian_method="BP")
    eigvals_SSIM, eigvects_SSIM, H_SSIM = hessian_compute(G, refvec, D, hessian_method="BP")
    eigvals_MSE, eigvects_MSE, H_MSE = hessian_compute(G, refvec, MSE, hessian_method="BP")
    #% eigvals_L1, eigvects_L1, H_L1 = hessian_compute(G, refvec, L1, hessian_method="BP")
    print("SSIM - LPIPS comparison")
    cc_SSIM, logcc_SSIM, reg_coef_SSIM, logreg_coef_SSIM = spectra_cmp(-eigvals_SSIM[::-1], eigvals_PS, show=True)
    H_cc_SSIM, logH_cc_SSIM = Hessian_cmp(-eigvals_SSIM[::-1], eigvects_SSIM, -H_SSIM, eigvals_PS, eigvects_PS, H_PS, show=True)
    SSIM_stat_col.append((idx, cc_SSIM, logcc_SSIM, *tuple(reg_coef_SSIM), *tuple(logreg_coef_SSIM), H_cc_SSIM, logH_cc_SSIM))
    print("MSE - LPIPS comparison")
    cc_MSE, logcc_MSE, reg_coef_MSE, logreg_coef_MSE = spectra_cmp(eigvals_MSE, eigvals_PS, show=True)
    H_cc_MSE, logH_cc_MSE = Hessian_cmp(eigvals_MSE, eigvects_MSE, H_MSE, eigvals_PS, eigvects_PS, H_PS, show=True)
    MSE_stat_col.append((idx, cc_MSE, logcc_MSE, *tuple(reg_coef_MSE), *tuple(logreg_coef_MSE), H_cc_MSE, logH_cc_MSE))
    np.savez(join(savedir,"Hess_cmp_%03d.npz"%idx), **{"eva_PS":eigvals_PS, "evc_PS":eigvects_PS, "H_PS":H_PS,
                      "eva_SSIM":eigvals_SSIM, "evc_SSIM":eigvects_SSIM, "H_SSIM":H_SSIM,
                      "eva_MSE":eigvals_MSE, "evc_MSE":eigvects_MSE, "H_MSE":H_MSE,})

np.savez(join(savedir, "H_cmp_stat.npz"), MSE_stat=MSE_stat_col, SSIM_stat=SSIM_stat_col)
MSE_stat_tab = pd.DataFrame(MSE_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
MSE_stat_tab.to_csv(join(savedir, "H_cmp_MSE_stat.csv"))
SSIM_stat_tab = pd.DataFrame(SSIM_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp", "H_cc", "logH_cc"])
SSIM_stat_tab.to_csv(join(savedir, "H_cmp_SSIM_stat.csv"))

#%%
cc = np.corrcoef((-eigvals), (eigvals_PS[::-1]))[0, 1]
logcc = np.corrcoef(np.log10(-eigvals), np.log10(eigvals_PS[::-1]))[0, 1]
logreg_coef = np.polyfit(np.log10(-eigvals), np.log10(eigvals_PS[::-1]),1)
reg_coef = np.polyfit((-eigvals), (eigvals_PS[::-1]), 1)
#%% functions to compare the Hessian computed by different distance function at same location.
#%%
im1 = G.visualize(torch.randn(1,256).cuda())
im2 = G.visualize(torch.randn(1,256).cuda())
D(im1, im2)
refvec = 0.1 * torch.randn(1,256)
eigvals, eigvects, H = hessian_compute(G, refvec, D, hessian_method="BP")
#%%
#%%
from glob import glob

# savedir = join(saveroot, "ImDist_cmp\\BigGAN")
savedir = join(saveroot, "ImDist_cmp\\PGGAN")
savedir = join(saveroot, "ImDist_cmp\\StyleGAN2\\Face256")

fullfns = glob(join(savedir, "Hess_cmp_*.npz"))
npzpatt = "Hess_cmp_%03d.npz"
SSIM_stat_col = []
MSE_stat_col = []
for idx, fn in enumerate(fullfns):
    assert (npzpatt % idx) in fn
    data = np.load(fn)
    eigvals_PS, eigvects_PS, H_PS = data["eva_PS"], data["evc_PS"], data["H_PS"]
    eigvals_SSIM, eigvects_SSIM, H_SSIM = data["eva_SSIM"], data["evc_SSIM"], data["H_SSIM"]
    eigvals_MSE, eigvects_MSE, H_MSE = data["eva_MSE"], data["evc_MSE"], data["H_MSE"]

    print("SSIM - LPIPS comparison")
    cc_SSIM, logcc_SSIM, reg_coef_SSIM, logreg_coef_SSIM = spectra_cmp(-eigvals_SSIM[::-1], eigvals_PS, show=True)
    H_cc_SSIM, logH_cc_SSIM = Hessian_cmp(-eigvals_SSIM[::-1], eigvects_SSIM[:, ::-1], -H_SSIM,
                                        eigvals_PS, eigvects_PS, H_PS, show=True)
    xcc_SSIM2PS, xlogcc_SSIM2PS, xcc_PS2SSIM, xlogcc_PS2SSIM = eigvec_H_corr(-eigvals_SSIM[::-1], eigvects_SSIM[:, ::-1], -H_SSIM,
                                        eigvals_PS, eigvects_PS, H_PS, show=True)
    evc_cc_SSIM = top_eigvec_corr(eigvects_SSIM[:, ::-1], eigvects_PS, eignum=10)
    SSIM_stat_col.append((idx, cc_SSIM, logcc_SSIM, *tuple(reg_coef_SSIM), *tuple(logreg_coef_SSIM),
                          H_cc_SSIM, logH_cc_SSIM, xcc_SSIM2PS, xlogcc_SSIM2PS, xcc_PS2SSIM, xlogcc_PS2SSIM, *tuple(evc_cc_SSIM)))

    print("MSE - LPIPS comparison")
    cc_MSE, logcc_MSE, reg_coef_MSE, logreg_coef_MSE = spectra_cmp(eigvals_MSE, eigvals_PS, show=True)
    H_cc_MSE, logH_cc_MSE = Hessian_cmp(eigvals_MSE, eigvects_MSE, H_MSE, eigvals_PS, eigvects_PS, H_PS, show=True)
    xcc_MSE2PS, xlogcc_MSE2PS, xcc_PS2MSE, xlogcc_PS2MSE,= eigvec_H_corr(eigvals_MSE, eigvects_MSE, H_MSE, eigvals_PS, eigvects_PS, H_PS, show=True)
    evc_cc_MSE = top_eigvec_corr(eigvects_MSE, eigvects_PS, eignum=10)
    MSE_stat_col.append((idx, cc_MSE, logcc_MSE, *tuple(reg_coef_MSE), *tuple(logreg_coef_MSE), H_cc_MSE,
                         logH_cc_MSE, xcc_MSE2PS, xlogcc_MSE2PS, xcc_PS2MSE, xlogcc_PS2MSE, *tuple(evc_cc_MSE)))

np.savez(join(savedir, "H_cmp_stat.npz"), MSE_stat=MSE_stat_col, SSIM_stat=SSIM_stat_col)
SSIM_stat_tab = pd.DataFrame(SSIM_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp",
                                       "H_cc", "logH_cc", "xcc_SSIM2PS", "xlogcc_SSIM2PS", "xcc_PS2SSIM", "xlogcc_PS2SSIM"] + ["eigvec%d_cc" % i for i in range(10)])
SSIM_stat_tab.to_csv(join(savedir, "H_cmp_SSIM_stat.csv"))

MSE_stat_tab = pd.DataFrame(MSE_stat_col, columns=["id", "cc", "logcc", "reg_slop", "reg_intcp", "reg_log_slop", "reg_log_intcp",
                                       "H_cc", "logH_cc", "xcc_MSE2PS", "xlogcc_MSE2PS", "xcc_PS2MSE", "xlogcc_PS2MSE"] + ["eigvec%d_cc" % i for i in range(10)])
MSE_stat_tab.to_csv(join(savedir, "H_cmp_MSE_stat.csv"))
#%% Final Synopsis across GANs.
GANdir_col = ["ImDist_cmp\\BigGAN", "ImDist_cmp\\PGGAN", "ImDist_cmp\\StyleGAN2\\Face256"]
GANnms = ["BigGAN", "PGGAN", "StyleGAN2"]
Stats_nms = ["H_cc", "logH_cc", "cc", "logcc", "xcc_MSE2PS", "xlogcc_MSE2PS", "reg_log_slop", "reg_log_intcp"]
for GANnm, GANdir in zip(GANnms, GANdir_col):
    print("\n%s MSE - PS" % GANnm, end="\t")
    savedir = join(saveroot, GANdir)
    tab = pd.read_csv(join(savedir, "H_cmp_MSE_stat.csv"))
    meanstd_tab = pd.concat([tab.mean(axis=0), tab.std(axis=0)], 1).rename(columns={0: "mean", 1: "std"})
    keystats = meanstd_tab.loc[
        ["H_cc", "logH_cc", "cc", "logcc", "xcc_MSE2PS", "xlogcc_MSE2PS", "reg_log_slop", "reg_log_intcp"]]
    # print(keystats)
    for varnm, row in keystats.iterrows():
        print("%.2f(%.2f)" % (row[0], row[1]), end="\t")
    print("\n%s SSIM - PS" % GANnm, end="\t")
    tab = pd.read_csv(join(savedir, "H_cmp_SSIM_stat.csv"))
    meanstd_tab = pd.concat([tab.mean(axis=0), tab.std(axis=0)], 1).rename(columns={0:"mean",1:"std"})
    keystats = meanstd_tab.loc[
        ["H_cc", "logH_cc", "cc", "logcc", "xcc_SSIM2PS", "xlogcc_SSIM2PS", "reg_log_slop", "reg_log_intcp"]]
    for varnm, row in keystats.iterrows():
        print("%.2f(%.2f)" % (row[0], row[1]), end="\t")

