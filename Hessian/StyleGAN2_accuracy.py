"""Analyze the accuracy of different Hessian computation method comparing to each other"""

#%%
import sys
from tqdm import tqdm
import os
from os.path import join
from time import time
import torch
import numpy as np
import matplotlib.pylab as plt
from GAN_utils import loadBigBiGAN, loadStyleGAN2, BigBiGAN_wrapper, StyleGAN2_wrapper, loadBigGAN, BigGAN_wrapper
# sys.path.append(r"/home/binxu/PerceptualSimilarity")
# sys.path.append(r"D:\Github\PerceptualSimilarity")
# sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
# import models
# ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
import lpips
ImDist = lpips.LPIPS(net="squeeze").cuda()
from Hessian.GAN_hessian_compute import hessian_compute
#%%
SGAN = loadStyleGAN2("ffhq-512-avg-tpurun1.pt", size=512)
G = StyleGAN2_wrapper(SGAN)
#%% Accuracy dependency on the EPS value
feat = 0.5 * torch.randn(1, 512).detach().clone().cuda()
triali = 0
T0 = time()
eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
print("%.2f sec" % (time() - T0))  # 2135.00 sec
H_col = []
for EPS in [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, ]:
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
    print("%.2f sec" % (time() - T0))  # 325.83 sec
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
    H_col.append((eva_FI, evc_FI, H_FI))
T0 = time()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % (np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]))
print("%.2f sec" % (time() - T0))  # 2132.44 sec

np.savez("Hess_cmp_%d.npz"%triali, eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                    eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
                                    eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
print("Save finished")
#%% Compute the Hessian with 3 different methods and different EPS
for triali in range(6):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 2135.00 sec
    H_col = []
    for EPS in [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2]:
        T0 = time()
        eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
        print("%.2f sec" % (time() - T0))  # 325.83 sec
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
        H_col.append((eva_FI, evc_FI, H_FI))
    T0 = time()
    eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % (np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]))
    print("%.2f sec" % (time() - T0))  # 2132.44 sec
    np.savez("Hess_accuracy_cmp_%d.npz" % triali, eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                        eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
                                        eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
    print("Save finished")
#%% Visualize the accuracy
EPS_list = [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2]
raw_corr_tab = []
PSD_corr_tab = []
for triali in range(6):
    print("Computation trial %d"%triali)
    data = np.load("Hess_accuracy_cmp_%d.npz" % triali)
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
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"

plt.plot(PSD_corr_tab.T)
plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
plt.ylabel("Correlation for Vectorized Hessian")
plt.xlabel("EPS for Forward Diff")
plt.title("StyleGAN2 BP vs ForwardIter Pos-Semi-Definite Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN2_BP-FI-PSD-HessCorr.png"))
plt.show()

plt.plot(raw_corr_tab.T)
plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
plt.ylabel("Correlation for Vectorized Hessian")
plt.xlabel("EPS for Forward Diff")
plt.title("StyleGAN2 BP vs ForwardIter Raw Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN2_BP-FI-raw-HessCorr.png"))
plt.show()
#%%
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
plt.title("StyleGAN2 BP vs ForwardIter Hessian Correlation")
plt.savefig(join(figdir, "StyleGAN2_BP-FI-HessCorr-cmp.png"))
plt.show()


# for triali in range(5):
#     EPS = 1E-2
#     T0 = time()
#     eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
#     print("%.2f sec" % (time() - T0))  # 2132.44 sec
#     T0 = time()
#     eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter")
#     print("%.2f sec" % (time() - T0))  # 325.83 sec
#     T0 = time()
#     eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#     print("%.2f sec" % (time() - T0))  # 2135.00 sec
#     np.savez("Hess_cmp_%d.npz"%triali, eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
#                                     eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI,
#                                     eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#     print("Save finished")
#%% Newer version of this with more trials (all relavent data in it.)
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
EPS_list = [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]

raw_corr_tab = []  # correlation with BP
PSD_corr_tab = []  # Positive spectrum version
raw_corr_BI_FI_tab = []  # correlation with BI of different FI method
PSD_corr_BI_FI_tab = []  # Positive spectrum version
for triali in range(16):
    data = np.load(join(savedir, "Hess_cmp_%d.npz" % triali), allow_pickle=True)
    H_col = data["H_col"]
    eva_BP, evc_BP, H_BP = data["eva_BP"], data["evc_BP"], data["H_BP"]
    eva_BI, evc_BI, H_BI = data["eva_BI"], data["evc_BI"], data["H_BI"]
    H_BI_PSD = evc_BI @ np.diag(np.abs(eva_BI)) @ evc_BI.T
    corr_vals = [np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]]  # the first column is the correlation with BI
    PSD_corr_vals = [np.corrcoef(H_BP.flatten(), H_BI_PSD.flatten())[0, 1]]  # the first column is the correlation with BI
    corr_vals_BI_FI = []
    PSD_corr_vals_BI_FI = []
    for EPSi, EPS in enumerate(EPS_list):
        eva_FI, evc_FI, H_FI = H_col[EPSi, :]
        H_PSD = evc_FI @ np.diag(np.abs(eva_FI)) @ evc_FI.T
        corr_vals.append(np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
        PSD_corr_vals.append(np.corrcoef(H_BP.flatten(), H_PSD.flatten())[0, 1])
        corr_vals_BI_FI.append(np.corrcoef(H_BI.flatten(), H_FI.flatten())[0, 1])
        PSD_corr_vals_BI_FI.append(np.corrcoef(H_BI_PSD.flatten(), H_PSD.flatten())[0, 1])
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
            EPS, corr_vals[-1]))
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter (AbsHess) %.3f" % (
            EPS, PSD_corr_vals[-1]))
    raw_corr_tab.append(corr_vals)
    PSD_corr_tab.append(PSD_corr_vals)
    raw_corr_BI_FI_tab.append(corr_vals_BI_FI)
    PSD_corr_BI_FI_tab.append(PSD_corr_vals_BI_FI)
raw_corr_tab = np.array(raw_corr_tab)
PSD_corr_tab = np.array(PSD_corr_tab)
raw_corr_BIFI = np.array(raw_corr_BI_FI_tab)
PSD_corr_BIFI = np.array(PSD_corr_BI_FI_tab)
raw_corr_FI, raw_corr_BI = raw_corr_tab[:, 1:], raw_corr_tab[:, :1]
PSD_corr_FI, PSD_corr_BI = PSD_corr_tab[:, 1:], PSD_corr_tab[:, :1]
np.savez(join(figdir, "Hess_method_acc_data.npy"), raw_corr_FI=raw_corr_FI, raw_corr_BI=raw_corr_BI,
         PSD_corr_FI=PSD_corr_FI, raw_corr_BIFI=raw_corr_BIFI, PSD_corr_BIFI=PSD_corr_BIFI)
#%%
def hess_accuracy_curve(raw_corr_tab, PSD_corr_tab, EPS_list, figdir=figdir, raw_corr_BI=None, PSD_corr_BI=None,
                        savestr="StyleGAN2"):
    fig1 = plt.figure()
    plt.plot(PSD_corr_tab.T)
    plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
    plt.ylabel("Correlation of Vectorized Hessian")
    plt.xlabel("EPS for Forward Diff")
    plt.title("StyleGAN2 BP vs ForwardIter Pos-Semi-Definite Hessian Correlation")
    plt.savefig(join(figdir, "%s_BP-FI-PSD-HessCorr.png"%savestr))
    plt.show()
    fig2 = plt.figure()
    plt.plot(raw_corr_tab.T)
    plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
    plt.ylabel("Correlation of Vectorized Hessian")
    plt.xlabel("EPS for Forward Diff")
    plt.title("StyleGAN2 BP vs ForwardIter Raw Hessian Correlation")
    plt.savefig(join(figdir, "%s_BP-FI-raw-HessCorr.png"%savestr))
    plt.show()
    fig3 = plt.figure()
    men = raw_corr_tab.mean(axis=0)
    err = raw_corr_tab.std(axis=0)/np.sqrt(raw_corr_tab.shape[0])
    plt.plot(men, )
    plt.fill_between(range(len(men)), men-err, men+err, alpha=0.3, label="FI raw-BP")
    men = PSD_corr_tab.mean(axis=0)
    err = PSD_corr_tab.std(axis=0)/np.sqrt(PSD_corr_tab.shape[0])
    plt.plot(men, )
    plt.fill_between(range(len(men)), men-err, men+err, alpha=0.3, label="FI PSD-BP")
    if raw_corr_BI is not None:
        men_corr_BI, std_corr_BI = raw_corr_BI.mean(), raw_corr_BI.std()
        plt.hlines(men_corr_BI, 0, len(men)-1, colors="red")# ], )
        plt.fill_between(range(len(men)), men_corr_BI - std_corr_BI, men_corr_BI + std_corr_BI, alpha=0.3,
                         label="FI raw-BI")
        men_corr_BI, std_corr_BI = PSD_corr_BI.mean(), PSD_corr_BI.std()
        plt.hlines(men_corr_BI, 0, len(men)-1, colors="blue") #plt.plot([0, len(men)], men_corr_BI)
        plt.fill_between(range(len(men)), men_corr_BI - std_corr_BI, men_corr_BI + std_corr_BI, alpha=0.3,
                         label="FI PSD-BI")

    plt.xticks(np.arange(len(EPS_list)), labels=EPS_list)
    plt.legend()
    plt.ylabel("Correlation of Vectorized Hessian")
    plt.xlabel("EPS for Forward Diff")
    plt.title("StyleGAN2 BP vs ForwardIter Hessian Correlation")
    plt.savefig(join(figdir, "%s_BP-FI-HessCorr-cmp.png"%savestr))
    plt.savefig(join(figdir, "%s_BP-FI-HessCorr-cmp.pdf"%savestr))
    plt.show()
    return fig1, fig2, fig3
#%%

hess_accuracy_curve(raw_corr_FI, PSD_corr_FI, EPS_list, savestr="StyleGAN2")
hess_accuracy_curve(raw_corr_FI, PSD_corr_FI, EPS_list, raw_corr_BI=raw_corr_BI, PSD_corr_BI=PSD_corr_BI,
                    savestr="StyleGAN2_all")

#%%
SGAN = loadStyleGAN2("ffhq-256-config-e-003810.pt")
G = StyleGAN2_wrapper(SGAN)

#%%
G.random = False
tmp = torch.randn(1,512).cuda()
img1 = G.visualize(tmp)
img2 = G.visualize(tmp)
print((img2-img1).cpu().abs().max())
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
os.makedirs(join(figdir, "accuracy_fix"), exist_ok=True)
EPS_list = [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]
for triali in tqdm(range(7)):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 107 sec
    H_col = []
    for EPS in EPS_list:
        T0 = time()
        eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
        print("%.2f sec" % (time() - T0))  # 79 sec
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
        H_col.append((eva_FI, evc_FI, H_FI))
    T0 = time()
    eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % (np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]))
    print("%.2f sec" % (time() - T0))  # 104.7 sec
    np.savez("Hess_accuracy_cmp_%d.npz" % triali, eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                        eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
                                        eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
    print("Save finished")
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
EPS_list = [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]

raw_corr_tab = []  # correlation with BP
PSD_corr_tab = []  # Positive spectrum version
raw_corr_BI_FI_tab = []  # correlation with BI of different FI method
PSD_corr_BI_FI_tab = []  # Positive spectrum version
for triali in range(7):
    data = np.load(join("Hess_accuracy_cmp_%d.npz" % triali), allow_pickle=True)
    H_col = data["H_col"]
    eva_BP, evc_BP, H_BP = data["eva_BP"], data["evc_BP"], data["H_BP"]
    eva_BI, evc_BI, H_BI = data["eva_BI"], data["evc_BI"], data["H_BI"]
    H_BI_PSD = evc_BI @ np.diag(np.abs(eva_BI)) @ evc_BI.T
    corr_vals = [np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]]  # the first column is the correlation with BI
    PSD_corr_vals = [np.corrcoef(H_BP.flatten(), H_BI_PSD.flatten())[0, 1]]  # the first column is the correlation with BI
    corr_vals_BI_FI = []
    PSD_corr_vals_BI_FI = []
    for EPSi, EPS in enumerate(EPS_list):
        eva_FI, evc_FI, H_FI = H_col[EPSi, :]
        H_PSD = evc_FI @ np.diag(np.abs(eva_FI)) @ evc_FI.T
        corr_vals.append(np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
        PSD_corr_vals.append(np.corrcoef(H_BP.flatten(), H_PSD.flatten())[0, 1])
        corr_vals_BI_FI.append(np.corrcoef(H_BI.flatten(), H_FI.flatten())[0, 1])
        PSD_corr_vals_BI_FI.append(np.corrcoef(H_BI_PSD.flatten(), H_PSD.flatten())[0, 1])
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (
            EPS, corr_vals[-1]))
        print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter (AbsHess) %.3f" % (
            EPS, PSD_corr_vals[-1]))
    raw_corr_tab.append(corr_vals)
    PSD_corr_tab.append(PSD_corr_vals)
    raw_corr_BI_FI_tab.append(corr_vals_BI_FI)
    PSD_corr_BI_FI_tab.append(PSD_corr_vals_BI_FI)
raw_corr_tab = np.array(raw_corr_tab)
PSD_corr_tab = np.array(PSD_corr_tab)
raw_corr_BIFI = np.array(raw_corr_BI_FI_tab)
PSD_corr_BIFI = np.array(PSD_corr_BI_FI_tab)
raw_corr_FI, raw_corr_BI = raw_corr_tab[:, 1:], raw_corr_tab[:, :1]
PSD_corr_FI, PSD_corr_BI = PSD_corr_tab[:, 1:], PSD_corr_tab[:, :1]
np.savez(join(figdir, "Hess_method_acc_data_SGfix.npz"), raw_corr_FI=raw_corr_FI, raw_corr_BI=raw_corr_BI,
         PSD_corr_FI=PSD_corr_FI, raw_corr_BIFI=raw_corr_BIFI, PSD_corr_BIFI=PSD_corr_BIFI)

#%%
hess_accuracy_curve(raw_corr_FI, PSD_corr_FI, EPS_list, savestr="StyleGAN2_fix")
hess_accuracy_curve(raw_corr_FI, PSD_corr_FI, EPS_list, raw_corr_BI=raw_corr_BI, PSD_corr_BI=PSD_corr_BI,
                    savestr="StyleGAN2_fix_all")