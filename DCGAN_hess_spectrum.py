import torch
import numpy as np
from tqdm import tqdm
from time import time
from os.path import join
import sys
import lpips
from GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
#%%
use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
ImDist = lpips.LPIPS(net='squeeze').cuda()

class DCGAN_wrapper():  # nn.Module
    def __init__(self, DCGAN, ):
        self.DCGAN = DCGAN

    def visualize(self, code, scale=1.0):
        imgs = self.DCGAN(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = DCGAN_wrapper(model.avgG)
#%%
noise, _ = model.buildNoiseData(1)
feat = noise.detach().clone().cuda()
T0 = time()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
print("%.2f sec" % (time() - T0))  # 13.40 sec
T0 = time()
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter")
print("%.2f sec" % (time() - T0))  # 6.89 sec
T0 = time()
eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
print("%.2f sec" % (time() - T0))  # 12.5 sec
print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1])
print("Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" %
      np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
print("Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter %.3f"%
      np.corrcoef(H_FI.flatten(), H_BI.flatten())[0, 1])
H_col = []
for EPS in [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, ]:
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
    print("%.2f sec" % (time() - T0))  # 325.83 sec
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
    H_col.append((eva_FI, evc_FI, H_FI))

# Correlation of Flattened Hessian matrix BP vs BackwardIter 1.000
# Correlation of Flattened Hessian matrix BP vs ForwardIter 0.845
# Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter 0.845
# EPS 1.0e-06 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.978
# EPS 1.0e-05 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.978
# EPS 1.0e-04 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.978
# EPS 1.0e-03 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.958
# EPS 1.0e-02 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.858
# EPS 1.0e-01 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.199
#%%
savedir = r"E:\Cluster_Data\DCGAN"
np.savez(join(savedir, "Hessian_EPS_cmp.npz"), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                        eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI,
                                        eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\DCGAN"
savedir = r"E:\Cluster_Data\DCGAN"
for triali in tqdm(range(168, 300)):
    noise, _ = model.buildNoiseData(1)
    feat = noise.detach().clone().cuda()
    T0 = time()
    eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("%.2f sec" % (time() - T0))  # 13.40 sec
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=1E-4)
    print("%.2f sec" % (time() - T0))  # 6.89 sec
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 12.5 sec
    print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1])
    print("Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" %
          np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
    print("Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter %.3f"%
          np.corrcoef(H_FI.flatten(), H_BI.flatten())[0, 1])
    np.savez(join(savedir, "Hessian_cmp_%d.npz" % triali), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
                                        eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI,
                                        eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
    print("Save finished")
#%% Visualize Spectra
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\DCGAN"
savedir = r"E:\Cluster_Data\DCGAN"
eva_col = []
evc_col = []
for triali in tqdm(range(300)):
    data = np.load(join(savedir, "Hessian_cmp_%d.npz" % triali))
    eva_BP = data["eva_BP"]
    evc_BP = data["evc_BP"]
    eva_col.append(eva_BP)
    evc_col.append(evc_BP)

eva_col = np.array(eva_col)
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\DCGAN"
savedir = r"E:\Cluster_Backup\DCGAN"
from hessian_analysis_tools import plot_spectra, compute_hess_corr, plot_consistency_example, plot_consistentcy_mat, \
    plot_consistency_hist, average_H, scan_hess_npz
eva_col, evc_col, feat_col, meta = scan_hess_npz(savedir, "Hessian_cmp_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col)
np.savez(join(figdir, "H_avg_%s.npz"%"DCGAN"), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
#%%
fig = plot_spectra(eva_col, figdir=figdir, titstr="DCGAN", )
#%%
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False)
# without cuda 2:12 mins, with cuda 6:55
#%%
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="DCGAN")
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="DCGAN")
#%%
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="DCGAN",)
