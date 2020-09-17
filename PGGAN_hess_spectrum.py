import torch
import numpy as np
from tqdm import tqdm
from time import time
import sys
from os.path import join
import lpips
from GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
ImDist = lpips.LPIPS(net='squeeze').cuda()
use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=use_gpu)
num_images = 1
noise, _ = model.buildNoiseData(num_images)
noise.requires_grad_(True)
# with torch.no_grad():
generated_images = model.test(noise)
#%%
img = model.avgG.forward(noise)
#%%
class PGGAN_wrapper():  # nn.Module
    def __init__(self, PGGAN, ):
        self.PGGAN = PGGAN

    def visualize(self, code, scale=1):
        imgs = self.PGGAN.forward(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale
G = PGGAN_wrapper(model.avgG)

#%%
feat = noise.detach().clone().cuda()
EPS = 1E-2
T0 = time()
eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
print("%.2f sec" % (time() - T0))  # 95.7 sec
T0 = time()
eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter")
print("%.2f sec" % (time() - T0))  # 61.8 sec
T0 = time()
eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
print("%.2f sec" % (time() - T0))  # 95.4 sec
#%%
print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1])
print("Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" %
      np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1])
print("Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter %.3f"%
      np.corrcoef(H_FI.flatten(), H_BI.flatten())[0, 1])

# Correlation of Flattened Hessian matrix BP vs BackwardIter 1.000
# Correlation of Flattened Hessian matrix BP vs ForwardIter 0.877
# Correlation of Flattened Hessian matrix ForwardIter vs BackwardIter 0.877
#%%
H_col = []
for EPS in [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 2, 10]:
    T0 = time()
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
    print("%.2f sec" % (time() - T0))  # 325.83 sec
    print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
    H_col.append((eva_FI, evc_FI, H_FI))
# EPS 1.0e-05 Correlation of Flattened Hessian matrix BP vs ForwardIter 1.000
# EPS 1.0e-04 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.999
# EPS 1.0e-03 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.989
# EPS 1.0e-02 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.901
# EPS 1.0e-01 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.398
# EPS 1.0e+00 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.046
# EPS 2.0e+00 Correlation of Flattened Hessian matrix BP vs ForwardIter 0.008
# EPS 1.0e+01 Correlation of Flattened Hessian matrix BP vs ForwardIter -0.003
#%%
#%% Visualize Spectra
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\PGGAN"
savedir = r"E:\Cluster_Backup\PGGAN"
eva_col = []
evc_col = []
for triali in tqdm(range(400)):
    data = np.load(join(savedir, "Hessian_cmp_%d.npz" % triali))
    eva_BP = data["eva_BP"]
    evc_BP = data["evc_BP"]
    eva_col.append(eva_BP)
    evc_col.append(evc_BP)

eva_col = np.array(eva_col)
#%%
from hessian_analysis_tools import plot_spectra, compute_hess_corr, plot_consistency_example, plot_consistentcy_mat
fig = plot_spectra(eva_col, figdir=figdir, titstr="PGGAN", )
#%%
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=True)
# without cuda 12:11 mins, with cuda 8:21
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False)

#%%
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, posN=300, figdir=figdir, titstr="PGGAN")
#%%
fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="PGGAN",)
fig3.show()
