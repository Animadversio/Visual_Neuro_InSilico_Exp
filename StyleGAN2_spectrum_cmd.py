import sys
from os.path import join
from time import time
import torch
import numpy as np
sys.path.append("E:\Github_Projects\Visual_Neuro_InSilico_Exp")
import lpips
# ImDist = lpips.LPIPS(net="squeeze").cuda()
ImDist = lpips.PerceptualLoss(net="squeeze").cuda()
from GAN_hessian_compute import hessian_compute
from GAN_utils import loadStyleGAN, StyleGAN_wrapper
SGAN = loadStyleGAN("stylegan2-cat-config-f.pt", size=256, channel_multiplier=2)
G = StyleGAN_wrapper(SGAN)
#%%
# for triali in range(1,16):
#     feat = 0.5 * torch.randn(1, 512).detach().clone().cuda()
#     T0 = time()
#     eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
#     print("%.2f sec" % (time() - T0))  # 2135.00 sec
#     H_col = []
#     for EPS in [1E-4, 3E-4, 1E-3, 3E-3, 1E-2, 5E-2, 1E-1]:
#         T0 = time()
#         eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=EPS)
#         print("%.2f sec" % (time() - T0))  # 325.83 sec
#         print("EPS %.1e Correlation of Flattened Hessian matrix BP vs ForwardIter %.3f" % (EPS, np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]))
#         H_col.append((eva_FI, evc_FI, H_FI))
#     T0 = time()
#     eva_BI, evc_BI, H_BI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
#     print("Correlation of Flattened Hessian matrix BP vs BackwardIter %.3f" % (np.corrcoef(H_BP.flatten(), H_BI.flatten())[0, 1]))
#     print("%.2f sec" % (time() - T0))  # 2132.44 sec
#
#     np.savez(join("E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2", "Hess_cmp_%d.npz"%triali), eva_BI=eva_BI, evc_BI=evc_BI, H_BI=H_BI,
#                                         eva_FI=eva_FI, evc_FI=evc_FI, H_FI=H_FI, H_col=H_col,
#                                         eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())
#     print("Save finished")
#%%
savedir = r"E:\Cluster_Data\StyleGAN2"
for triali in range(150):
    feat = torch.randn(1, 512).detach().clone().cuda()
    T0 = time()
    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 2135.00 sec
    np.savez(join(savedir, "Hess_BP_%d.npz"%triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP, feat=feat.detach().cpu().numpy())