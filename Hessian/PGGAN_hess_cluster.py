import torch
import numpy as np
from tqdm import tqdm
from time import time
import sys
from os.path import join
import lpips
from Hessian.GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
ImDist = lpips.LPIPS(net='squeeze').cuda()
use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=use_gpu)

class PGGAN_wrapper():  # nn.Module
    def __init__(self, PGGAN, ):
        self.PGGAN = PGGAN

    def visualize(self, code, scale=1):
        imgs = self.PGGAN.forward(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale
G = PGGAN_wrapper(model.avgG)
#%%
from argparse import ArgumentParser
parser = ArgumentParser(description='Computing Hessian at different part of the code space in PG GAN')
parser.add_argument('--dataset', type=str, default="rand", help='dataset name `pasu` or `evol`, `text`')
parser.add_argument('--method', type=str, default="BP", help='Method of computing Hessian can be `BP` or `ForwardIter` `BackwardIter` ')
parser.add_argument('--idx_rg', type=int, default=[0, 200], nargs="+", help='range of index of vectors to use')
parser.add_argument('--EPS', type=float, default=1E-4, help='EPS of finite differencing HVP operator, will only be ')
args = parser.parse_args()
if len(args.idx_rg) == 2:
    id_str, id_end = args.idx_rg[0], args.idx_rg[1]
else:
    id_str, id_end = 0, 200
    print("doing it all! ")

#%%
# figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\DCGAN"
savedir = r"/scratch/binxu/GAN_hessian"
for triali in tqdm(range(id_str, id_end)):
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
