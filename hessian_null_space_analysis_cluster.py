import sys
sys.path.append(r"/home/binxu/PerceptualSimilarity")
sys.path.append(r"/home/binxu/Visual_Neuro_InSilico_Exp")
import torch
import torch.optim as optim
import torch.nn.functional as F
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardMetricHVPOperator, GANForwardHVPOperator, \
    compute_hessian_eigenthings, get_full_hessian

import numpy as np
import matplotlib.pylab as plt
from time import time
import os
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
#%%
from argparse import ArgumentParser
parser = ArgumentParser(description='Computing Hessian at different part of the code space in FC6 GAN')
parser.add_argument('--GAN', type=str, default="fc6", help='GAN model can be fc6, fc7, fc8, fc6_shuf')
parser.add_argument('--dataset', type=str, default="pasu", help='dataset name `pasu` or `evol`')
parser.add_argument('--method', type=str, default="BP", help='Method of computing Hessian can be `BP` or '
                                                             '`ForwardIter` `BackwardIter` ')
parser.add_argument('--idx_rg', type=int, default=[0, 50], nargs="+", help='range of index of vectors to use')
parser.add_argument('--EPS', type=float, default=1E-2, help='EPS of finite differencing HVP operator, will only be '
                                                            'used when method is `ForwardIter`')
args = parser.parse_args()  # ["--dataset", "pasu", '--method', "BP", '--idx_rg', '0', '50', '--EPS', '1E-2'])
#%%
from FeatLinModel import FeatLinModel, get_model_layers
import models # from PerceptualSimilarity
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
if args.GAN in ["fc6", "fc7", "fc8"]:
    G = upconvGAN(args.GAN)
elif args.GAN == "fc6_shfl":
    G = upconvGAN("fc6")
    SD = G.state_dict()
    shuffled_SD = {}
    for name, Weight in SD.items():
        idx = torch.randperm(Weight.numel())
        W_shuf = Weight.view(-1)[idx].view(Weight.shape)
        shuffled_SD[name] = W_shuf
    G.load_state_dict(shuffled_SD)
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
#%%
# import torchvision as tv
# # VGG = tv.models.vgg16(pretrained=True)
# alexnet = tv.models.alexnet(pretrained=True).cuda()
# for param in alexnet.parameters():
#     param.requires_grad_(False)

#%%
out_dir = r"/scratch/binxu/GAN_hessian/FC6GAN"
out_dir = r"/scratch/binxu/GAN_hessian/%sGAN"%(args.GAN)

from scipy.io import loadmat
if args.dataset == 'pasu':
    code_path = r"/home/binxu/pasu_fit_code.mat"
    data = loadmat(code_path)
    codes_all = data['pasu_code']
elif args.dataset == 'evol':
    code_path = r"/home/binxu/evol_codes_all.npz"
    data = np.load(code_path)
    codes_all = data["code_arr"]
#%%
hessian_method = args.method
labeldict = {"BP": "bpfull", "BackwardIter": "bkwlancz", "ForwardIter": "frwlancz"}
if len(args.idx_rg) == 2:
    id_str, id_end = args.idx_rg[0], args.idx_rg[1]
    id_end = min(id_end, codes_all.shape[0])
else:
    print("doing it all! ")
    id_str, id_end = 0, codes_all.shape[0]

t0 = time()
for imgi in range(id_str, id_end):#range(pasu_codes.shape[0] - 1, 0, -1):
    code = codes_all[imgi, :]
    feat = torch.from_numpy(code[np.newaxis, :])
    feat.requires_grad_(False)
    if hessian_method == "BackwardIter":
        metricHVP = GANHVPOperator(G, feat, model_squ)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True)  # takes 113 sec on K20x cluster,
        eigvects = eigvects.T  # note the output shape from lanczos is different from that of linalg.eigh, row is eigvec
        # the spectrum has a close correspondance with the full Hessian. since they use the same graph.
    elif hessian_method == "ForwardIter":
        metricHVP = GANForwardMetricHVPOperator(G, feat, model_squ, preprocess=lambda img:img, EPS=args.EPS) #1E-3,)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True, max_steps=200, tol=1e-6,)
        eigvects = eigvects.T
        # EPS=1E-2, max_steps=20 takes 84 sec on K20x cluster.
        # The hessian is not so close
    elif hessian_method == "BP":  # 240 sec on cluster
        ref_vect = feat.detach().clone().float().cuda()
        mov_vect = ref_vect.float().detach().clone().requires_grad_(True)
        imgs1 = G.visualize(ref_vect)
        imgs2 = G.visualize(mov_vect)
        dsim = model_squ(imgs1, imgs2)
        H = get_full_hessian(dsim, mov_vect)  # 122 sec for a 256d hessian, # 240 sec on cluster for 4096d hessian
        eigvals, eigvects = np.linalg.eigh(H)

    print("Finish computing img %d %.2f sec passed, max %.2e min %.2e 5th %.1e 10th %.1e 50th %.1e 100th %.1e 200th "
          "%.1e 400th %.1e" % (imgi, time() - t0, max(np.abs(eigvals)), min(np.abs(eigvals)), eigvals[-5],
          eigvals[-10], eigvals[-50], eigvals[-100],eigvals[-200], eigvals[-400]))
    np.savez(join(out_dir, "%s_%03d_%s.npz" % (args.dataset, imgi, labeldict[hessian_method])), eigvals=eigvals,
             eigvects=eigvects, code=code)
