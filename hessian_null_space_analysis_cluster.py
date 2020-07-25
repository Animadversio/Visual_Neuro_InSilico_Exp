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
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
#%%
from FeatLinModel import FeatLinModel, get_model_layers
import sys
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"/home/binxu/PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch

# import torchvision as tv
# # VGG = tv.models.vgg16(pretrained=True)
# alexnet = tv.models.alexnet(pretrained=True).cuda()
# for param in alexnet.parameters():
#     param.requires_grad_(False)
hessian_method = "BP"  # "ForwardIter" "BackwardIter"
#%%
from scipy.io import loadmat
code_path = r"/home/binxu/pasu_fit_code.mat"
out_dir = r"/scratch/binxu/GAN_hessian"
data = loadmat(code_path)
pasu_codes = data['pasu_code']
#%%
t0 = time()
for imgi in range(pasu_codes.shape[0] - 1, 0, -1):
    code = pasu_codes[imgi, :]
    feat = torch.from_numpy(code[np.newaxis, :])
    feat.requires_grad_(False)
    if hessian_method == "BackwardIter":
        metricHVP = GANHVPOperator(G, feat, model_squ)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True)
    elif hessian_method == "BackwardIter":
        metricHVP = GANForwardMetricHVPOperator(G, feat, model_squ, preprocess=lambda img:img)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True)
    elif hessian_method == "BP":
        ref_vect = feat.detach().clone().cuda()
        mov_vect = ref_vect.detach().clone().requires_grad_(True)
        imgs1 = G.visualize(ref_vect)
        imgs2 = G.visualize(mov_vect)
        dsim = model_squ(imgs1, imgs2)
        H = get_full_hessian(dsim, mov_vect)  # 122 sec for a 256d hessian
        eigvals, eigvects = np.linalg.eigh(H)
    print("Finish computing img %d %.2f sec passed, max %.2e min %.2e 10th %.1e 50th %.e 100th %.1e" % (imgi,
    time() - t0, max(np.abs(eigvals)), min(np.abs(eigvals)), eigvals[-10], eigvals[-50], eigvals[-100]))
    np.savez(join(out_dir, "pasu_%03d.npz" % imgi), eigvals=eigvals, eigvects=eigvects, code=code)
