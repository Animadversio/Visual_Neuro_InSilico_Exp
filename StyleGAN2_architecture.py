import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings, get_full_hessian
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
import torchvision.models as tv
from GAN_utils import loadStyleGAN, StyleGAN_wrapper, ckpt_root
g_ema = loadStyleGAN("stylegan2-cat-config-f.pt")
G = StyleGAN_wrapper(g_ema)
#%%
paramnum = 0
for name, param in g_ema.state_dict().items():
    paramnum += np.prod(param.shape)
for name, param in g_ema.state_dict().items():
    print(name, list(param.shape), "%.1f%%"%(100*np.prod(param.shape) / paramnum))