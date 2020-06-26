import torch
import torchvision as tv
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, compute_hessian_eigenthings
import lucent
from lucent.optvis import render

#%% Prepare the Networks
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda() # this notation is incorrect in older pytorch


VGG = tv.models.vgg16(pretrained=True)
layernames = lucent_layernames(VGG)
#%%
feat = torch.randn((4096), dtype=torch.float32).requires_grad_(False).cuda()
GHVP = GANHVPOperator(G, feat, model_squ)
GHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())

#%% 

from lucent.optvis import render, param, transform, objectives

net = 
obj_f 
activHVP = GANHVPOperator(G, feat, obj_f)


