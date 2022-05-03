#%%
import torch
import torchvision.models as models
from lpips import LPIPS
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from easydict import EasyDict
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
import torch
from torch_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from geometry_utils import LERP, SLERP, orthogonalize, SLERP_torch, LERP_torch
figdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Rebuttal\SphereGeometry"
#%%
vec1 = torch.randn(1, 4096) * 3
vec2 = torch.randn(1, 4096) * 3
vec2 = vec2 - (vec2 @ vec1.T) / (vec1 @ vec1.T)
vec2 = vec2 / torch.norm(vec2) * torch.norm(vec1)
#%%
interp_vecs = SLERP_torch(vec1, vec2, 21, lim=(0, 2))
imgs = G.visualize(interp_vecs.cuda())
distmat = Dist.forward_distmat(imgs).squeeze()
# show_imgrid(imgs.cpu(), nrow=21)
#%
save_imgrid(imgs.cpu()[::3, :, :, :], join(figdir, "spherepert_imgs.png"))
plt.imshow(distmat.cpu().detach().numpy(), cmap="jet")
plt.title("Angular equi-angle perturbation")
plt.yticks(range(0,21,5))
plt.colorbar()
plt.savefig(join(figdir, "ang_pert_distmat.png"))
plt.savefig(join(figdir, "ang_pert_distmat.pdf"))
plt.show()
#%%
interp_vecs_lin = LERP_torch(vec1, vec2, 21)
imgs_lin = G.visualize(interp_vecs_lin.cuda())
distmat_lin = Dist.forward_distmat(imgs_lin).squeeze()
# show_imgrid(imgs_lin.cpu(), nrow=21)
#%
save_imgrid(imgs_lin.cpu()[::3, :, :, :], join(figdir, "lin_intp_imgs.png"))
plt.imshow(distmat_lin.cpu().detach().numpy(), cmap="jet")
plt.title("Linear equidistant interpolation")
plt.yticks(range(0,21,5))
plt.colorbar()
plt.savefig(join(figdir, "lin_intp_distmat.png"))
plt.savefig(join(figdir, "lin_intp_distmat.pdf"))
plt.show()
#%%
interp_vecs_linpert = vec1 + torch.linspace(0, 2, 21).unsqueeze(1) @ vec2 #LERP_torch(vec1, vec2, 21)
imgs_lin = G.visualize(interp_vecs_linpert.cuda())
# show_imgrid(imgs_lin.cpu(), nrow=21)
distmat_lin = Dist.forward_distmat(imgs_lin).squeeze()
save_imgrid(imgs_lin.cpu()[::3, :, :, :], join(figdir, "linpert_imgs.png"))
plt.imshow(distmat_lin.cpu().detach().numpy(), cmap="jet")
plt.colorbar()
plt.yticks(range(0,21,5))
plt.title("Linear equi-distance perturbation")
plt.savefig(join(figdir, "lin_pert_distmat.png"))
plt.savefig(join(figdir, "lin_pert_distmat.pdf"))
plt.show()
#%%
vecs1_scale = torch.linspace(0, 2, 7).view(-1,1) @ vec1
vecs2_scale = torch.linspace(0, 2, 7).view(-1,1) @ vec2
save_imgrid(G.visualize(vecs1_scale.cuda()), join(figdir, "vecs1_scale.png"))
save_imgrid(G.visualize(vecs2_scale.cuda()), join(figdir, "vecs2_scale.png"))
