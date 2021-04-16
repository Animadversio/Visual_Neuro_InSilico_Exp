""" This analysis tries to prove the prediction that there have to be some bad dots on the manifold

"""
#%%
from GAN_utils import BigGAN_wrapper, loadBigGAN, StyleGAN2_wrapper, loadStyleGAN2, loadPGGAN, PGGAN_wrapper
from load_hessian_data import load_Haverage
import matplotlib.pylab as plt
import torch, numpy as np
from os.path import join
import os
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pylab as plt
from torch_utils import show_imgrid, save_imgrid

saveroot = r"E:\OneDrive - Washington University in St. Louis\GAN_baddots"
#%% Progressive Growing GAN
modelsnm = "PGGAN"
savedir = join(saveroot, modelsnm)
os.makedirs(savedir, exist_ok=True)
PGAN = loadPGGAN()
G = PGGAN_wrapper(PGAN)
H, eva, evc = load_Haverage("PGGAN", descend=True)
randvec = G.sample_vector(10, 'cpu')
distnorm = randvec.norm(dim=1).mean().item()
# distnorm = np.sqrt(randvec.shape[1])
#%
baddot_imgs_pos = G.visualize_batch_np(distnorm * evc[:, :20].T)
show_imgrid(baddot_imgs_pos, nrow=5, padding=2,)
save_imgrid(baddot_imgs_pos, join(savedir, "GAN_baddots_pos.png"), nrow=5, padding=2,)
baddot_imgs_neg = G.visualize_batch_np(-distnorm * evc[:, :20].T)
show_imgrid(baddot_imgs_neg, nrow=5, padding=2,)
save_imgrid(baddot_imgs_neg, join(savedir, "GAN_baddots_neg.png"), nrow=5, padding=2,)
#%%
modelsnm = "StyleGAN2-Face512_Z"
savedir = join(saveroot, modelsnm)
os.makedirs(savedir, exist_ok=True)
SGAN = loadStyleGAN2("ffhq-512-avg-tpurun1.pt")
G = StyleGAN2_wrapper(SGAN)
H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
randvec = G.sample_vector(10, 'cpu')
distnorm = randvec.norm(dim=1).mean().item()
# distnorm = np.sqrt(randvec.shape[1])
#%%
baddot_imgs_pos = G.visualize_batch_np(distnorm * evc[:, :20].T)
show_imgrid(baddot_imgs_pos, nrow=5, padding=2,)
save_imgrid(baddot_imgs_pos, join(savedir, "GAN_space_baddots_pos.png"), nrow=5, padding=2,)
baddot_imgs_neg = G.visualize_batch_np(-distnorm * evc[:, :20].T)
show_imgrid(baddot_imgs_neg, nrow=5, padding=2,)
save_imgrid(baddot_imgs_neg, join(savedir, "GAN_space_baddots_neg.png"), nrow=5, padding=2,)
#%%
badregion_vects = 5*np.random.randn(20, 20)  @ evc[:, :20].T + np.random.randn(20, 492)  @ evc[:, 20:].T
baddot_imgs_pos = G.visualize_batch_np(badregion_vects)
show_imgrid(baddot_imgs_pos, nrow=5, padding=2,)
# save_imgrid(baddot_imgs_pos, join(savedir, "GAN_space_baddots_pos.png"), nrow=5, padding=2,)
baddot_imgs_neg = G.visualize_batch_np(- badregion_vects)
show_imgrid(baddot_imgs_neg, nrow=5, padding=2,)
# save_imgrid(baddot_imgs_neg, join(savedir, "GAN_space_baddots_neg.png"), nrow=5, padding=2,)

#%%
from scipy.stats import truncnorm, norm
topeigDist = truncnorm(2, np.inf)
elseDist = norm(0, 1)
weights = np.concatenate((topeigDist.rvs((10, 3)), elseDist.rvs((10, 509))), axis=1)
badregion_vects = weights @ evc.T
baddot_imgs_pos = G.visualize_batch_np(badregion_vects)
show_imgrid(baddot_imgs_pos, nrow=5, padding=2,)
baddot_imgs_neg = G.visualize_batch_np(- badregion_vects)
show_imgrid(baddot_imgs_neg, nrow=5, padding=2,)
