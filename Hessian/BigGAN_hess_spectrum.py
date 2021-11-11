"""This script is dedicated to compute the Hessian Spectrum of
Different Generative models and at different place
Much inspired by BigGAN Hessian demo code. But here doesn't care the neuron response.
"""
#%%
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from Hessian.lanczos_generalized import lanczos_generalized
from Hessian.GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings, get_full_hessian
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
import torchvision.models as tv
#%%
from geometry_utils import LERP, SLERP, ExpMap
from PIL import Image
from skimage.io import imsave
from torchvision.utils import make_grid
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
def LExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis]
    interp_vects = steps @ tangvect + refvect
    return interp_vects

def SExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis] * np.pi / 2
    interp_vects = steps @ tangvect + refvect
    return interp_vects
#%%
BGAN = BigGAN.from_pretrained("biggan-deep-256")
for param in BGAN.parameters():
    param.requires_grad_(False)
embed_mat = BGAN.embeddings.parameters().__next__().data
BGAN.cuda()
# the model is on cuda from this.

#%% import BigBiGAN! 
import sys
sys.path.append(r"E:\Github_Projects\BigGANsAreWatching")
from BigGAN.gan_load import UnconditionalBigGAN, make_big_gan
from BigGAN.model.BigGAN import Generator
BBGAN = make_big_gan(r"E:\Github_Projects\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth", resolution=128)

#%% 
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
#%%
from torchvision.transforms import Normalize, Compose
RGB_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
RGB_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()
preprocess = Compose([lambda img: (F.interpolate(img, (224, 224), mode='bilinear', align_corners=True) - RGB_mean) / RGB_std])
preprocess_resize = Compose([lambda img: F.interpolate(img, (224, 224), mode='bilinear', align_corners=True) ])
#%%
import torch.nn as nn
from GAN_utils import BigGAN_wrapper
# class BigGAN_wrapper():#nn.Module
#     def __init__(self, BigGAN, space="class"):
#         self.BigGAN = BigGAN
#         self.space = space
#
#     def visualize(self, code, scale=1.0):
#         imgs = self.BigGAN.generator(code, 0.6) # Matlab version default to 0.7
#         return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = BigGAN_wrapper(BGAN)
# H = get_full_hessian()
#%%
savedir = r"E:\iclr2021\Results"
savedir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp"
#%%
T00 = time()
for class_id in [17, 79, 95, 107, 224, 346, 493, 542, 579, 637, 667, 754, 761, 805, 814, 847, 856, 941, 954, 968]:
    classvec = embed_mat[:, class_id:class_id+1].cuda().T
    noisevec = torch.from_numpy(truncated_noise_sample(1, 128, 0.6)).cuda()
    ref_vect = torch.cat((noisevec, classvec, ), dim=1).detach().clone()
    mov_vect = ref_vect.detach().clone().requires_grad_(True)
    #%%
    imgs1 = G.visualize(ref_vect)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H = get_full_hessian(dsim, mov_vect)  # 77sec to compute a Hessian.
    # ToPILImage()(imgs[0,:,:,:].cpu()).show()
    eigvals, eigvects = np.linalg.eigh(H)  # 75 ms
    #%%
    noisevec.requires_grad_(True)
    classvec.requires_grad_(False)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_nois = get_full_hessian(dsim, noisevec)  # 39.3 sec to compute a Hessian.
    eigvals_nois, eigvects_nois = np.linalg.eigh(H_nois)  # 75 ms
    #%
    noisevec.requires_grad_(False)
    classvec.requires_grad_(True)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_clas = get_full_hessian(dsim, classvec)  # 77sec to compute a Hessian.
    eigvals_clas, eigvects_clas = np.linalg.eigh(H_clas)  # 75 ms
    classvec.requires_grad_(False)

    np.savez(join(savedir, "Hess_cls%d.npz" % class_id), H=H, H_nois=H_nois, H_clas=H_clas, eigvals=eigvals,
             eigvects=eigvects, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, eigvals_nois=eigvals_nois,
             eigvects_nois=eigvects_nois, vect=ref_vect.cpu().numpy(),
             noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())
    #%%
    plt.figure(figsize=[7,5])
    plt.subplot(1, 2, 1)
    plt.plot(eigvals)
    plt.ylabel("eigenvalue")
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(eigvals))
    plt.ylabel("eigenvalue (log)")
    plt.suptitle("Hessian Spectrum Full Space")
    plt.savefig(join(savedir, "Hessian_cls%d.jpg"%class_id))
    #%
    plt.figure(figsize=[7,5])
    plt.subplot(1, 2, 1)
    plt.plot(eigvals_nois, label="noise")
    plt.plot(eigvals_clas, label="class")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(eigvals_nois), label="noise")
    plt.plot(np.log10(eigvals_clas), label="class")
    plt.ylabel("eigenvalue (log)")
    plt.legend()
    plt.suptitle("Hessian Spectrum Full Space")
    plt.tight_layout(pad=0.6)
    plt.savefig(join(savedir, "Hessian_sep_cls%d.jpg"%class_id))
    # plt.show()
    print("Spent %.1f sec from start" % (time() - T00))

    #%% Interpolation in the full space
    img_all = None
    for eigi in range(50): #eigvects.shape[1]
        interp_codes = LExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 11, (-2.5, 2.5))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))

    imggrid = make_grid(img_all, nrow=11)
    PILimg = ToPILImage()(imggrid)#.show()
    PILimg.save(join(savedir, "eigvect_full_cls%d.jpg"%class_id))
    #% Interpolation in the class space
    img_all = None
    for eigi in range(50): # eigvects_clas.shape[1]
        interp_class = LExpMap(classvec.cpu().numpy(), eigvects_clas[:, -eigi-1], 11, (-2.5, 2.5))
        interp_codes = np.hstack((noisevec.cpu().numpy().repeat(11, axis=0), interp_class, ))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 128))

    imggrid = make_grid(img_all, nrow=11)
    PILimg2 = ToPILImage()(imggrid)#.show()
    PILimg2.save(join(savedir, "eigvect_clas_cls%d.jpg"%class_id))
    #% Interpolation in the noise space
    img_all = None
    for eigi in range(50):#eigvects_nois.shape[1]
        interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-4.5, 4.5))
        interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 128))

    imggrid = make_grid(img_all, nrow=11)
    PILimg3 = ToPILImage()(imggrid)#.show()
    PILimg3.save(join(savedir, "eigvect_nois_cls%d.jpg"%class_id))
    print("Spent %.1f sec from start" % (time() - T00))
#%%
# go through spectrum in batch, and plot B number of axis in a row
def vis_eigen_frame(eigvect_avg, eigv_avg, ref_code=None, figdir=figdir, page_B=50):
    if ref_code is None:
        ref_code = np.zeros((1, 4096))
    t0=time()
    csr = 0
    img_page = []
    for eigi in range(1, 4097):
        interp_codes = LERP(ref_code, eigvect_avg[:, -eigi], 11, (-200, 200))
        img_list = G.render(interp_codes)
        img_page.extend(img_list)
        if (eigi == csr + page_B) or eigi == 4096:
            mtg = build_montages(img_page, (256, 256), (11, page_B))[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%d-%d_%.e~%.e.jpg" %
                        (csr+1, eigi, eigv_avg[-csr-1], eigv_avg[-eigi])), np.uint8(mtg * 255.0))
            img_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(csr, eigi, time()-t0))
            csr = eigi