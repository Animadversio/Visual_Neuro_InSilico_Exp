from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
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
    refnorm, tannorm = np.linalg.norm(refvect), np.linalg.norm(tangvect)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis] * np.pi / 2
    interp_vects = (np.sin(steps) @ tangvect / tannorm + np.cos(steps) @ refvect / refnorm) * refnorm
    return interp_vects
#%%
import sys
sys.path.append(r"D:\Github\BigGANsAreWatching")
sys.path.append(r"E:\Github_Projects\BigGANsAreWatching")
from BigGAN.gan_load import UnconditionalBigGAN, make_big_gan
from BigGAN.model.BigGAN import Generator
BBGAN = make_big_gan(r"E:\Github_Projects\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth", resolution=128)
for param in BBGAN.parameters():
    param.requires_grad_(False)
BBGAN.eval()
# the model is on cuda from this.
class BigBiGAN_wrapper():#nn.Module
    def __init__(self, BigBiGAN, ):
        self.BigGAN = BigBiGAN

    def visualize(self, code, scale=1.0, resolution=256):
        imgs = self.BigGAN(code, )
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = BigBiGAN_wrapper(BBGAN)
#%%
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
#%%
T00 = time()
for triali in range(20):
    for trunc in [0.1, 1, 3, 6, 9, 10, 12, 15]:
        if trunc == 0.1:
            continue
        RND = np.random.randint(1000)
        noisevect = torch.randn(1, 120)
        noisevect = noisevect / noisevect.norm()
        ref_vect = trunc * noisevect.detach().clone().cuda()
        mov_vect = ref_vect.detach().clone().requires_grad_(True)
        imgs1 = G.visualize(ref_vect)
        imgs2 = G.visualize(mov_vect)
        dsim = ImDist(imgs1, imgs2)
        H = get_full_hessian(dsim, mov_vect)  # 77sec to compute a Hessian.
        # ToPILImage()(imgs[0,:,:,:].cpu()).show()
        eigvals, eigvects = np.linalg.eigh(H)
        plt.figure(figsize=[7,5])
        plt.subplot(1, 2, 1)
        plt.plot(eigvals)
        plt.ylabel("eigenvalue")
        plt.subplot(1, 2, 2)
        plt.plot(np.log10(eigvals))
        plt.ylabel("eigenvalue (log)")
        plt.suptitle("Hessian Spectrum Full Space")
        plt.savefig(join(savedir, "Hessian_norm%d_%03d.jpg" % (trunc, RND)))
        np.savez(join(savedir, "Hess_norm%d_%03d.npz" % (trunc, RND)), H=H, eigvals=eigvals, eigvects=eigvects, vect=ref_vect.cpu().numpy(),)
        #%
        img_all = None
        for eigi in range(50): #eigvects.shape[1]
            interp_codes = LExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 15, (-2.5, 2.5))
            with torch.no_grad():
                img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda(), resolution=128).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            clear_output(wait=True)
            progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))
        imggrid = make_grid(img_all, nrow=15)
        PILimg = ToPILImage()(imggrid)#.show()
        PILimg.save(join(savedir, "eigvect_lin_norm%d_%03d.jpg" % (trunc, RND)))
        #%
        img_all = None
        for eigi in range(50): #eigvects.shape[1]
            interp_codes = SExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 21, (-1, 1))
            with torch.no_grad():
                img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda(), resolution=128).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            clear_output(wait=True)
            progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))
        imggrid = make_grid(img_all, nrow=21)
        PILimg = ToPILImage()(imggrid)#.show()
        PILimg.save(join(savedir, "eigvect_sph_norm%d_%03d.jpg" % (trunc, RND)))
        print("Spent time %.1f sec"%(time() - T00))