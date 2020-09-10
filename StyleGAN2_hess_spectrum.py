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
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
#%%
StyleGAN_root = r"E:\DL_Projects\Vision\stylegan2-pytorch"
StyleGAN_root = r"D:\Github\stylegan2-pytorch"
sys.path.append(StyleGAN_root)
from model import Generator
def loadStyleGAN(ckpt_name):
    ckpt_path = join(StyleGAN_root, "checkpoint", ckpt_name)
    size = 256
    device = "cpu"
    latent = 512
    n_mlp = 8
    channel_multiplier = 2
    g_ema = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    checkpoint = torch.load(ckpt_path)
    g_ema.load_state_dict(checkpoint['g_ema'])
    g_ema.eval()
    for param in g_ema.parameters():
        param.requires_grad_(False)
    g_ema.cuda()
    return g_ema

#%%
ckpt_name = r"stylegan2-cat-config-f.pt"# r"stylegan2-ffhq-config-f.pt"#
# r"AbstractArtFreaGAN.pt"#r"2020-01-11-skylion-stylegan2-animeportraits.pt"
ckpt_path = join(StyleGAN_root, "checkpoint", ckpt_name)
size = 256
device = "cpu"
latent = 512
n_mlp = 8
channel_multiplier = 2
g_ema = Generator(
    size, latent, n_mlp, channel_multiplier=channel_multiplier
).to(device)
checkpoint = torch.load(ckpt_path)
g_ema.load_state_dict(checkpoint['g_ema'])
g_ema.eval()
for param in g_ema.parameters():
    param.requires_grad_(False)
g_ema.cuda()
#%%
class StyleGAN_wrapper():#nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN
        self.truncation = None
        self.mean_latent = None

    def select_trunc(self, truncation, mean_latent):
        self.truncation = truncation
        self.mean_latent = mean_latent

    def visualize(self, code, scale=1.0, resolution=256, truncation=1, mean_latent=None, preset=True):
        if preset and self.truncation is not None:
            imgs, _ = self.StyleGAN([code], truncation=self.truncation, truncation_latent=self.mean_latent)
        else:
            imgs, _ = self.StyleGAN([code], truncation=truncation, truncation_latent=mean_latent)
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation, mean_latent, B=5):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                       truncation=truncation, mean_latent=mean_latent, preset=False).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
            clear_output(wait=True)
            progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

g_ema = loadStyleGAN("stylegan2-cat-config-f.pt")
G = StyleGAN_wrapper(g_ema)
#%% Forward Operator Factorization
from GAN_hvp_operator import GANForwardMetricHVPOperator
#%%
truncation = 0.8
truncation_mean = 4096
mean_latent = g_ema.mean_latent(truncation_mean)
G.select_trunc(truncation, mean_latent)
savedir = r"E:\Cluster_Backup\StyleGAN2\Cats_forw"
for triali in range(10):
    for HVP_eps in [1E-1, 5E-2, 2E-2, 1E-2, 5E-3, 2E-3]:
        RND = np.random.randint(10000)
        T0 = time()
        ref_z = torch.randn(1, latent, device="cuda").cuda()
        SGhvp = GANForwardMetricHVPOperator(G, ref_z, ImDist, preprocess=lambda img: img, EPS=HVP_eps,)
        eigenvals, eigenvecs = lanczos(SGhvp, num_eigenthings=250, max_steps=200, tol=1e-5,)
        print(time() - T0, " sec")# 10 eigvect takes 12 sec
        # 50 eigvect takes 40 sec 40.1 sec
        # 200 eigvect, 100 steps takes 163 sec
        #%
        eigenvecs = eigenvecs.T
        sort_idx = np.argsort(np.abs(eigenvals))
        eigabs_sort = eigenvals[sort_idx]
        eigvect_sort = eigenvecs[:, sort_idx]
        #%
        np.savez(join(savedir, "Hess_trunc%.1f_eps%.E_%03d.npz" % (truncation, HVP_eps, RND)), eigvals=eigenvals,
                 eigvects=eigenvecs, vect=ref_z.cpu().numpy(), )
        # plt.figure();plt.plot(np.log10(eigenvals));plt.xlim([0, len(eigenvals)])
        # plt.savefig(join(savedir, "Hessian_trunc%.1f_EPS%.E_%03d.jpg" % (truncation, HVP_eps, RND)))
        plt.figure(figsize=[7, 5])
        plt.subplot(1, 2, 1)
        plt.plot(eigenvals[::-1])
        plt.plot(np.abs(eigabs_sort[::-1]))
        plt.xlim([0, len(eigenvals)])
        plt.ylabel("eigenvalue")
        plt.subplot(1, 2, 2)
        plt.plot(np.log10(eigenvals[::-1]))
        plt.plot(np.log10(np.abs(eigabs_sort[::-1])))
        plt.xlim([0, len(eigenvals)])
        plt.ylabel("eigenvalue (log)")
        plt.suptitle("Hessian Spectrum Forward Decomposition")
        plt.savefig(join(savedir, "Hessian_trunc%.1f_EPS%.E_%03d.jpg" % (truncation, HVP_eps, RND)))
    #%%
        T00 = time()
        eigstr = 0
        while eigstr < eigenvecs.shape[1]:
            eigend = min(eigstr + 50, eigenvecs.shape[1])
            codes_all = []
            for eigi in range(eigstr, eigend):  # eigvects.shape[1]
                interp_codes = SExpMap(ref_z.cpu().numpy(), eigvect_sort[:, -eigi-1], 15, (-0.5, 0.5))
                codes_all.append(interp_codes.copy())
            codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
            img_all = G.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=8)
            imggrid = make_grid(img_all, nrow=15)
            PILimg2 = ToPILImage()(imggrid)#.show()
            PILimg2.save(join(savedir, "eigvect_sph_fin_eig%d-%d_trunc%.1f_EPS%.E_%04d.jpg" % (eigstr, eigend,
                                                                                 truncation, HVP_eps, RND)))
            print("Spent time %.1f sec" % (time() - T00))
            eigstr = eigend
    #%%
        T00 = time()
        codes_all = []
        for eigi in range(50):  # eigvects.shape[1]
            interp_codes = LExpMap(ref_z.cpu().numpy(), eigvect_sort[:, -eigi-1], 11, (-10, 10))
            codes_all.append(interp_codes.copy())
        codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
        img_all = G.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=8)
        imggrid = make_grid(img_all, nrow=11)
        PILimg = ToPILImage()(imggrid)  # .show()
        PILimg.save(join(savedir, "eigvect_lin_trunc%.1f_EPS%.E_%04d.jpg" % (truncation, HVP_eps, RND)))
        print("Spent time %.1f sec" % (time() - T00))





#%%
T00 = time()
codes_all = []
for eigi in range(50):  # eigvects.shape[1]
    interp_codes = SExpMap(ref_z.cpu().numpy(), eigenvecs[:, -eigi-1], 15, (-0.5, 0.5))
    codes_all.append(interp_codes.copy())
codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
img_all = G.visualize_batch_np(codes_all_arr, truncation=1, mean_latent=None, B=5)
imggrid = make_grid(img_all, nrow=15)
PILimg2 = ToPILImage()(imggrid)#.show()
PILimg2.save(join(savedir, "eigvect_sph_fin_trunc%.1f_%04d.jpg" % (1, RND)))
print("Spent time %.1f sec" % (time() - T00))

#%%
T00 = time()
truncation = 0.8
truncation_mean = 4096
RND = np.random.randint(10000)
mean_latent = g_ema.mean_latent(truncation_mean)
ref_z = torch.randn(1, latent, device=device).cuda()
mov_z = ref_z.detach().clone().requires_grad_(True) # requires grad doesn't work for 1024 images.
ref_samp = G.visualize(ref_z, truncation=truncation, mean_latent=mean_latent)
mov_samp = G.visualize(mov_z, truncation=truncation, mean_latent=mean_latent)
dsim = ImDist(ref_samp, mov_samp)
H = get_full_hessian(dsim, mov_z)
print(time() - T00, " sec")

#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\StyleGAN2"
truncation = 0.5
T00 = time()
for triali in range(3):
    for truncation in [1, 0.8, 0.6]:
        if truncation == 1 and triali == 0:
            continue
        T00 = time()
        truncation_mean = 4096
        RND = np.random.randint(1000)
        mean_latent = g_ema.mean_latent(truncation_mean)
        ref_z = torch.randn(1, latent, device=device).cuda()
        mov_z = ref_z.detach().clone().requires_grad_(True)
        ref_samp = G.visualize(ref_z, truncation=truncation, mean_latent=mean_latent)
        mov_samp = G.visualize(mov_z, truncation=truncation, mean_latent=mean_latent)
        dsim = ImDist(ref_samp, mov_samp)
        H = get_full_hessian(dsim, mov_z)
        print("Computing Hessian Completed, %.1f sec" % (time()-T00))
#%%
        eigvals, eigvects = np.linalg.eigh(H)
        plt.figure(figsize=[7,5])
        plt.subplot(1, 2, 1)
        plt.plot(eigvals)
        plt.ylabel("eigenvalue")
        plt.subplot(1, 2, 2)
        plt.plot(np.log10(eigvals))
        plt.ylabel("eigenvalue (log)")
        plt.suptitle("Hessian Spectrum Full Space")
        plt.savefig(join(savedir, "Hessian_trunc%.1f_%03d.jpg" % (truncation, RND)))
        np.savez(join(savedir, "Hess_trunc%.1f_%03d.npz" % (truncation, RND)), H=H, eigvals=eigvals, eigvects=eigvects, vect=ref_z.cpu().numpy(),)
        del dsim
        torch.cuda.empty_cache()
#%%
        T00 = time()
        codes_all = []
        for eigi in range(50): # eigvects.shape[1]
            interp_codes = LExpMap(ref_z.cpu().numpy(), eigvects[:, -eigi-1], 11, (-10, 10))
            codes_all.append(interp_codes.copy())
        codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
        img_all = G.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=5)
        imggrid = make_grid(img_all, nrow=11)
        PILimg = ToPILImage()(imggrid)  # .show()
        PILimg.save(join(savedir, "eigvect_lin_trunc%.1f_%03d.jpg" % (truncation, RND)))
        print("Spent time %.1f sec" % (time() - T00))
        #%%
        T00 = time()
        codes_all = []
        for eigi in range(50):  # eigvects.shape[1]
            interp_codes = SExpMap(ref_z.cpu().numpy(), eigvects[:, -eigi-1], 11, (-1, 1))
            codes_all.append(interp_codes.copy())
        codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
        img_all = G.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=5)
        imggrid = make_grid(img_all, nrow=11)
        PILimg2 = ToPILImage()(imggrid)#.show()
        PILimg2.save(join(savedir, "eigvect_sph_trunc%.1f_%03d.jpg" % (truncation, RND)))
        print("Spent time %.1f sec" % (time() - T00))
        #%%
        T00 = time()
        codes_all = []
        for eigi in range(50):  # eigvects.shape[1]
            interp_codes = SExpMap(ref_z.cpu().numpy(), eigvects[:, -eigi-1], 15, (-0.5, 0.5))
            codes_all.append(interp_codes.copy())
        codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
        img_all = G.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=5)
        imggrid = make_grid(img_all, nrow=15)
        PILimg2 = ToPILImage()(imggrid)#.show()
        PILimg2.save(join(savedir, "eigvect_sph_fin_trunc%.1f_%03d.jpg" % (truncation, RND)))
        print("Spent time %.1f sec" % (time() - T00))
#%%
