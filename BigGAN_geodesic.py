import cma
import tqdm
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import os
import sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from os.path import join
from imageio import imread
from scipy.linalg import block_diag
#%%
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
for param in ImDist.parameters():
    param.requires_grad_(False)
def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean(axis=1)

#%%
def image_distmat(imgs, ImDist):
    """Compute image distance matrix of a batch of images using Perceptual Similarity """
    imgn = imgs.shape[0]
    distmat = np.zeros((imgn, imgn))
    with torch.no_grad():
        for i in tqdm.trange(imgn):
            dsims = ImDist(imgs[i,:], imgs)
            dsims_np = dsims.squeeze().cpu().detach().numpy()
            distmat[:, i] = dsims_np
    return distmat

def torch_distmat(merge_vecs):
    innprod = merge_vecs @ merge_vecs.T
    vecnorm = (merge_vecs ** 2).sum(dim=1, keepdim=True)
    vecdist = vecnorm + vecnorm.T - 2*innprod
    return vecdist

#%%
BGAN = BigGAN.from_pretrained("biggan-deep-256")
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
#%%
if sys.platform == "linux":
    # BGAN = get_BigGAN()
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    Hpath = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    imgfolder = r"/scratch/binxu/Datasets/ImageTranslation/GAN_real/B/train"
    savedir = r"/scratch/binxu/GAN_geodesic/BigGAN"
else:
    # BGAN = BigGAN.from_pretrained("biggan-deep-256")
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    Hpath = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    imgfolder = r"E:\Cluster_Backup\Datasets\ImageTranslation\GAN_real\B\train"
    savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_geodesic"

data = np.load(Hpath)
evc_clas = torch.from_numpy(data['eigvects_clas_avg'])#.cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg'])#.cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()
evc_sep = torch.from_numpy(block_diag(data['eigvects_nois_avg'], data['eigvects_clas_avg'])).cuda()
evc_none = torch.eye(256).cuda()
#%%
cls_raw1 = torch.zeros([1,1000]).cuda();cls_raw1[0,459] = 1
class_vec1 = BGAN.embeddings(cls_raw1)
cls_raw2 = torch.zeros([1,1000]).cuda();cls_raw2[0,689] = 1
class_vec2 = BGAN.embeddings(cls_raw2)
#%%
EmbedMat = BGAN.embeddings.weight
class_vec1 = EmbedMat[:, 373:374].T
# class_vec2 = EmbedMat[:, 689:690].T
class_vec2 = EmbedMat[:, 501:502].T
#%
noise_vec = torch.from_numpy(truncated_noise_sample(2, 128)).cuda()
noise_vec1 = noise_vec[:1, :]
noise_vec2 = noise_vec[1:, :]
latentvec1 = torch.cat((noise_vec1, class_vec1), 1)
latentvec2 = torch.cat((noise_vec2, class_vec2), 1)
#%
with torch.no_grad():
    img1 = BGAN.generator(torch.cat((latentvec1, latentvec2)), 0.7)
    img1 = (img1 + 1) / 2
#%
ToPILImage()(make_grid(img1).cpu()).show()
#%%
#%%
ticks = torch.linspace(0, 1, 16).cuda().unsqueeze(1)
latentvecs = ticks @ latentvec1 + (1 - ticks) @ latentvec2
with torch.no_grad():
    imgs = BGAN.generator(latentvecs, 0.7).cpu()
imgs = (imgs + 1) / 2
ToPILImage()(make_grid(imgs)).show()
#%%
distmat = image_distmat(imgs, ImDist)
plt.figure()
plt.matshow(distmat, fignum=0)
plt.title("Image Distance")
plt.colorbar()
plt.show()
#%%
vec_distmat = torch_distmat(latentvecs)
plt.figure()
plt.matshow(vec_distmat.cpu(), fignum=0)
plt.title("Vector Distance")
plt.colorbar()
plt.show()
#%%
coef1 = latentvec1 @ evc_all
coef2 = latentvec2 @ evc_all
coef2 - coef1
#%%
XX, YY = torch.meshgrid(torch.arange(-1, 257), torch.arange(256),)
merge_mask = (XX >= YY).cuda()
#%
merge_coef = merge_mask * coef1 + (~ merge_mask) * coef2
merge_vecs = merge_coef @ evc_all.T
#%%
B = 30
with torch.no_grad():
    csr = 0
    img_traj = None
    while csr < merge_vecs.shape[0]:
        csr_end = min(csr + B, merge_vecs.shape[0])
        img_batch = BGAN.generator(merge_vecs[csr:csr_end, :], 0.7).cpu()
        img_batch = (img_batch + 1.0) / 2
        img_traj = img_batch if img_traj is None else torch.cat((img_traj, img_batch))
        csr = csr_end
#%%
ToPILImage()(make_grid(img_traj, nrow=16).cpu()).show()
#%%
vecdist = torch_distmat(merge_vecs)
#%
plt.figure()
plt.matshow(vecdist.cpu(), fignum=0)
plt.title("Vector Distance")
plt.colorbar()
plt.show()
#%%
distmat_traj = image_distmat(img_traj[::5,:], ImDist)
plt.figure()
plt.matshow(distmat_traj, fignum=0)
plt.title("Image Distance")
plt.colorbar()
plt.show()
#%%
