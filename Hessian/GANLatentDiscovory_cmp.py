#%%
import sys
sys.path.append(r"E:\Github_Projects\Visual_Neuro_InSilico_Exp")
from os.path import join
import os
from time import time
import torch
import numpy as np
from numpy.linalg import norm
from torch_utils import show_imgrid, save_imgrid
from GAN_hessian_compute import hessian_compute
import torch.nn.functional as F
sys.path.append(r"E:\DL_Projects\Vision\GANLatentDiscovery")
sys.path.append(r"D:\DL_Projects\Vision\GANLatentDiscovery")
from loading import load_from_dir, load_human_annotation
#%% PGGAN!
savedir = r"E:\OneDrive - Washington University in St. Louis\GAN_prev_comp\UnsuperLatentDiscov\PGGAN_celebA"
summarydir = join(savedir, "summary")
os.makedirs(summarydir, exist_ok=True)

deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/ProgGAN/',
    G_weights='./models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth')
annot_dict = load_human_annotation('./models/pretrained/deformators/ProgGAN/human_annotation.txt')

deformator.requires_grad_(False)
G.requires_grad_(False)
shift_predictor.requires_grad_(False)
Wmat = deformator.linear.weight.detach().clone()
#%
prev_axes = Wmat[:,list(annot_dict.values())].T.cpu().numpy()
np.savez(join(savedir, "prev_axes.npz"), basis=Wmat.cpu().numpy(),
         annot_dict=annot_dict, prev_axes=prev_axes)
#%%
refvec = torch.randn(1, 512).cuda()
movvecs = refvec + torch.linspace(-9,9,5).cuda().unsqueeze(1)@Wmat[:,20:21].T
imgs = G((movvecs).unsqueeze(2).unsqueeze(3)) #
show_imgrid([torch.clamp((imgs+1)/2,0,1)])
#%%
from lpips import LPIPS
ImDist = LPIPS(net="squeeze").cuda()
ImDist.requires_grad_(False)
#%%
class PGGAN_wrapper2():  # nn.Module
    """
    model = loadPGGAN(onlyG=False)
    G = PGGAN_wrapper(model.avgG)

    model = loadPGGAN()
    G = PGGAN_wrapper(model)
    """
    def __init__(self, PGGAN, ):
        self.PGGAN = PGGAN

    def sample_vector(self, sampn=1, device="cuda"):
        refvec = torch.randn((sampn, 512)).to(device)
        return refvec

    def visualize(self, code, scale=1.0, res=1024):
        imgs = self.PGGAN.forward(code.unsqueeze(2).unsqueeze(3),)
        imgs = F.interpolate(imgs, size=(res, res))
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch(self, codes, scale=1.0, res=1024, device="cpu", B=5):
        img_all = None
        csr = 0
        imgn = codes.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            imgs = self.visualize(codes[csr:csr_end, :], scale=scale, res=res).to(device)
            img_all = imgs if img_all is None else torch.cat((img_all, imgs), 0)
            csr = csr_end
        return img_all
Gw = PGGAN_wrapper2(G)
#%%
T0 = time()
eigvals, eigvects, H = hessian_compute(Gw, refvec, ImDist, hessian_method="BackwardIter", cutoff=50,
                preprocess=lambda img: F.interpolate(img, size=(256, 256)), device="cuda") # EPS=1E-5,
print("%.3f sec" % (time() - T0)) # 77sec
#%%
T0 = time()
eigvals_BP, eigvects_BP, H_BP = hessian_compute(Gw, refvec, ImDist, hessian_method="BP",
                preprocess=lambda img: F.interpolate(img, size=(256, 256)), device="cuda") # EPS=1E-5,
print("%.3f sec" % (time() - T0)) # 377sec
#%%
movvecs = refvec + (torch.linspace(-9,9,5).unsqueeze(1)@torch.tensor(eigvects[:,3:4].T)).cuda()
imgs = Gw.visualize(movvecs, res=256)
show_imgrid(imgs)
#%%
np.savez(join(savedir,"H_BP_%03d.npz"%0), eva_BP=eigvals_BP, evc_BP=eigvects_BP, H_BP=H_BP, feat=refvec.cpu().numpy())
#%%
#%%
eigi = 510
movvecs2 = torch.randn(1,512).cuda() + (torch.linspace(-10,10,5).unsqueeze(1)@torch.tensor(eigvects_BP[:,eigi:eigi+1].T)).cuda()
movvecs3 = torch.randn(1,512).cuda() + (torch.linspace(-10,10,5).unsqueeze(1)@torch.tensor(eigvects_BP[:,eigi:eigi+1].T)).cuda()
imgs = Gw.visualize_batch(torch.cat((movvecs2, movvecs3)), res=256)
show_imgrid(imgs, nrow=5)
#%%
movvecs_pos = refvec + 8 * Wmat[:,list(annot_dict.values())].T
movvecs_neg = refvec - 8 * Wmat[:,list(annot_dict.values())].T
imgs_pos = Gw.visualize(movvecs_pos, res=256).cpu() #
imgs_neg = Gw.visualize(movvecs_neg, res=256).cpu() #
show_imgrid([imgs_pos, imgs_neg],nrow=6)
#%%

#%%
np.sum((prev_axes @ eigvects_BP) **2 * np.abs(eigvals_BP), axis=1)
#%%
randvec = np.random.randn(100,512)
randvec = randvec / norm(randvec, axis=1, keepdims=True) * 1.25
np.sum((randvec**2) * np.abs(eigvals_BP), axis=1)
#%%
"""In their paper they admitted that Progressive growing GAN is hard for their model to find interpretable axes
and random axes can achieve a similar subjective score as unsupervised found ones. """
#%%

#%%
torch.load("models\\pretrained\\deformators\\BigGAN\\models\\deformator_0.pt")
#%%
data = torch.load("models\\pretrained\\deformators\\ProgGAN\\models\\deformator_0.pt")
Wmat = data['linear.weight']
bmat = data['linear.bias']
#%%


#%%
import torch
from ortho_utils import torch_expm
data = torch.load("models\\pretrained\\deformators\\StyleGAN2\\models\\deformator_0.pt")
list(data)
logWhalf = data['log_mat_half']
Wmat = torch_expm((logWhalf - logWhalf.transpose(0, 1)).unsqueeze(0))
#%%
#%%

#%%
SGAN = loadStyleGAN2("stylegan2-ffhq-config-f.pt")
G = StyleGAN2_wrapper(SGAN)
img = G.visualize(torch.linspace(-5,5,4).cuda().unsqueeze(1)@Wmat[:,24:25].T)
#%%



