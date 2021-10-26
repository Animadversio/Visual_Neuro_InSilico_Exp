""""""
from GAN_utils import loadStyleGAN2, loadBigGAN
from GAN_utils import StyleGAN2_wrapper, BigGAN_wrapper

import torch
import numpy as np
import matplotlib.pylab as plt
import os
from os.path import join
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
#%%
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
compressdir = r"E:\OneDrive - Washington University in St. Louis\GANcompress"
figdir = r"E:\OneDrive - Washington University in St. Louis\GANcompress\BigGAN"
os.makedirs(figdir, exist_ok=True)
os.makedirs(compressdir, exist_ok=True)
#%% BigGAN
BGAN = loadBigGAN()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
data = np.load(join(rootdir, 'BigGAN', "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
H_BG = data['H_avg']
#%%
cutoff = 100
samp_n = 10
classid = np.random.randint(0, 1000, samp_n)
refvec = np.vstack((0.7*np.random.randn(128,samp_n), EmbedMat[:, classid])).T
refvec_proj = refvec@evc_BG[:,-cutoff:]@evc_BG[:,-cutoff:].T
orig_img = BG.visualize_batch_np(refvec)
proj_img = BG.visualize_batch_np(refvec_proj)
mtg = make_grid(torch.cat((orig_img, proj_img)), nrow=samp_n)
ToPILImage()(mtg).show()
#%%
cutoff_list = 5, 10, 20, 40, 80, 120, 160, 200, 250
samp_n = 10
# if RND is None:
RND = np.random.randint(10000)
classid = np.random.randint(0, 1000, samp_n)
refvec = np.vstack((0.7*np.random.randn(128,samp_n), EmbedMat[:, classid])).T
codes_all = [refvec]
for cutoff in cutoff_list:
    refvec_proj = refvec@evc_BG[:,-cutoff:]@evc_BG[:,-cutoff:].T
    codes_all.append(refvec_proj)
codes_all = np.concatenate(tuple(codes_all), axis=0)
img_all = BG.visualize_batch_np(codes_all)
mtg = make_grid(img_all, nrow=samp_n)
ToPILImage()(mtg).show()
ctf_str = "_".join([str(ct) for ct in cutoff_list])
plt.save(join(figdir, "BigGAN_proj_%s_%04d.png"%(ctf_str, RND)), mtg.numpy(), )


#%% StyleGAN2
Hessdir = join(rootdir, 'StyleGAN2')
modellist = ["ffhq-512-avg-tpurun1",
			"stylegan2-cat-config-f",
			"2020-01-11-skylion-stylegan2-animeportraits"]
modelsnms = ["Face512",
			"Cat256",
			"Anime"]
#%% for modelnm, modelsnm in zip(modellist, modelsnms):
modelnm, modelsnm = modellist[1], modelsnms[1]
SGAN = loadStyleGAN2(modelnm+".pt", size=256)
SG = StyleGAN2_wrapper(SGAN)
figdir = join(compressdir, "StyleGAN2", modelsnm)
os.makedirs(figdir, exist_ok=True)
data = np.load(join(Hessdir, "H_avg_%s.npz"%modelnm))#, H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
H_avg = data["H_avg"]
eva_avg = data["eva_avg"]
evc_avg = data["evc_avg"]
#%%
def vis_eigen_cutoff(refvec, evc_avg, cutoff_list, G, figdir="", savestr="StyleGAN2_proj", RND=None):
    if RND is None: RND = np.random.randint(10000)
    samp_n = refvec.shape[0]
    codes_all = [refvec]
    for cutoff in cutoff_list:
        refvec_proj = refvec@evc_avg[:,-cutoff:]@evc_avg[:,-cutoff:].T
        codes_all.append(refvec_proj)
    codes_all = np.concatenate(tuple(codes_all), axis=0)
    img_all = G.visualize_batch_np(codes_all)
    mtg = make_grid(img_all, nrow=samp_n)
    ToPILImage()(mtg).show()
    ctf_str = "_".join([str(ct) for ct in cutoff_list])
    plt.save(join(figdir, "%s_%s_%04d.png"%(savestr, ctf_str, RND)), mtg.numpy(), )
    return mtg

samp_n = 10
refvec = 0.7*np.random.randn(samp_n, 512)
cutoff_list = 5, 10, 20, 40, 80, 120, 160, 200, 250, 300, 350, 400, 450, 500
vis_eigen_cutoff(refvec, evc_avg, cutoff_list, SG, figdir=figdir)