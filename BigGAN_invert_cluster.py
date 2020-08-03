#%%
from pytorch_pretrained_biggan import BigGAN, BigGANConfig, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
from skimage.transform import resize, rescale
from imageio import imread, imsave
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sys
import os
from os.path import join
from time import time
#%%
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN

if sys.platform == "linux":
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    BGAN = get_BigGAN()
    Hpath = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    imgfolder = r"/scratch/binxu/Datasets/ImageTranslation/GAN_real/B/train"
    savedir = r"/scratch/binxu/GAN_invert/ImageNet"
else:
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    Hpath = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    imgfolder = r"E:\Cluster_Backup\Datasets\ImageTranslation\GAN_real\B\train"
    savedir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet"
BGAN.cuda().eval()
for param in BGAN.parameters():
    param.requires_grad_(False)

#%% Set up loss
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])

def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean()

alpha = 5  # relative weight
#%%
data = np.load(Hpath)
evc_clas = torch.from_numpy(data['eigvects_clas_avg']).cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg']).cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()
#%%
def resize_center_crop(curimg, final_L=256):
    """Useful handy function to crop image net images to center square and resize it to
        desired resolution (final_L)"""
    if len(curimg.shape) == 2:
        curimg = np.repeat(curimg[:, :, np.newaxis], 3, 2)
    H, W, _ = curimg.shape
    if H <= W:
        newW = round(float(W) / H * final_L)
        rsz_img = resize(curimg, (final_L, newW))
        offset = (newW - final_L) // 2
        fin_img = rsz_img[:, offset:offset + final_L, :]
    else:
        newH = round(float(H) / W * final_L)
        rsz_img = resize(curimg, (newH, final_L))
        offset = (newH - final_L) // 2
        fin_img = rsz_img[offset:offset + final_L, :, :]
    return fin_img

def BigGAN_invert(target_tsr, param, basis="all", maxstep=600, init_code=None,
                  ckpt_steps=(50, 100, 200, 300, 400, 500), savedir=savedir,
                  namestr="", RND=None):
    lr = 10 ** param[0, 0]
    beta1 = 1 - 10 ** param[0, 1]  # param[2] = log10(1 - beta1)
    beta2 = 1 - 10 ** param[0, 2]  # param[3] = log10(1 - beta2)
    reg_w1 = 10 ** param[0, 3]  # param[2] = log10(1 - beta1)
    reg_w2 = 10 ** param[0, 4]  # param[3] = log10(1 - beta2)
    sched_gamma = param[0, 5]
    if init_code is None:
        noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
        class_init = 0.06 * torch.randn(1, 128).cuda()
    elif isinstance(init_code, np.ndarray):
        code_init = torch.from_numpy(init_code)
        noise_init = code_init[:,:128]
        class_init = code_init[:,128:]
    elif isinstance(init_code, torch.Tensor):
        noise_init = init_code[:,:128]
        class_init = init_code[:,128:]
    else:
        raise
    if basis == "all":
        latent_coef = (torch.cat((noise_init, class_init), dim=1) @ evc_all).detach().clone().requires_grad_(True)
    elif basis == "sep":
        latent_coef = (torch.cat((noise_init @ evc_nois, class_init @ evc_clas), dim=1)).detach().clone().requires_grad_(True)
    else:
        latent_coef = (torch.cat((noise_init, class_init), dim=1)).detach().clone().requires_grad_(True)
    optim = Adam([latent_coef], lr=lr, weight_decay=0, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=sched_gamma)
    RNDid = np.random.randint(1000000) if RND is None else RND
    scores_all = []
    nos_norm = []
    cls_norm = []
    for step in range(maxstep):
        optim.zero_grad()
        if basis == "all":
            latent_code = latent_coef @ evc_all.T
        elif basis == "sep":
            latent_code = torch.cat((latent_coef[:, :128] @ evc_nois.T, latent_coef[:, 128:] @ evc_clas.T), dim=1)
        else:
            latent_code = latent_coef
        noise_vec = latent_code[:, :128]
        class_vec = latent_code[:, 128:]
        fitimg = BGAN.generator(latent_code, 0.7)
        fitimg = torch.clamp((1.0 + fitimg) / 2.0, 0, 1)
        dsim = alpha * ImDist(fitimg, target_tsr) + L1loss(fitimg, target_tsr)  #
        loss = dsim + reg_w1 * noise_vec.pow(2).sum() + reg_w2 * class_vec.pow(2).sum()
        loss.backward()
        optim.step()
        scheduler.step()
        scores_all.append(dsim.item())
        nos_norm.append(noise_vec.norm().item())
        cls_norm.append(class_vec.norm().item())
        if (step + 1) % 10 == 0:
            print("step%d loss %.2f norm: cls: %.2f nois: %.1f" % (step, scores_all[-1], cls_norm[-1], nos_norm[-1]))
        if (step + 1) in ckpt_steps:
            imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
            imcmp.save(join(savedir, "%s_H%sreg%06d_%.3f_s%d.jpg" % (namestr, basis, RNDid, dsim.item(), step + 1)))

    imcmp = ToPILImage()(make_grid(torch.cat((fitimg, target_tsr)).cpu()))
    imcmp.save(join(savedir, "%s_H%sreg%06d_%.3f_final.jpg" % (namestr, basis, RNDid, dsim.item())))
    # imcmp.show()
    fig, ax = plt.subplots()
    ax.plot(scores_all, label="Loss")
    ax.set_ylabel("Image Dissimilarity", color="blue", fontsize=14)
    plt.legend()
    ax2 = ax.twinx()
    ax2.plot(nos_norm, color="orange", label="noise")
    ax2.plot(cls_norm, color="magenta", label="class")
    ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
    plt.legend()
    plt.title("lr %.E beta1 %.3f beta2 %.3f wd_nos %.E wd_cls %.E gamma %.1f"%(lr,beta1,beta2,reg_w1,reg_w2,sched_gamma))
    plt.savefig(join(savedir, "%s_traj_H%sreg%06d_%.3f.jpg" % (namestr, basis, RNDid, dsim.item())))
    np.savez(join(savedir, "%s_code_H%sreg%06d.jpg" % (namestr, basis, RNDid,)), dsim=dsim.item(), scores_all=np.array(
        scores_all), nos_norm=np.array(nos_norm),cls_norm=np.array(cls_norm), code=latent_code.detach().cpu().numpy())
    return dsim.item()

#%%
from tqdm import tqdm
exprecord = []
csr_min, csr_max = 550, 600
if len(sys.argv) > 1:
    csr_min = int(sys.argv[1])
    csr_max = int(sys.argv[2])

for imgid in tqdm(range(csr_min, csr_max)):
    print("Processing image %d" %imgid)
    imgnm = "val_crop_%08d"%imgid
    img = imread(join(imgfolder, "val_crop_%08d.JPEG"%imgid))
    target_tsr = torch.from_numpy(img / 255.0).permute([2, 0, 1]).unsqueeze(0)
    target_tsr = target_tsr.float().cuda()
    #%%
    for triali in range(5):
        RNDid = np.random.randint(1000000)
        noise_init = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
        class_init = 0.06 * torch.randn(1, 128).cuda()
        init_code = torch.cat((noise_init,class_init),1)
        dsim_all = BigGAN_invert(target_tsr, np.array([[-1.0, -0.5, -2.44, -4.66, -3.2552, 0.4563]]),
                     init_code=init_code,
                    basis="all", maxstep=600, ckpt_steps=(50, 100, 200, 300, 400, 500),
                    savedir=savedir, namestr=imgnm, RND=RNDid)
        dsim_sep = BigGAN_invert(target_tsr, np.array([[-1, -0.5, -2.24, -5, -3.5, 0.59]]),
                      init_code=init_code,
                      basis="sep", maxstep=600, ckpt_steps=(50, 100, 200, 300, 400, 500),
                      savedir=savedir, namestr=imgnm, RND=RNDid)
        dsim_none = BigGAN_invert(target_tsr, np.array([[-1, -0.78, -2.15, -4.22, -2.77, 0.76]]),
                      init_code=init_code,
                      basis="none", maxstep=600, ckpt_steps=(50, 100, 200, 300, 400, 500),
                      savedir=savedir, namestr=imgnm, RND=RNDid)
        exprecord.append([imgid, imgnm, triali, dsim_all, dsim_sep, dsim_none, RNDid])

np.savez(join(savedir, "cmp_exprecord_%d-%d.npz"%(csr_min, csr_max)), exprecord=exprecord)
