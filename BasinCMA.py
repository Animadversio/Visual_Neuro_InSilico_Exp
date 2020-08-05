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
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN

if sys.platform == "linux":
    BGAN = get_BigGAN()
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    Hpath = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    imgfolder = r"/scratch/binxu/Datasets/ImageTranslation/GAN_real/B/train"
    savedir = r"/scratch/binxu/GAN_invert/BasinCMA/ImageNet"
else:
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    Hpath = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    imgfolder = r"E:\Cluster_Backup\Datasets\ImageTranslation\GAN_real\B\train"
    # savedir = r"E:\Cluster_Backup\BigGAN_invert\ImageNet"
    savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\BasinCMA\ImageNet"
#%%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--ImDist", type=str, default="squeeze", help="Network model to use for Image distance computation")
parser.add_argument("--dataset", type=str, default="ImageNet", help="")
parser.add_argument("--imgidx", type=int, nargs='+', default=[200, 220], help="Network model to use for Image distance "
                                                                   "computation")
parser.add_argument("--cmasteps", type=int, default=10, help="")
parser.add_argument("--gradsteps", type=int, default=10, help="")
parser.add_argument("--finalgradsteps", type=int, default=500, help="")
parser.add_argument("--CMApostGrad", type=bool, default=True, help="")
parser.add_argument("--basis", type=str, default="all", help="")
args = parser.parse_args([])
#%%
import models  # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net=args.ImDist, use_gpu=1, gpu_ids=[0])
for param in ImDist.parameters():
    param.requires_grad_(False)
def L1loss(target, img):
    return (img - target).abs().sum(axis=1).mean(axis=1)

# alpha = 5 # relative weight
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
#%% Load the precomputed Hessian and Form them as basis matrix
data = np.load(Hpath)
evc_clas = torch.from_numpy(data['eigvects_clas_avg'])#.cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg'])#.cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()
evc_sep = torch.from_numpy(block_diag(data['eigvects_nois_avg'], data['eigvects_clas_avg'])).cuda()
evc_none = torch.eye(256).cuda()
#%%
"""Main function to do inversion."""
def BasinCMA(target_tsr, cmasteps=30, gradsteps=40, finalgrad=500, batch_size=4,
             basis="all", CMApostAdam=False, RND=None, savedir=savedir, imgnm=""):
    Record = {"L1cma": [],"dsimcma": [], "L1Adam": [], "dsimAdam": [], "L1refine":[], "dsimrefine":[], "classnorm":[], "noisenorm":[]}
    # choose the basis vector to use in Adam optimization
    basisvec = {"all": evc_all, "sep": evc_sep, "none": evc_none}[basis]
    fixnoise = truncated_noise_sample(1, 128)
    optim_noise = cma.CMAEvolutionStrategy(fixnoise, 0.4)#0.4)  # 0.2
    optim_class = cma.CMAEvolutionStrategy(128 * [0.0], 0.2)#0.12)  # 0.06
    # noise_vec = torch.from_numpy(fixnoise)
    RND = np.random.randint(1E6) if RND is None else RND
    # Outer Loop: CMA optimization of initial points
    for i in tqdm.trange(cmasteps, desc="CMA steps"):
        class_codes = optim_class.ask()
        noise_codes = optim_noise.ask()
        # TODO: boundary handling by projection in code space
        # Evaluate the cma proposed codes `latent_code` at first.
        codes_tsr = torch.from_numpy(np.array(class_codes)).float()
        noise_tsr = torch.from_numpy(np.array(noise_codes)).float()
        latent_code = torch.cat((noise_tsr, codes_tsr), dim=1).cuda()  # this initialize inner loop
        with torch.no_grad():
            imgs = BGAN.generator(latent_code, 0.7)
            imgs = (imgs + 1.0) / 2.0
            dsims = ImDist(imgs, target_tsr).squeeze()
            L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
        scores = dsims.detach().cpu().numpy()
        L1score = L1dsim.detach().cpu().numpy()
        print("step %d pre-ADAM dsim %.3f L1 %.3f (norm %.2f noise norm %.2f)" % (
            i, scores.mean(), L1score.mean(), codes_tsr.norm(dim=1).mean(), noise_tsr.norm(dim=1).mean()))
        Record["L1cma"].append(L1score)
        Record["dsimcma"].append(scores)
        # Inner loop: ADAM optimization from the cma proposed codes `latent_code` batch by batch
        codes_post = np.zeros_like(np.hstack((noise_codes, class_codes)))
        scores_post = np.zeros_like(scores)
        L1score_post = np.zeros_like(L1score)
        if gradsteps > 0:
            csr = 0
            while csr < len(class_codes):  # pbar = tqdm.tqdm(total=len(codes), initial=csr, desc="batchs")
                csr_end = min(len(class_codes), csr + batch_size)
                # codes_batch = codes_tsr[csr:csr_end, :].detach().clone().requires_grad_(True)
                coef_batch = (latent_code[csr:csr_end, :] @ basisvec).detach().clone().requires_grad_(True)
                optim = Adam([coef_batch], lr=0.05, )
                for step in range(gradsteps):  # tqdm.trange(gradsteps, desc="ADAM steps"):
                    optim.zero_grad()
                    latent_batch = coef_batch @ basisvec.T
                    imgs = BGAN.generator(latent_batch, 0.7)
                    imgs = (imgs + 1.0) / 2.0
                    dsims = ImDist(imgs, target_tsr).squeeze()
                    L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
                    loss = (dsims + L1dsim).sum()
                    loss.backward()
                    optim.step()
                    if (step + 1) % 10 == 0:
                        print("step %d dsim %.3f L1 %.3f" % (step, dsims.mean().item(), L1dsim.mean().item(),))
                code_batch = (coef_batch @ evc_all.T).detach().cpu().numpy()
                scores_batch = dsims.detach().cpu().numpy()
                L1score_batch = L1dsim.detach().cpu().numpy()
                codes_post[csr:csr_end, :] = code_batch
                scores_post[csr:csr_end] = scores_batch
                L1score_post[csr:csr_end] = L1score_batch
                csr = csr_end
            # Use the ADAM optimized scores as utility for `latent_code` and do cma update
            print("step %d post-ADAM dsim %.3f L1 %.3f (norm %.2f, norm %.2f)" % (
                i, scores_post.mean(), L1score_post.mean(),
                np.linalg.norm(codes_post[:, 128:], axis=1).mean(),
                np.linalg.norm(codes_post[:, :128], axis=1).mean()))
        else:  # if no grad step is performed
            scores_post = scores
            L1score_post = L1score
            codes_post = np.hstack((noise_codes, class_codes))
        Record["L1Adam"].append(L1score_post)
        Record["dsimAdam"].append(scores_post)
        if CMApostAdam:
            optim_class.tell(codes_post[:, :128], scores_post + L1score_post)
            optim_noise.tell(codes_post[:, 128:], scores_post + L1score_post)
        else:
            optim_class.tell(class_codes, scores_post + L1score_post)
            optim_noise.tell(noise_codes, scores_post + L1score_post)

    # Sort the scores and find the codes with the least scores to be final refined
    idx = np.argsort((L1score_post + scores_post))
    codes_batch = torch.from_numpy(codes_post[idx[:4]]).float().cuda()
    fit_img = BGAN.generator(codes_batch, 0.7)
    fit_img = (fit_img + 1) / 2.0
    CMAfitimg = ToPILImage()(make_grid(torch.cat((fit_img, target_tsr)).cpu()))
    CMAfitimg.save(join(savedir, "%s_CMA_final%06d.jpg" % (imgnm, RND)))
    CMAfitimg.show()
    # Linear Reparametrization using basisvec
    coef_batch = (codes_batch @ basisvec).detach().clone().requires_grad_(True)
    optim = Adam([coef_batch], lr=0.05, )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.5)
    for step in range(finalgrad):  # tqdm.trange(gradsteps, desc="ADAM steps"):
        optim.zero_grad()
        # latent_code = torch.cat((noise_vec.repeat(codes_batch.shape[0], 1), codes_batch), dim=1).cuda()
        latent_code = coef_batch @ basisvec.T  #evc_all.T
        imgs = BGAN.generator(latent_code, 0.7)
        imgs = (imgs + 1.0) / 2.0
        dsims = ImDist(imgs, target_tsr).squeeze()
        L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
        loss = (dsims + L1dsim).sum()
        loss.backward()
        optim.step()
        scheduler.step()
        Record["L1refine"].append(L1dsim.detach().cpu().numpy())
        Record["dsimrefine"].append(dsims.detach().cpu().numpy())
        Record["classnorm"].append(latent_code[:, 128:].norm(dim=1).detach().cpu().numpy())
        Record["noisenorm"].append(latent_code[:, :128].norm(dim=1).detach().cpu().numpy())
        if (step + 1) % 10 == 0:
            print("step %d dsim %.3f L1 %.3f (norm %.2f)" % (
                step, dsims.mean().item(), L1dsim.mean().item(), latent_code.norm(dim=1).mean().item()))
    scores_final = dsims.detach().cpu().numpy()
    L1score_final = L1dsim.detach().cpu().numpy()
    finalimg = ToPILImage()(make_grid(torch.cat((imgs, target_tsr)).cpu()))
    finalimg.save(join(savedir, "%srefinefinal%06d.jpg" % (imgnm, RND)))
    finalimg.show()

    fig = visualize_optim(Record, titlestr="cmasteps %d gradsteps %d refinesteps %d Hbasis %s, CMApostAdam %d"%(cmasteps, gradsteps, finalgrad, basis, CMApostAdam))
    fig.savefig(join(savedir, "%straj_H%s%s_%d_dsim_%.3f_L1_%.3f.jpg" % (imgnm, basis, "_postAdam" if CMApostAdam else "",
                                                                        RND, scores_final.min(), L1score_final.min())))
    np.savez(join(savedir, "%soptim_data_%d.npz" % (imgnm, RND)), codes=latent_code.cpu().detach().numpy(),
             Record=Record, dsims=scores_final, L1dsims=L1score_final)

    return imgs.cpu().detach().numpy(), latent_code.cpu().detach().numpy(), scores_final, L1score_final, Record

def visualize_optim(Record, titlestr="", savestr=""):
    fig, ax = plt.subplots()
    L1_cma_tr = np.array(Record["L1cma"])
    dsim_cma_tr = np.array(Record["dsimcma"])
    L1_adam_tr = np.array(Record["L1Adam"])
    dsim_adam_tr = np.array(Record["dsimAdam"])
    dsim_tr = np.array(Record["dsimrefine"])
    L1_tr = np.array(Record["L1refine"])
    nos_norm = np.array(Record["noisenorm"])
    cls_norm = np.array(Record["classnorm"])
    cma_steps, popsize = L1_cma_tr.shape
    xticks_arr = np.arange(-cma_steps*10, 0, 10)[:,np.newaxis].repeat(popsize, 1)
    ax.scatter(xticks_arr, L1_cma_tr, color="blue", s=15)
    ax.scatter(xticks_arr, dsim_cma_tr, color="green", s=15)
    ax.scatter(xticks_arr+0.2, L1_adam_tr, color="blue", s=15)
    ax.scatter(xticks_arr+0.2, dsim_adam_tr, color="green", s=15)
    ax.plot(dsim_tr, label="dsim", color="green", alpha=0.7)
    ax.plot(L1_tr, label="L1", color="blue", alpha=0.7)
    ax.set_ylabel("Image Dissimilarity", color="blue", fontsize=14)
    plt.legend()
    ax2 = ax.twinx()
    ax2.plot(nos_norm, color="orange", label="noise", alpha=0.7)
    ax2.plot(cls_norm, color="magenta", label="class", alpha=0.7)
    ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
    plt.legend()
    plt.title(titlestr)
    plt.show()
    return fig
#%%
from imageio import imread
target = imread("block042_thread000_gen_gen041_001030.bmp")
target_tsr = torch.from_numpy(target / 255.0).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
#%%
# savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\BasinCMA\ImageNet"
# imgfolder = r"E:\Datasets\ImageNet\resize"
# Append the name of parameters to the folder
expdir = join(savedir, "CMA%dAdam%dFinal%d%s_%s"%(args.cmasteps, args.gradsteps, args.finalgradsteps,
                                                  "_postAdam" if args.CMApostGrad else "", args.basis))
os.makedirs(expdir, exist_ok=True)
idmin, idmax = args.imgidx
for imgid in tqdm.trange(idmin, idmax):
    print("Processing image %d" %imgid)
    imgnm = "val_crop_%08d"%imgid
    img = imread(join(imgfolder, "val_crop_%08d.JPEG"%imgid))
    target_tsr = torch.from_numpy(img / 255.0).permute([2, 0, 1]).unsqueeze(0)
    target_tsr = target_tsr.float().cuda()
    imgs_np, codes_np, dsims, L1dsims, Record = BasinCMA(target_tsr, cmasteps=args.cmasteps, gradsteps=args.gradsteps, finalgrad=args.finalgradsteps,
                                                         batch_size=4, basis=args.basis, CMApostAdam=args.CMApostGrad,
                                                         imgnm=imgnm, savedir=expdir)
    # imgs_np, codes_np, dsims, L1dsims, Record = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500,
    #                                                      batch_size=4, basis="all", CMApostAdam=True)

#%%


#%%
# L1_cma_tr = np.array(Record["L1cma"])
# dsim_cma_tr = np.array(Record["dsimcma"])
# L1_adam_tr = np.array(Record["L1Adam"])
# dsim_adam_tr = np.array(Record["dsimAdam"])
# dsim_tr = np.array(Record["dsimrefine"])
# L1_tr = np.array(Record["L1refine"])
# nos_norm = np.array(Record["noisenorm"])
# cls_norm = np.array(Record["classnorm"])
#
# cma_steps, popsize = L1_cma_tr.shape
# xticks_arr = np.arange(-cma_steps, 0)[:, np.newaxis].repeat(popsize,1)
# plt.scatter(xticks_arr, L1_cma_tr, color="blue")
# plt.scatter(xticks_arr, dsim_cma_tr, color="green")
# plt.scatter(xticks_arr+0.2, L1_adam_tr, color="blue")
# plt.scatter(xticks_arr+0.2, dsim_adam_tr, color="green")
# plt.show()


#%%
# imgs_np, codes_np, dsims, L1dsims, Record = BasinCMA(target_tsr, cmasteps=50, gradsteps=0, finalgrad=400, batch_size=4, basis="all")
# #%%
# imgs_np2, codes_np2, dsims2, L1dsims2, Record2 = BasinCMA(target_tsr, cmasteps=50, gradsteps=0, finalgrad=400, batch_size=4, basis="all")
# #%%
# imgs_np3, codes_np3, dsims3, L1dsims3, Record3 = BasinCMA(target_tsr, cmasteps=10, gradsteps=20, finalgrad=400, batch_size=4, basis="all")
# #%%
# imgs_np4, codes_np4, dsims4, L1dsims4, Record4 = BasinCMA(target_tsr, cmasteps=50, gradsteps=0, finalgrad=300, batch_size=4, basis="all")
#
# #%%
# RND = np.random.randint(1E6)
# imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=5, gradsteps=5, finalgrad=400, batch_size=4, basis="all", CMApostAdam=True,  RND=RND)
# fig = visualize_optim(Record5, titlestr="cmasteps5 gradsteps5 Hbasis all, CMApostAdam")
# fig.savefig(join(savedir, "traj_%d_dsim_%.3f_L1_%.3f.jpg"%(RND, dsims5.min(), L1dsims5.min())))
# np.savez(join(savedir, "optim_data_%d.npz"%RND), codes=codes_np5, Record=Record5, dsims=dsims5, L1dsims=L1dsims5)
# #%%
# RND = np.random.randint(1E6)
# imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="all", CMApostAdam=True,  RND=RND)
# fig = visualize_optim(Record5, titlestr="cmasteps5 gradsteps5 Hbasis all, CMApostAdam")
# fig.savefig(join(savedir, "traj_Hall_%d_dsim_%.3f_L1_%.3f.jpg"%(RND, dsims5.min(), L1dsims5.min())))
# np.savez(join(savedir, "optim_data_%d.npz"%RND), codes=codes_np5, Record=Record5, dsims=dsims5, L1dsims=L1dsims5)
# #%%
# RND = np.random.randint(1E6)
# imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="all", CMApostAdam=True,  RND=RND)
# fig = visualize_optim(Record5, titlestr="cmasteps5 gradsteps5 Hbasis all, CMApostAdam")
# fig.savefig(join(savedir, "traj_Hall_%d_dsim_%.3f_L1_%.3f.jpg"%(RND, dsims5.min(), L1dsims5.min())))
# np.savez(join(savedir, "optim_data_%d.npz"%RND), codes=codes_np5, Record=Record5, dsims=dsims5, L1dsims=L1dsims5)
# #%% This is pretty gfood
# imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=400, batch_size=4, basis="all", CMApostAdam=True)
# #%%
# imgs_np6, codes_np6, dsims6, L1dsims6, Record6 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="none")
# #%%
#
# from time import time
# T0 = time()
# for trial in range(5):
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="all", CMApostAdam=True)
#
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="sep", CMApostAdam=True)
#
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="none", CMApostAdam=True)
#
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="all", CMApostAdam=False)
#
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="sep", CMApostAdam=False)
#
#     imgs_np5, codes_np5, dsims5, L1dsims5, Record5 = BasinCMA(target_tsr, cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="none", CMApostAdam=False)
#     print(time() - T0) # 5 hours
