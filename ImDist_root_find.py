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
from time import time
from imageio import imread
from scipy.linalg import block_diag
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, GANForwardMetricHVPOperator, compute_hessian_eigenthings, get_full_hessian
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
from GAN_utils import BigGAN_wrapper
BGAN = BigGAN.from_pretrained("biggan-deep-256")
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
EmbedMat = BGAN.embeddings.weight
G = BigGAN_wrapper(BGAN)
#%%
# noise_vec = torch.from_numpy(truncated_noise_sample(1, 128)).cuda()
# class_vec = EmbedMat[:, 373:374].T
# final_latent = torch.cat((noise_vec, class_vec), dim=1).cuda()
# fit_img = BGAN.generator(final_latent, 0.7)
# fit_img = (fit_img + 1.0) / 2.0
# #%%
# mov_latent = final_latent.detach().clone().requires_grad_(True)
# mov_img = BGAN.generator(mov_latent, 0.7)
# mov_img = (mov_img + 1.0) / 2.0
# dsim = ImDist(fit_img, mov_img)
# H = get_full_hessian(dsim, mov_latent)
# eigval, eigvec = np.linalg.eigh(H)
# del dsim
# torch.cuda.empty_cache()
# #%%
# eigi = -1
# tan_vec = torch.from_numpy(eigvec[:, eigi]).unsqueeze(0).float().cuda()
# stepsize = (0.5 / eigval[eigi])**(1/2)
# #%%
# ticks = np.arange(1,5) * 2 * stepsize
#
# step_latents = torch.tensor(ticks).unsqueeze(1).float().cuda() @ tan_vec + final_latent
# with torch.no_grad():
#     step_imgs = BGAN.generator(step_latents, 0.7)
#     step_imgs = (step_imgs + 1.0) / 2.0
#     dist_steps = ImDist(step_imgs, fit_img).squeeze()
# #%%
# ticks = np.arange(1,5) * 2 * stepsize
# target_dist = torch.tensor([0.1, 0.2, 0.3, 0.4, ]).cuda()
# tick_tsr = torch.tensor(ticks).unsqueeze(1).float().cuda().requires_grad_(True)
# optimizer = Adam([tick_tsr], lr=0.1)
# #%%
# for i in range(50):
#     optimizer.zero_grad()
#     step_latents = tick_tsr @ tan_vec + final_latent
#     step_imgs = BGAN.generator(step_latents, 0.7)
#     step_imgs = (step_imgs + 1.0) / 2.0
#     dist_steps = ImDist(step_imgs, fit_img).squeeze()
#     loss = (target_dist - dist_steps).pow(2).mean()
#     loss.backward()
#     optimizer.step()
#     if (i) % 10 == 0:
#         print("step %d dsim %.3f" % (i, loss.item(), ))
# #%%
# from scipy.optimize import root_scalar
# #%%
# def dist_step(tick):
#     step_latents = tick * tan_vec + final_latent
#     with torch.no_grad():
#         step_imgs = BGAN.generator(step_latents, 0.7)
#         step_imgs = (step_imgs + 1.0) / 2.0
#         dist_steps = ImDist(step_imgs, fit_img).squeeze()
#     return dist_steps.item()
# #%%
# from time import time
# t0 = time()
# pos_ticks_target = []
# neg_ticks_target = []
# for dist_target in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     xsol = root_scalar(lambda tick: dist_step(tick)-dist_target, bracket=(0, 1), xtol=1E-4)
#     pos_ticks_target.append(xsol.root)
#     xsol2 = root_scalar(lambda tick: dist_step(tick) - dist_target, bracket=(-1, 0), xtol=1E-4)
#     neg_ticks_target.append(xsol2.root)
# print(time() - t0)
#%%
data = np.load("N:\Hess_imgs\summary\Hess_mat.npz")
refvec = data["vect"]
evc_clas = data['eigvects_clas']
evc_clas_tsr = torch.from_numpy(data['eigvects_clas'][:, ::-1].copy()).float().cuda()
eva_clas = data['eigvals_clas'][::-1]
evc_nois = data['eigvects_nois']
evc_nois_tsr = torch.from_numpy(data['eigvects_nois'][:, ::-1].copy()).float().cuda()
eva_nois = data['eigvals_nois'][::-1]
#%%
reftsr = torch.tensor(refvec).float().cuda()
refimg = G.visualize(reftsr)
ToPILImage()(refimg[0, :].cpu())
#%%
def dist_step2(ticks, refvec, tanvec, refimg):
    step_latents = torch.tensor(ticks).float().cuda().view(-1, 1) @ tanvec + refvec
    with torch.no_grad():
        step_imgs = BGAN.generator(step_latents, 0.7)
        step_imgs = (step_imgs + 1.0) / 2.0
        dist_steps = ImDist(step_imgs, refimg).squeeze()
    return dist_steps.squeeze().cpu().numpy(), step_imgs
#%% Develop the root finding algorithms here.
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, PchipInterpolator
from scipy.optimize import newton, root_scalar, minimize_scalar
targ_val = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
eigid = 72; space = "noise"; pos = False
if space == "class":
    tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid + 1].T), dim=1)
elif space == "noise":
    tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid + 1].T, torch.zeros(1, 128).cuda()), dim=1)

t0 = time()
xval = [0]
yval = [0]
sign = 1 if pos else -1
bbox = [0, 4] if pos else [-4, 0]
xnext = sign * np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3])
for step in range(1+5):
    xcur = xnext
    ycur, imgs = dist_step2(xcur, reftsr, tan_vec, refimg)
    xval.extend(list(xcur))
    yval.extend(list(ycur))
    uniq_x, uniq_idx = np.unique(xval, return_index=True)  # sort and unique x data
    uniq_y = np.array(yval)[uniq_idx]
    interp_fn = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
    # interp_fn = InterpolatedUnivariateSpline(uniq_x, uniq_y,  k=3, ext=0)#bbox=bbox,
    # idx = np.argsort(xval)
    # interp_fn = interp1d(np.array(xval)[idx], np.array(yval)[idx], 'quadratic')
    xnext = []
    sol = []
    for fval in targ_val:
        lowidx  = np.where((uniq_y < fval))[0]  #  * (uniq_x >= 0 if pos else uniq_x <= 0)
        highidx = np.where((uniq_y > fval))[0]  #  * (uniq_x >= 0 if pos else uniq_x <= 0)
        # lowidx should never be empty.
        lowrelidx = np.abs(uniq_y[lowidx] - fval).argmin()
        lowbnd_x = uniq_x[lowidx[lowrelidx]]  # this should be closer to 0.
        if len(highidx) == 0:  # no point reach such distance, have to choose points with lower distances
            lowbnd_x2 = uniq_x[lowidx[lowrelidx] - 1] if pos else uniq_x[lowidx[lowrelidx] + 1]
            unbound = True
        else:  # some point reaches higher distance, so you can bound your search in a bracket.
            highrelidx = np.abs(uniq_y[highidx] - fval).argmin()
            highbnd_x = uniq_x[highidx[highrelidx]]
            unbound = False
        try:
            if unbound:
                interp_fn2 = lambda x: np.abs(interp_fn(x) - fval)
                result = minimize_scalar(interp_fn2, bounds=[lowbnd_x, 20] if pos else [-20, lowbnd_x],
                                        method='bounded',)
                xhat = result.x
                # xhat = root_scalar(interp_fn2, x0=lowbnd_x, x1=lowbnd_x2, )
            else:
                interp_fn2 = lambda x: interp_fn(x) - fval
                result = root_scalar(interp_fn2, x0=lowbnd_x, x1=highbnd_x, bracket=(lowbnd_x, highbnd_x))
                xhat = result.root
            # if idx == len(uniq_x)-1:# and pos:
            #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=max(uniq_x)+1)
            # elif idx == 0:# and not pos:
            #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=min(uniq_x) - 1)
            # else:
            #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=uniq_x[idx + 1])
            # xhat = newton(interp_fn2, xval[idx])
            xnext.append(xhat)
            sol.append(xhat)
        except RuntimeError as e:
            print(e.args)
            xextra = max(xval) + 1 if pos else min(xval) - 1
            xnext.append(xextra)
    if step > 0:
        if np.max(np.abs(targ_val - ycur)) < 1E-4:
            break
ycur, imgs = dist_step2(sol, reftsr, tan_vec, refimg)
print(time() - t0)
ToPILImage()(make_grid(imgs).cpu()).show()
#%%
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, PchipInterpolator
from scipy.optimize import newton, root_scalar, minimize_scalar
def find_level_step(targ_val, reftsr, tan_vec, refimg, iter=2, pos=True):
    xval = [0]
    yval = [0]
    sign = 1 if pos else -1
    bbox = [0, 4] if pos else [-4, 0]
    xnext = sign * np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3])
    for step in range(1 + iter):
        xcur = xnext
        ycur, imgs = dist_step2(xcur, reftsr, tan_vec, refimg)
        xval.extend(list(xcur))
        yval.extend(list(ycur))
        uniq_x, uniq_idx = np.unique(xval, return_index=True)  # sort and unique x data
        uniq_y = np.array(yval)[uniq_idx]
        interp_fn = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
        # interp_fn = InterpolatedUnivariateSpline(uniq_x, uniq_y,  k=3, ext=0)#bbox=bbox,
        # idx = np.argsort(xval)
        # interp_fn = interp1d(np.array(xval)[idx], np.array(yval)[idx], 'quadratic')
        xnext = []
        sol = []
        for fval in targ_val:
            lowidx = np.where((uniq_y < fval))[0]  # * (uniq_x >= 0 if pos else uniq_x <= 0)
            highidx = np.where((uniq_y > fval))[0]  # * (uniq_x >= 0 if pos else uniq_x <= 0)
            # lowidx should never be empty.
            lowrelidx = np.abs(uniq_y[lowidx] - fval).argmin()
            lowbnd_x = uniq_x[lowidx[lowrelidx]]  # this should be closer to 0.
            if len(highidx) == 0:  # no point reach such distance, have to choose points with lower distances
                lowbnd_x2 = uniq_x[lowidx[lowrelidx] - 1] if pos else uniq_x[lowidx[lowrelidx] + 1]
                unbound = True
            else:  # some point reaches higher distance, so you can bound your search in a bracket.
                highrelidx = np.abs(uniq_y[highidx] - fval).argmin()
                highbnd_x = uniq_x[highidx[highrelidx]]
                unbound = False
            try:
                if unbound:
                    interp_fn2 = lambda x: np.abs(interp_fn(x) - fval)
                    result = minimize_scalar(interp_fn2, bounds=[lowbnd_x, 20] if pos else [-20, lowbnd_x],
                                             method='bounded', )
                    xhat = result.x
                    # xhat = root_scalar(interp_fn2, x0=lowbnd_x, x1=lowbnd_x2, )
                else:
                    interp_fn2 = lambda x: interp_fn(x) - fval
                    result = root_scalar(interp_fn2, x0=lowbnd_x, x1=highbnd_x, bracket=(lowbnd_x, highbnd_x))
                    xhat = result.root
                # if idx == len(uniq_x)-1:# and pos:
                #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=max(uniq_x)+1)
                # elif idx == 0:# and not pos:
                #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=min(uniq_x) - 1)
                # else:
                #     xhat = root_scalar(interp_fn2, x0=uniq_x[idx], x1=uniq_x[idx + 1])
                # xhat = newton(interp_fn2, xval[idx])
                xnext.append(xhat)
                sol.append(xhat)
            except RuntimeError as e:
                print(e.args)
                xextra = max(xval) + 1 if pos else min(xval) - 1
                xnext.append(xextra)
        if step > 0:
            if np.max(np.abs(targ_val - ycur)) < 1E-5:
                break
    ycur, imgs = dist_step2(sol, reftsr, tan_vec, refimg)
    print(np.abs(targ_val - ycur))
    return sol, ycur, imgs
#%%
newimg_dir = r"N:\Hess_imgs_new_new"
summary_dir = r"N:\Hess_imgs_new_new\summary"
from imageio import imwrite
targ_val = np.array([0.08, 0.16, 0.24, 0.32, 0.4])
space = "noise"
imgall = None
xtick_col = []
dsim_col = []
vecs_col = []
img_names = []

t0 = time()
for eigid in range(128): # [0,1,2,3,4,5,6,7,8,10,20,30,40]:#
    if space == "class":
        tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid+1].T), dim=1)
    elif space == "noise":
        tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid+1].T, torch.zeros(1, 128).cuda()), dim=1)
    xtar_pos, ytar_pos, stepimgs_pos = find_level_step(targ_val, reftsr, tan_vec, refimg, iter=6, pos=True)
    xtar_neg, ytar_neg, stepimgs_neg = find_level_step(targ_val, reftsr, tan_vec, refimg, iter=6, pos=False)
    imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
    xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
    dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
    vecs_row = torch.tensor(xticks_row).cuda().view(-1,1) @ tan_vec + reftsr

    xtick_col.append(xticks_row)
    dsim_col.append(dsim_row)
    vecs_col.append(vecs_row.cpu().numpy())
    img_names.extend("noise_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in np.linspace(-0.4, 0.4, 11))  # dsim_row)
    imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))
    print(time() - t0)
#%%
mtg1 = ToPILImage()(make_grid(imgall, nrow=11).cpu())  # 20sec for 13 rows not bad
mtg1.show()
mtg1.save(join(summary_dir, "noise_space_all_var.jpg"))
npimgs = imgall.permute([2,3,1,0]).numpy()
for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
#%%
xtick_arr = np.array(xtick_col)
dsim_arr = np.array(dsim_col)
vecs_arr = np.array(vecs_col)
np.savez(join(summary_dir, "ImDist_root_data.npz"), xtick_arr=xtick_arr, dsim_arr=dsim_arr, vecs_arr=vecs_arr,
         targ_val=targ_val)
#%%
plt.figure(figsize=[10, 7])
plt.plot(xtick_arr)
plt.xlabel("Eigenvalue index")
plt.ylabel("L2 deviation from center")
plt.legend(["Neg%.2f"%d for d in targ_val[::-1]]+["orig"]+["Pos%.2f"%d for d in targ_val])
plt.title("Distance Travel Along Given Eigen vector to achieve certain Image Distance")
plt.savefig(join(summary_dir, "noise_code_deviation.jpg"))
plt.show()
#%%
plt.figure(figsize=[10, 7])
plt.plot(dsim_arr)
plt.xlabel("Eigenvalue index")
plt.ylabel("Image Distance")
plt.legend(["Neg%.2f"%d for d in targ_val[::-1]]+["orig"]+["Pos%.2f"%d for d in targ_val])
plt.title("Achieved Image Distance Along Each Axis")
plt.savefig(join(summary_dir, "noise_space_dist_curv.jpg"))
plt.show()
#%%
plt.figure(figsize=[4, 20])
plt.matshow(dsim_arr, fignum=0)
plt.colorbar()
plt.savefig(join(summary_dir, "noise_space_distmat.jpg"))
plt.show()
#%%
space = "class"
imgall = None
xtick_col = []
dsim_col = []
vecs_col = []
img_names = []
t0 = time()
for eigid in [0, 1, 2, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40,]: # [0,1,2,3,4,5,6,7,8,10,20,30,40]:#
    if space == "class":
        tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid+1].T), dim=1)
    elif space == "noise":
        tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid+1].T, torch.zeros(1, 128).cuda()), dim=1)
    xtar_pos, ytar_pos, stepimgs_pos = find_level_step(targ_val, reftsr, tan_vec, refimg, iter=6, pos=True)
    xtar_neg, ytar_neg, stepimgs_neg = find_level_step(targ_val, reftsr, tan_vec, refimg, iter=6, pos=False)
    imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
    xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
    dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
    vecs_row = torch.tensor(xticks_row).cuda().view(-1,1) @ tan_vec + reftsr
    xtick_col.append(xticks_row)
    dsim_col.append(dsim_row)
    vecs_col.append(vecs_row.cpu().numpy())
    img_names.extend("class_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in np.linspace(-0.4, 0.4, 11))  # dsim_row)
    imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))
    print(time() - t0)
# ToPILImage()(make_grid(imgrow).cpu()).show()
ToPILImage()(make_grid(imgall, nrow=11).cpu()).show() # 20sec for 13 rows not bad
npimgs = imgall.permute([2,3,1,0]).numpy()
for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
#%%

dsim_arr = np.array(dsim_col)
plt.figure()
plt.matshow(dsim_arr)
plt.colorbar()
plt.show()
#%%
