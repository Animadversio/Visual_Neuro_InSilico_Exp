"""This script explore StyleGAN inversion of real photoes
And photo editting using it"""
import os
from os.path import join
import torch, numpy as np
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor
import matplotlib.pylab as plt
from GAN_utils import StyleGAN2_wrapper, loadStyleGAN2
from lpips import LPIPS
from load_hessian_data import load_Haverage
from torch_utils import show_imgrid, save_imgrid
def MSE(im1, im2, mask=None):
    # mask is size [sampn, H, W]
    if mask is None:
        return (im1 - im2).pow(2).mean(dim=[1,2,3])
    else:
        valnum = mask.sum([1, 2])
        diffsum = ((im1 - im2).pow(2).mean(1) * mask).sum([1, 2])
        return diffsum / valnum
#%
D = LPIPS(net="vgg")
D.cuda()
D.requires_grad_(False)
D.spatial = True
def mask_LPIPS(im1, im2, mask=None):
    diffmap = D(im1, im2) # note there is a singleton channel dimension
    if mask is None:
        return diffmap.mean([1, 2, 3])
    else:
        diffsum = (diffmap[:, 0, :, :] * mask).sum([1, 2])
        valnum = mask.sum([1, 2])
        return diffsum / valnum

#%%
imroot = r"E:\OneDrive - Washington University in St. Louis\GAN_photoedit\src"
resdir = r"E:\OneDrive - Washington University in St. Louis\GAN_photoedit\results"
SGAN = loadStyleGAN2("ffhq-512-avg-tpurun1.pt")
G = StyleGAN2_wrapper(SGAN)
G.StyleGAN.requires_grad_(False)
G.StyleGAN.eval()
alpha_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
beta_mat = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]).cuda()
def img_project(srctsr_rsz, D, imgnm="img", paramstr="", stepn=300, sampN=3, regist_freq=10,
                initvec=None, initEuclid=None, RND=None, resdir=resdir,
                wspace=False, euclid_tfm=True, tfm_target=False, hess_precon=True, ):
    if wspace:
        G.use_wspace(True)
        H, eva, evc = load_Haverage("StyleGAN2-Face512_W", descend=True)
        evctsr = torch.tensor(evc).cuda().float()
    else:
        G.use_wspace(False)
        H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
        evctsr = torch.tensor(evc).cuda().float()
    preconMat = evctsr if hess_precon else torch.eye(evctsr.shape[1]).cuda()
    if initvec is None:
        fitvec = G.sample_vector(sampN, device='cuda')
    else:
        fitvec = torch.tensor(initvec).detach().clone().cuda()
    fitcoef = fitvec @ preconMat
    fitcoef.requires_grad_(True)
    optimizer = Adam([fitcoef, ], lr=0.05, weight_decay=1E-4)  # 0.01 is good step for StyleGAN2
    if initEuclid is None:
        EuclidParam = torch.tensor([1.0, 0.0, 0.0, 0.0]).cuda().float()  # 1.0,
    else:
        EuclidParam = torch.tensor(initEuclid).cuda().float()  # 1.0,
    if not tfm_target:
        EuclidParam = EuclidParam.repeat(sampN, 1)
    if euclid_tfm:
        EuclidParam.requires_grad_(True)
        regist_optim = SGD([EuclidParam, ], lr=0.003, )  # 0.01 is good step for StyleGAN2
    # 0.0886 0.0127
    # SGD 0.05-0.01 is not good
    for step in range(stepn):
        optimizer.zero_grad()
        if euclid_tfm:  regist_optim.zero_grad()
        fitvec = fitcoef @ preconMat.T
        fittsr = G.visualize(fitvec)
        if euclid_tfm:
            if tfm_target:
                theta = torch.cat((EuclidParam[0]*
                   (alpha_mat * torch.cos(EuclidParam[1]) + beta_mat * torch.sin(EuclidParam[1])),
                               EuclidParam[2:].unsqueeze(1)), dim=1).unsqueeze(0)
                grid = F.affine_grid(theta, srctsr_rsz.size())
                validmsk = (grid[:, :, :, 0] > 0) * (grid[:, :, :, 0] < 1) * \
                           (grid[:, :, :, 1] > 0) * (grid[:, :, :, 1] < 1)
                srctsr_rsz_tfm = F.grid_sample(srctsr_rsz, grid)
                dsim = D(srctsr_rsz_tfm, fittsr, validmsk)
                MSE_err = MSE(srctsr_rsz_tfm, fittsr, validmsk)
            else:
                # Scale * Rotation * Translation
                theta = torch.cat(tuple(
                    torch.cat((EuclidParam[i, 0]*
                        (alpha_mat * torch.cos(EuclidParam[i, 1]) + beta_mat * torch.sin(EuclidParam[i, 1])),
                        EuclidParam[i, 2:].unsqueeze(1)), dim=1).unsqueeze(0)
                    for i in range(sampN)))
                grid = F.affine_grid(theta, fittsr.size())
                fittsr_tfm = F.grid_sample(fittsr, grid)
                validmsk = (grid[:, :, :, 0] > 0) * (grid[:, :, :, 0] < 1) * \
                           (grid[:, :, :, 1] > 0) * (grid[:, :, :, 1] < 1)
                dsim = D(srctsr_rsz, fittsr_tfm, validmsk)

                MSE_err = MSE(srctsr_rsz, fittsr_tfm, validmsk)
        else:
            dsim = D(srctsr_rsz, fittsr)
            MSE_err = MSE(srctsr_rsz, fittsr)
        loss = dsim + MSE_err
        loss.sum().backward()
        optimizer.step()
        if euclid_tfm and (step) % regist_freq:
            regist_optim.step()
            for i in range(10):
                regist_optim.zero_grad()
                if tfm_target:
                    theta = torch.cat((EuclidParam[0]*
                       (alpha_mat * torch.cos(EuclidParam[1]) + beta_mat * torch.sin(EuclidParam[1])),
                                   EuclidParam[2:].unsqueeze(1)), dim=1).unsqueeze(0)
                    grid = F.affine_grid(theta, srctsr_rsz.size())
                    srctsr_rsz_tfm = F.grid_sample(srctsr_rsz, grid)
                    dsim = D(srctsr_rsz_tfm, fittsr)
                    MSE_err = MSE(srctsr_rsz_tfm, fittsr)
                else:
                    theta = torch.cat(tuple(
                        torch.cat((EuclidParam[i, 0]*
                            (alpha_mat * torch.cos(EuclidParam[i, 1]) + beta_mat * torch.sin(EuclidParam[i, 1])),
                            EuclidParam[i, 2:].unsqueeze(1)), dim=1).unsqueeze(0)
                        for i in range(sampN)))
                    grid = F.affine_grid(theta, fittsr.size())
                    fittsr_tfm = F.grid_sample(fittsr, grid)
                    dsim = D(srctsr_rsz, fittsr_tfm)
                    MSE_err = MSE(srctsr_rsz, fittsr_tfm)
                regist_optim.step()
        if step % 10 == 0:
            print(
                "LPIPS %.3f MSE %.3f norm %.1f" % (dsim.min().item(), MSE_err.min().item(), fitvec.norm(dim=1).mean()))
    if RND is None: RND = np.random.randint(1000)
    if not euclid_tfm:
        im_res = show_imgrid([srctsr_rsz, fittsr.detach()], )
    elif tfm_target:
        im_res = show_imgrid([srctsr_rsz, srctsr_rsz_tfm.detach(), fittsr.detach()], )
    else:
        im_res = show_imgrid([srctsr_rsz, fittsr_tfm.detach()], )
    im_res.save(join(resdir, "%s_%s%04d.png" % (imgnm, paramstr, RND)))
    refvecs = fitvec.detach().clone()
    refimgs = fittsr.detach().clone()
    torch.save({"fitvecs":refvecs, "fitimgs":refimgs, "EuclidParams":EuclidParam.detach()},
            join(resdir, "%s_%s%04d.pt" % (imgnm, paramstr, RND)))
    return refvecs, refimgs, EuclidParam
#%%
srcimg = plt.imread(join(imroot, "joe-biden-gettyimages_crop.jpg"))
srctsr = ToTensor()(srcimg).unsqueeze(0)
srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
refvecs, refimgs, EuclidParam = img_project(srctsr_rsz, mask_LPIPS, imgnm="Biden", paramstr="",
            regist_freq=100, stepn=300, sampN=4, initEuclid=None, #initvec=None,
            wspace=False, euclid_tfm=True, hess_precon=True, tfm_target=False)
# initEuclid=[ 0.8100,  0.1050, -0.1273, -0.0267],
#%%
srcimg = plt.imread(join(imroot, "Binxu_1_crop.jpg"))
srctsr = ToTensor()(srcimg).unsqueeze(0)
srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
refvecs, refimgs, EuclidParam = img_project(srctsr_rsz, mask_LPIPS, imgnm="binxu", paramstr="",
            regist_freq=50, stepn=300, sampN=4, initEuclid=[ 0.8100,  0.1050, -0.1273, -0.0267], #initvec=None,
            wspace=False, euclid_tfm=True, hess_precon=True, tfm_target=True)
#%%
srcimg = plt.imread(join(imroot, "joe-biden-gettyimages_crop.jpg"))
srctsr = ToTensor()(srcimg).unsqueeze(0)
srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
refvecs, refimgs, EuclidParam = img_project(srctsr_rsz, mask_LPIPS, imgnm="Biden", paramstr="",
            regist_freq=100, stepn=300, sampN=4, initEuclid=None, #initvec=None,
            wspace=False, euclid_tfm=True, hess_precon=True, tfm_target=True)
#%%
srcimg = plt.imread(join(imroot, "Binxu_1_crop.jpg"))
srctsr = ToTensor()(srcimg).unsqueeze(0)
srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
refvecs, refimgs, EuclidParam = img_project(srctsr_rsz, mask_LPIPS, imgnm="Binxu", paramstr="",
            regist_freq=100, stepn=300, sampN=4, initEuclid=None, #initvec=None,
            wspace=False, euclid_tfm=True, hess_precon=True, tfm_target=False)



#%%
H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
evatsr = torch.tensor(eva).cuda().float()
evctsr = torch.tensor(evc).cuda().float()
#%%
H, eva, evc = load_Haverage("StyleGAN2-Face512_W", descend=True)
evatsr = torch.tensor(eva).cuda().float()
evctsr = torch.tensor(evc).cuda().float()
#%%
# https://www.history.com/topics/us-politics/joe-biden
# https://en.wikipedia.org/wiki/File:Official_Portrait_of_President_Reagan_1981.jpg
srcimg = plt.imread(join(imroot, "joe-biden-gettyimages_crop.jpg"))  #
# "Official_Portrait_of_President_Reagan_1981_crop.jpg"

srctsr = ToTensor()(srcimg).unsqueeze(0)
srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()

#%%
wspace = False
euclid_tfm = True
hess_precon = True
sampN = 3
stepn = 20
initvec = None
tfm_target = False
RND = None
alpha_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
beta_mat = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]).cuda()
if wspace:
    G.use_wspace(True)
    H, eva, evc = load_Haverage("StyleGAN2-Face512_W", descend=True)
    evctsr = torch.tensor(evc).cuda().float()
else:
    G.use_wspace(False)
    H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
    evctsr = torch.tensor(evc).cuda().float()
preconMat = evctsr if hess_precon else torch.eye(evctsr.shape[1]).cuda()
if initvec is None:
    fitvec = G.sample_vector(sampN, device='cuda')
else:
    fitvec = torch.tensor(initvec).detach().clone().cuda()
fitcoef = fitvec @ preconMat
fitcoef.requires_grad_(True)
optimizer = Adam([fitcoef, ], lr=0.05, weight_decay=.3E-3)  # 0.01 is good step for StyleGAN2
EuclidParam = torch.tensor([1.0, 0.0, 0.0, 0.0]).cuda().float()  # 1.0,
if not tfm_target:
    EuclidParam = EuclidParam.repeat(sampN, 1)
if euclid_tfm:
    EuclidParam.requires_grad_(True)
    regist_optim = SGD([EuclidParam, ], lr=0.005, )  # 0.01 is good step for StyleGAN2
# 0.0886 0.0127
# SGD 0.05-0.01 is not good
for step in range(stepn):
    optimizer.zero_grad()
    if euclid_tfm:  regist_optim.zero_grad()
    fitvec = fitcoef @ preconMat.T
    fittsr = G.visualize(fitvec)
    if euclid_tfm:
        if tfm_target:
            theta = torch.cat((EuclidParam[0] *
                   (alpha_mat * torch.cos(EuclidParam[1]) + beta_mat * torch.sin(EuclidParam[1])),
                               EuclidParam[2:].unsqueeze(1)), dim=1).unsqueeze(0)
            grid = F.affine_grid(theta, srctsr_rsz.size())
            srctsr_rsz_tfm = F.grid_sample(srctsr_rsz, grid)
            dsim = D(srctsr_rsz_tfm, fittsr)
            MSE_err = MSE(srctsr_rsz_tfm, fittsr)
        else:
            # Scale * Translation
            # theta = torch.cat(tuple(
            #     torch.cat((torch.diag(EuclidParam[i, 0].repeat(2)),
            #                EuclidParam[i, 1:].unsqueeze(1)), dim=1).unsqueeze(0)
            #     for i in range(sampN)))
            # Scale * Rotation * Translation
            theta = torch.cat(tuple(
                torch.cat((EuclidParam[i, 0] *
                   (alpha_mat * torch.cos(EuclidParam[i, 1]) + beta_mat * torch.sin(EuclidParam[i, 1])),
                     EuclidParam[i, 2:].unsqueeze(1)), dim=1).unsqueeze(0)
                for i in range(sampN)))
            grid = F.affine_grid(theta, fittsr.size())
            fittsr_tfm = F.grid_sample(fittsr, grid)
            dsim = D(srctsr_rsz, fittsr_tfm)
            MSE_err = MSE(srctsr_rsz, fittsr_tfm)
    else:
        dsim = D(srctsr_rsz, fittsr)
        MSE_err = MSE(srctsr_rsz, fittsr)
    loss = dsim + MSE_err
    loss.sum().backward()
    optimizer.step()
    if euclid_tfm: regist_optim.step()
    if step % 10 == 0:
        print(
            "LPIPS %.3f MSE %.3f norm %.1f" % (dsim.min().item(), MSE_err.min().item(), fitvec.norm(dim=1).mean()))
if RND is None: RND = np.random.randint(1000)
if not euclid_tfm:
    im_res = show_imgrid([srctsr_rsz, fittsr.detach()], )
elif tfm_target:
    im_res = show_imgrid([srctsr_rsz, srctsr_rsz_tfm.detach(), fittsr.detach()], )
else:
    im_res = show_imgrid([srctsr_rsz, fittsr_tfm.detach()], )
im_res.save(join(resdir, "%s_%s%04d.png" % (imgnm, paramstr, RND)))
refvecs = fitvec.detach().clone()
refimgs = fittsr.detach().clone()
#%%
refvec = fitvec.detach().clone()
refimg = fittsr.detach().clone()
with torch.no_grad():
    pos_imgs = G.visualize(refvec[0:1,:] + 0.8 * evctsr[:, :7].t())
    neg_imgs = G.visualize(refvec[0:1,:] - 0.8 * evctsr[:, :7].t())
im_pert = show_imgrid([pos_imgs, refimg[0:1,:].repeat(7,1,1,1), neg_imgs], nrow=7)
im_pert.save(join(resdir, "biden_%04d_pert.png" % RND))


#%%
