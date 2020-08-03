from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.optim import SGD, Adam
import numpy as np
import pandas as pd
from os.path import join
from imageio import imread
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

alpha = 5 # relative weight
#%%
BGAN = BigGAN.from_pretrained("biggan-deep-256")
BGAN.cuda()
BGAN.eval()
for param in BGAN.parameters():
    param.requires_grad_(False)
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\Hessian"
data = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz")
evc_clas = torch.from_numpy(data['eigvects_clas_avg']).cuda()
evc_nois = torch.from_numpy(data['eigvects_nois_avg']).cuda()
evc_all = torch.from_numpy(data['eigvects_avg']).cuda()
#%%
from imageio import imread
target = imread("block042_thread000_gen_gen041_001030.bmp")
target_tsr = torch.from_numpy(target / 255.0).permute([2, 0, 1]).unsqueeze(0)
target_tsr = target_tsr.float().cuda()
#%%
"""
Provide a bunch of metric, 
"""
#%% Official CMAES algorithm
import cma
optimizer = cma.CMAEvolutionStrategy(128 * [0.0], 0.06)
fixnoise = truncated_noise_sample(1,128)
noise_vec = torch.from_numpy(fixnoise)
#%%
import tqdm
for i in tqdm.trange(50):
    codes = optimizer.ask()
    # boundary handling

    # evaluate and passing values
    codes_tsr = torch.from_numpy(np.array(codes)).float()
    latent_code = torch.cat((noise_vec.repeat(18, 1), codes_tsr), dim=1).cuda()
    with torch.no_grad():
        imgs = BGAN.generator(latent_code, 0.7)
        imgs = (imgs + 1.0) / 2.0
        dsims = ImDist(imgs, target_tsr).squeeze()
        L1dsim = (imgs - target_tsr).abs().mean([1,2,3])
    scores = dsims.detach().cpu().numpy()
    L1score = L1dsim.detach().cpu().numpy()
    optimizer.tell(codes, scores + L1score)
    # optimizer.result_pretty()
    print("step %d dsim %.3f L1 %.3f (norm %.2f)" % (i, scores.mean(), L1score.mean(), codes_tsr.norm(dim=1).mean()))

#%%
xmean_tsr = torch.from_numpy(optimizer.mean).unsqueeze(0).float()
final_latent = torch.cat((noise_vec, xmean_tsr), dim=1).cuda()
fit_img = BGAN.generator(final_latent, 0.7)
fit_img = (fit_img + 1) / 2.0
ToPILImage()(fit_img[-1,:,:,:].cpu()).show()

#%%
#%%  CMAES Basin on Noise + Class space
"""
Here implements an illustrative version of Basin CMA algorithm. 
A few technical decision

* Separate CMA optimizer for noise and class space.
* Initialize CMA from 0 in class space, on shell (truncnorm(-2,2)) in noise space
*  
* Use Hessian basis parametrization in Adam. Can switch on and off
* 

Currently the computational cost for 18 point is still quite high. (seems same for original algorithm)
"""
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_invert\BasinCMA"
import cma
fixnoise = truncated_noise_sample(1, 128)
optimizer = cma.CMAEvolutionStrategy(128 * [0.0], 0.06)
optimizer2 = cma.CMAEvolutionStrategy(fixnoise, 0.2)
# noise_vec = torch.from_numpy(fixnoise)
RND = np.random.randint(1E6)
#%%
cmasteps = 30
gradsteps = 40
batch_size = 4
import tqdm
for i in tqdm.trange(cmasteps, desc="CMA steps"):
    codes = optimizer.ask()
    noise_codes = optimizer2.ask()
    # TODO: boundary handling by projection in code space

    # evaluate the cma proposed codes `latent_code` at first.
    codes_tsr = torch.from_numpy(np.array(codes)).float()
    noise_tsr = torch.from_numpy(np.array(noise_codes)).float()
    # latent_code = torch.cat((noise_vec.repeat(18, 1), codes_tsr), dim=1).cuda()
    latent_code = torch.cat((noise_tsr, codes_tsr), dim=1).cuda()
    with torch.no_grad():
        imgs = BGAN.generator(latent_code, 0.7)
        imgs = (imgs + 1.0) / 2.0
        dsims = ImDist(imgs, target_tsr).squeeze()
        L1dsim = (imgs - target_tsr).abs().mean([1,2,3])
    scores = dsims.detach().cpu().numpy()
    L1score = L1dsim.detach().cpu().numpy()
    print("step %d pre-ADAM dsim %.3f L1 %.3f (norm %.2f noise norm %.2f)" % (i, scores.mean(), L1score.mean(), codes_tsr.norm(dim=1).mean(), noise_tsr.norm(dim=1).mean()))
    # ADAM optimization from the cma proposed codes `latent_code` batch by batch
    codes_post = np.zeros_like(np.hstack((noise_codes, codes)))
    scores_post = np.zeros_like(scores)
    L1score_post = np.zeros_like(L1score)
    csr = 0
    # pbar = tqdm.tqdm(total=len(codes), initial=csr, desc="batchs")
    while csr < len(codes):
        csr_end = min(len(codes), csr + batch_size)
        # codes_batch = codes_tsr[csr:csr_end, :].detach().clone().requires_grad_(True)
        coef_batch = (latent_code[csr:csr_end, :] @ evc_all).detach().clone().requires_grad_(True)
        optim = Adam([coef_batch], lr=0.05, )
        for step in range(gradsteps):  # tqdm.trange(gradsteps, desc="ADAM steps"):
            optim.zero_grad()
            # latent_code = torch.cat((noise_vec.repeat(codes_batch.shape[0], 1), codes_batch), dim=1).cuda()
            latent_batch = coef_batch @ evc_all.T
            imgs = BGAN.generator(latent_batch, 0.7)
            imgs = (imgs + 1.0) / 2.0
            dsims = ImDist(imgs, target_tsr).squeeze()
            L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
            loss = (dsims + L1dsim).sum()
            loss.backward()
            optim.step()
            if (step + 1) % 10 == 0:
                print("step %d dsim %.3f L1 %.3f" % (step, scores_batch.mean(), L1score_batch.mean(),))
        code_batch = (coef_batch @ evc_all.T).detach().cpu().numpy()
        scores_batch = dsims.detach().cpu().numpy()
        L1score_batch = L1dsim.detach().cpu().numpy()
        codes_post[csr:csr_end, :] = code_batch
        scores_post[csr:csr_end] = scores_batch
        L1score_post[csr:csr_end] = L1score_batch
        csr = csr_end
        # pbar.update(csr_end - csr)
    # pbar.close()
    # Use the ADAM optimized scores as utility for `latent_code` and do cma update
    print("step %d post-ADAM dsim %.3f L1 %.3f (norm %.2f, norm %.2f)" % (
            i, scores_post.mean(), L1score_post.mean(),
            np.linalg.norm(codes_post[:,128:], axis=1).mean(), np.linalg.norm(codes_post[:,:128], axis=1).mean()))
    optimizer.tell(codes, scores_post + L1score_post)
    optimizer2.tell(noise_codes, scores_post + L1score_post)
    # optimizer.result_pretty()
#%%
idx = np.argsort((L1score_post + scores_post))
# class_tsr = torch.from_numpy(optimizer.mean).unsqueeze(0).float()
# noise_tsr = torch.from_numpy(optimizer2.mean).unsqueeze(0).float()
# final_latent = torch.cat((noise_tsr, class_tsr), dim=1).cuda()
final_latent = torch.from_numpy(codes_post[idx[0]]).unsqueeze(0).float().cuda()
fit_img = BGAN.generator(final_latent, 0.7)
fit_img = (fit_img + 1) / 2.0
ToPILImage()(fit_img[-1,:,:,:].cpu()).show()
CMAfitimg = ToPILImage()(fit_img[-1,:,:,:].cpu())
CMAfitimg.save(join(savedir, "CMA_final%06d.jpg"%RND))
#%%
final_gradsteps = 400
codes_batch = torch.from_numpy(codes_post[idx[:4]]).float().cuda()
coef_batch = (codes_batch @ evc_all).detach().clone().requires_grad_(True)
optim = Adam([coef_batch], lr=0.03, )
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
#%
for step in range(final_gradsteps):#tqdm.trange(gradsteps, desc="ADAM steps"):
    optim.zero_grad()
    # latent_code = torch.cat((noise_vec.repeat(codes_batch.shape[0], 1), codes_batch), dim=1).cuda()
    latent_code = coef_batch @ evc_all.T
    imgs = BGAN.generator(latent_code, 0.7)
    imgs = (imgs + 1.0) / 2.0
    dsims = ImDist(imgs, target_tsr).squeeze()
    L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
    loss = (dsims + L1dsim).sum()
    loss.backward()
    optim.step()
    if (step + 1) % 10 ==0:
        print("step %d dsim %.3f L1 %.3f (norm %.2f)" % (
            step, dsims.mean().item(), L1dsim.mean().item(), latent_code.norm(dim=1).mean().item()))
scores_final = dsims.detach().cpu().numpy()
L1score_final = L1dsim.detach().cpu().numpy()
#%%
finalimg = ToPILImage()(make_grid(imgs[:,:,:,:].cpu()))
finalimg.save(join(savedir, "refinefinal%06d.jpg"%RND))
finalimg.show()
