import torch
import numpy as np
import matplotlib.pylab as plt
from os.path import join
import torch.nn.functional as F
#%%
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
import torchvision as tv
# VGG = tv.models.vgg16(pretrained=True)
alexnet = tv.models.alexnet(pretrained=True).cuda()
for param in alexnet.parameters():
    param.requires_grad_(False)
#%% Test the new forward mode HVP computation
#   Forward differencing method. One Free parameter is the "eps" i.e. the norm of perturbation
#   to apply on the central vector. Too small norm of this will make it numerical unstable. too
#   large norm will make it in precise. So here we used multiple eps in computation and see how
#   well they corresponds to each other.
#   formula:
#      H@v = (g(x+eps*v) - g(x-eps*v)) / (2*eps)
savedir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
corrmat_col = []
for i in range(500):
    feat = 5 * torch.tensor(np.random.randn(4096)).float().cuda()
    feat.requires_grad_(False)
    basenorm = feat.norm()
    vect = torch.tensor(np.random.randn(4096)).float().cuda()
    vecnorm = vect.norm()
    vect = vect / vecnorm
    vect.requires_grad_(False)

    hvp_col = []
    for eps in [100, 50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ]:
        perturb_vecs = feat.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).cuda() * vect.detach()
        perturb_vecs.requires_grad_(True)
        img = G.visualize(perturb_vecs)
        resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
        obj = alexnet.features[:10](resz_img)[:, :, 6, 6].mean()   # esz_img.std()
        ftgrad_both = torch.autograd.grad(obj, perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)
        hvp = (ftgrad_both[0][0, :] - ftgrad_both[0][1, :]) / (2 * eps)
        hvp_col.append(hvp)
        # print(hvp)

    hvp_arr = torch.cat(tuple(hvp.unsqueeze(0) for hvp in hvp_col), dim=0)
    corrmat = np.corrcoef(hvp_arr.cpu().numpy())
    corrmat_col.append(corrmat)
    print("Trial ", i)
#%%
# np.mean([corrmat[0][np.newaxis,:,:] for corrmat in corrmat_col], axis=0)
corrmat_avg = np.mean(np.concatenate([corrmat[np.newaxis,:,:] for corrmat in corrmat_col], axis=0), axis=0)
plt.matshow(corrmat_avg, cmap=plt.cm.jet)
plt.yticks(range(12), labels=[50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ])
plt.xticks(range(12), labels=[50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ])
plt.ylim(top = -0.5, bottom=11.5)
plt.xlim(left = -0.5, right=11.5)
plt.xlabel("Perturb Vector Norm (Base Vector Norm 300)")
plt.suptitle("Correlation of HVP result (500 Trials)\nusing different EPS in forward differencing\n")
plt.colorbar()
plt.savefig(join(savedir, "HVP_corr_TrialAvg500.jpg"))
plt.show()
#%%
"""
So the result shows using a norm proportion 1E-2 of perturbation vs basis vector will have relatively coherent result
"""
