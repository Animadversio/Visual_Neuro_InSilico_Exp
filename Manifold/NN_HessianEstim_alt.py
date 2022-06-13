"""
Alternative not successful methods to compute Hessian for a piecewise linear function.
(ReLU network)
"""

import matplotlib.pyplot as plt
from insilico_Exp_torch import TorchScorer
from GAN_utils import upconvGAN
G = upconvGAN()
G.cuda().eval()
G.requires_grad_(False)
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from torch_utils import show_imgrid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from os.path import join
from NN_sparseness.sparse_invariance_lib import shorten_layername
from NN_sparseness.insilico_manif_configs import RN50_config, VGG16_config
#%%
#%%
scorer = TorchScorer("resnet50_linf8")
scorer.set_unit("score", ".layer3.Bottleneck5", unit=(5, 7, 7), ingraph=True)
unitstr = "resnet50_linf8-layer3.B5-unit%d-%d-%d" % (5, 7, 7)
#%%
scorer.set_unit("score", ".layer2.Bottleneck3", unit=(5, 14, 14), ingraph=True)
unitstr = "resnet50_linf8-layer2.B3-unit%d-%d-%d" % (5, 14, 14)
#%% Attempt 0: using direct Hessian computation.
z, act, failed_try = grad_evol_unit(scorer, eps=0.5)
show_imgrid(G.visualize(z))
#%%
dz = 3.0 * torch.randn(5, 4096).cuda()
# dz.requires_grad_(True)
imgs = G.visualize(z + dz)
act = scorer.score_tsr_wgrad(imgs)
grad = torch.autograd.grad(act.sum(), z, create_graph=True, retain_graph=True)[0]
H = []
for i in trange(grad.shape[1]):
    gradgrad = torch.autograd.grad(grad[0, i], z, retain_graph=True)[0]
    H.append(gradgrad)
    break
#%%
Hmat = torch.stack(H) # H is all 0
#%%
torch.svd(Hmat)  # zero matrix
#%%
# Q, R = torch.linalg.qr(torch.randn(4096, 4096).cuda())


#%%
"""
Atempt 1, compute gradinet around the maxima and solve the linear equation
to get the Hessian matrix
"""
Q = torch.eye(4096, 4096)
figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\gradHess"
netname = "resnet50_linf8"
scorer = TorchScorer("resnet50_linf8")
for chan in range(15, 20):
    for layer, unit_dict in RN50_config.items():
        if unit_dict["unit_pos"] is None:
            scorer.set_unit("score", layer, unit=(chan, ), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            scorer.set_unit("score", layer, unit=(chan, unit_x, unit_y), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        print(unitstr)
        z, act, failed_try = grad_evol_unit(scorer)
        if failed_try > 150:
            print("failed evolution, stop")
            continue
        #%%
        batchsize = 20
        dist = 20
        gradvecs = []
        actvecs = []
        for i in trange(0, 4096, batchsize):
            pertvecs = Q[i:i + batchsize, :].clone().cuda()
            pertvecs.requires_grad_(True)
            imgs = G.visualize(z.detach() + pertvecs * dist)
            act = scorer.score_tsr_wgrad(imgs)
            act.sum().backward()
            gradvecs.append(pertvecs.grad.cpu().clone())
            actvecs.append(act.detach().cpu().clone())
        #%
        gradmat = torch.concat(gradvecs, dim=0)
        actmat = torch.concat(actvecs, dim=0)
        #%%
        Hess = gradmat / actmat.unsqueeze(1)
        U, S, V = torch.svd(Hess.cuda())
        #%%
        plt.figure(figsize=(5, 5))
        plt.semilogy(S.cpu())
        plt.title(f"SVD of Hessian matrix\n{unitstr}")
        plt.savefig(join(figdir, f"{unitstr}_SVD_spectrum.png"))
        plt.show()
        #%%
        torch.save({"z": z.detach().cpu(), "act": act.detach().cpu(),
                    "gradmat": gradmat, "actmat": actmat, "S":S, "unit_dict":unit_dict},
                   join(figdir, f"{unitstr}_Hess_data.pt"))
        scorer.cleanup()
#%%
from collections import defaultdict
"""
Summarize the spectrums 
"""
spectrum_dict = defaultdict(list)
for layer, unit_dict in RN50_config.items():
    print(layer)
    for chan in range(15, 20):  #
        if unit_dict["unit_pos"] is None:
            scorer.set_unit("score", layer, unit=(chan, ), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            scorer.set_unit("score", layer, unit=(chan, unit_x, unit_y), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        try:
            data = torch.load(join(figdir, f"{unitstr}_Hess_data.pt"))
        except FileNotFoundError:
            print(f"{unitstr} not found")
            continue
        # z = data["z"].cuda()
        Hess = data["gradmat"] # / data["actmat"].unsqueeze(1)
        U, S, V = torch.svd(Hess.cuda())
        # S = data["S"].cpu()
        normS = (S / S.max()).cpu().numpy()
        spectrum_dict[layer].append(normS)
        print((normS > 1E-2).sum(), (normS > 1E-3).sum(), )

#%%
plt.figure(figsize=[6, 6])
for layer, spectrum in spectrum_dict.items():
    Sarr = np.array(spectrum_dict[layer])
    plt.semilogy(Sarr.mean(axis=0), label=shorten_layername(layer))
plt.xlim([0, 800])
plt.ylim([1e-4, 1])
plt.legend()
plt.show()
#%% Compute the Hessian trace by  a random orthogonal matrix
"""
Atempt 2, compute tuning curves arround the maxima. 
"""
netname = "resnet50_linf8"
for chan in range(11, 15):
    for layer, unit_dict in RN50_config.items():
        if unit_dict["unit_pos"] is None:
            scorer.set_unit("score", layer, unit=(chan, ), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            scorer.set_unit("score", layer, unit=(chan, unit_x, unit_y), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        print(unitstr)
        #%%
        z, failed_try = grad_evol_unit(scorer)
        if failed_try > 120:
            print("failed evolution, stop")
            continue
        #%%
        batchsize = 120
        dist = 100
        act_pos = []
        act_neg = []
        for i in trange(0, 4096, batchsize):
            pertvecs = Q[i:i + batchsize, :]
            imgs = G.visualize(z.detach() + pertvecs * dist)
            act = scorer.score_tsr_wgrad(imgs)
            act_pos.append(act)
            imgs = G.visualize(z.detach() - pertvecs * dist)
            act = scorer.score_tsr_wgrad(imgs)
            act_neg.append(act)
        act_pos_vec = torch.concat(act_pos, dim=0).cpu()
        act_neg_vec = torch.concat(act_neg, dim=0).cpu()
        #%%
        act_c = scorer.score_tsr_wgrad(G.visualize(z.detach())).cpu()
        #%%
        tunecurv = torch.stack([act_pos_vec, act_c *torch.ones(4096), act_neg_vec], )
        df = pd.DataFrame(tunecurv.numpy().T, columns=["pos", "cent", "neg"])

        pert_mean = (act_neg_vec.mean() + act_pos_vec.mean()) / 2
        pert_std = (act_neg_vec.std() + act_pos_vec.std()) / 2
        pert_ratio = (pert_mean / act_c).cpu().item()
        pert_std_ratio = (pert_std / act_c).cpu().item()
        #%
        plt.figure(figsize=(5, 6))
        sns.violinplot(data=df, inner="box")
        plt.title(f"Violin plot of activations\n{unitstr}")
        plt.savefig(join(figdir, f"{unitstr}_qr4096_violin.png"))
        plt.show()
        plt.figure(figsize=(4, 6))
        plt.plot(tunecurv.numpy(), alpha=0.3)
        plt.title(f"Tuning curve\n{unitstr}\n{pert_ratio:.3f} +/- {pert_std_ratio:.3f}")
        plt.savefig(join(figdir, f"{unitstr}_qr4096_tunecurve.png"))
        plt.show()
        #%%
        torch.save({"tunecurv":tunecurv, "dist":dist, "z":z, "unitstr":unitstr}, join(figdir, f"{unitstr}_qr4096.pth"))
#%%
unitvec = torch.randn(1, 4096)
ticks = torch.linspace(-150, 150, 21)
dz = (ticks.unsqueeze(1) @ unitvec / unitvec.norm()).cuda()
imgs = G.visualize(z.detach() + dz)
act = scorer.score_tsr_wgrad(imgs)
plt.plot(ticks.numpy(), act.detach().cpu().numpy())
plt.show()
#%%
