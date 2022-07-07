"""
Estimating Hessian for ReLU neural network activations using forward HVP + Lanzosc.
"""
import matplotlib.pyplot as plt
from insilico_Exp_torch import TorchScorer
from GAN_utils import upconvGAN
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
from NN_sparseness.insilico_manif_configs import RN50_config, VGG16_config, manifold_config, DenseNet169_config
from collections import defaultdict
from stats_utils import saveallforms
figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\tuning"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\summary"
#%%
G = upconvGAN()
G.cuda().eval()
G.requires_grad_(False)

from Hessian.load_hessian_data import load_Haverage
H, eva, evc = load_Haverage("fc6GAN")
#%%
def grad_evol_unit(scorer, eps=0.5):
    z = torch.randn(1, 4096).cuda()
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=1e-1)
    cnt = 0
    failed_try = 0
    while cnt < 150:
        imgs = G.visualize(z)
        act = scorer.score_tsr_wgrad(imgs)
        loss = -act
        if torch.isclose(act, torch.zeros(1).cuda()).any():
            z.data = z.data + eps * torch.randn(1, 4096).cuda()
            failed_try += 1
            if failed_try > 50:
                z.data = 1.5 * torch.randn(1, 4096).cuda()
            if failed_try > 1500:
                print("failed evolution, stop")
                break
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
        if cnt % 10 == 0 and cnt > 0:
            print(cnt, act.item())
    return z, act, failed_try


def perturb_activation(G, scorer, z, EPSs, sample_size=100):
    if isinstance(EPSs, float):
        EPSs = [EPSs]

    act_dict = {}
    for EPS in EPSs:
        dist = z.norm(dim=1).detach() * EPS
        with torch.no_grad():
            dz = torch.randn([sample_size, 4096]).cuda()
            dz = dz / dz.norm(dim=1, keepdim=True) * dist.unsqueeze(1)
            acts = scorer.score_tsr_wgrad(G.visualize(z.detach() + dz)).cpu()
        print(f"EPS={EPS} (D={dist.item():.2f}) mean={acts.mean():.2f} std={acts.std():.2f} median={torch.median(acts):.2f}"
              f" [{acts.min():.2f}-{acts.max():.2f}]")
        act_dict[EPS] = acts.numpy()
    return act_dict
#%%
from time import time
from hessian_eigenthings.lanczos import lanczos
from Hessian.lanczos_generalized import lanczos_generalized
from Hessian.GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, \
    compute_hessian_eigenthings, NNForwardHVPOperator, \
    GANForwardHVPOperator_multiscale, NNForwardHVPOperator_multiscale
from NN_sparseness.insilico_manif_configs import VGG16_config, RN50_config
#%%
"""
Attempt 3, using Lanczos iteration and HVP to compute the approximate Hessian
"""
#%%
savedir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\gradHess"
netname = "densenet169" #"vgg16"
scorer = TorchScorer(netname, rawlayername=False)
confg = manifold_config(netname)
# scorer.set_unit("score", '.layer3.Bottleneck4', unit=(55, 7, 7,), ingraph=True)
for chan in range(10, 20):
    for layer, unit_dict in confg.items():#RN50_config.items():
        if unit_dict["unit_pos"] is None:
            scorer.set_unit("score", layer, unit=(chan,), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            scorer.set_unit("score", layer, unit=(chan, unit_x, unit_y), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        #%%
        z, act, failed_try = grad_evol_unit(scorer, eps=0.5)
        if failed_try > 1500:
            print("failed evolution, stop")
            continue
        pert_actdict = perturb_activation(G, scorer, z,
                      EPSs=[1, 0.5, 0.2, 1E-1, 1E-2, ], sample_size=50)
        #%%
        activHVP = GANForwardHVPOperator_multiscale(G, z[0].detach(),
                                         lambda x: scorer.score_tsr_wgrad(x),
                                         preprocess=lambda x: x, EPS=0.2,
                                         scalevect=(4.0, 2.0, 1.0, 0.5))
        # activHVP = NNForwardHVPOperator_multiscale(net, cent, EPS=5E-1,
        #                            scalevect=(4.0, 2.0, 1.0, 0.5))
        t0 = time()
        eigvals, eigvects = lanczos(activHVP, num_eigenthings=2000, use_gpu=True)
        print(time() - t0)  # 146sec for 2000 eigens
        eigvals = eigvals[::-1]
        eigvects = eigvects[::-1, :]
        np.savez(join(savedir, f"{unitstr}_Hess_data_ForwardHVP_multiscale.pt"),
                  **{"z": z.detach().cpu().numpy(),
                   "act": act.detach().cpu().numpy(),
                   "eigvals": eigvals, "eigvects": eigvects,
                   "pert_actdict": pert_actdict
                   }, )
        #%%
        plt.figure(figsize=(5, 5))
        plt.semilogy(np.sort(np.abs(eigvals))[::-1])
        plt.title(f"SVD of Hessian matrix (Forward HVP)\n{unitstr}")
        plt.savefig(join(savedir, f"{unitstr}_SVD_spectrum_ForwardHVP_multiscale.png"))
        plt.show()
        #%%
        scorer.cleanup()

#%%
savedir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\gradHess"
spect_HVP_dict = defaultdict(list)
peakact_dict   = defaultdict(list)
z_dict         = defaultdict(list)
netname = "densenet169" #"vgg16"
confg = manifold_config(netname)

for layer, unit_dict in confg.items():
    for chan in range(10, 20):
        if unit_dict["unit_pos"] is None:
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        try:
            data = np.load(join(savedir, f"{unitstr}_Hess_data_ForwardHVP_multiscale.pt.npz"), allow_pickle=True)
            spect_HVP_dict[layer].append(data["eigvals"])
            peakact_dict[layer].append(data["act"])
            z_dict[layer].append(data["z"])
        except FileNotFoundError:
            print(f"{unitstr} not found")


#%%
netname = "vgg16"# "resnet50_linf8"
spect_HVP_dict = defaultdict(list)
peakact_dict   = defaultdict(list)
z_dict         = defaultdict(list)
for layer, unit_dict in VGG16_config.items():#RN50_config.items():
    for chan in range(10, 20):
        if unit_dict["unit_pos"] is None:
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        try:
            data = np.load(join(savedir, f"{unitstr}_Hess_data_ForwardHVP_multiscale.pt.npz"), allow_pickle=True)
            spect_HVP_dict[layer].append(data["eigvals"])
            peakact_dict[layer].append(data["act"])
            z_dict[layer].append(data["z"])
        except FileNotFoundError:
            print(f"{unitstr} not found")
#%%
netname = "vgg16"# "resnet50_linf8"
spect_HVP_dict = defaultdict(list)
peakact_dict   = defaultdict(list)
z_dict         = defaultdict(list)
tune_dict      = defaultdict(list)
for layer, unit_dict in VGG16_config.items():#RN50_config.items():
    for chan in range(10, 20):
        if unit_dict["unit_pos"] is None:
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        try:
            data = np.load(join(savedir, f"{unitstr}_Hess_data_ForwardHVP_multiscale.pt.npz"), allow_pickle=True)
            spect_HVP_dict[layer].append(data["eigvals"])
            peakact_dict[layer].append(data["act"])
            z_dict[layer].append(data["z"])
            tune_dict[layer].append(data['pert_actdict'].item())
        except FileNotFoundError:
            print(f"{unitstr} not found")
#%%
"""
Plot activation as a function of random perturbation size
"""
for layer, unit_dict in VGG16_config.items():
    plt.figure(figsize=(5, 5))
    actdicts = tune_dict[layer]
    actvec = peakact_dict[layer]
    for act, actdict in zip(actvec, actdicts):
        norm_actdict = {k: v / act for k, v in actdict.items()}
        # sns.scatterplot(data=norm_actdict, palette="Set1",)
        sns.stripplot(data=pd.DataFrame(norm_actdict),
                      palette="Set1", alpha=0.25, jitter=True)
        # for k, v in actdict.items():
        #     plt.scatter(k * np.ones_like(v), v, label=k)
    plt.ylim(-0.05, 1.05)
    plt.title(f"Activation at perturbation\n{layer}")
    plt.xlabel("EPS (fraction of |z|)")
    plt.ylabel("Activation normed to peak")
    saveallforms(sumdir, f"{netname}_{layer}_act_pert_multiscale")
    # plt.legend()
    plt.show()
#%%
"""
Plot Hessian spectrum 
"""
plt.figure(figsize=(6, 6))
for layer, spect_col in spect_HVP_dict.items():
    spect_arr = np.array(spect_col)
    act_arr = np.array(peakact_dict[layer])
    z_arr = np.array(z_dict[layer])
    znorm = np.linalg.norm(z_arr, axis=1)
    spect_arr = np.sort(np.abs(spect_arr), axis=1)[:, ::-1]
    # norm_spect_arr = spect_arr / np.nanmax(spect_arr, axis=1, keepdims=True)# / act_arr  #
    norm_spect_arr = spect_arr / act_arr #/ znorm
    norm_spect_range = np.nanpercentile(norm_spect_arr, [25, 75], axis=0)
    plt.semilogy(np.nanmean(norm_spect_arr, axis=0),
                 label=shorten_layername(layer), linewidth=2, alpha=0.7)
    print(f"{layer}: znorm {znorm} activation {act_arr}")
    plt.fill_between(range(len(norm_spect_range[0])),
                     norm_spect_range[0],
                     norm_spect_range[1], alpha=0.2)
    # plt.plot(np.log10(np.nanmedian(norm_spect_arr, axis=0)), label=layer)

# plt.semilogy(data["eigvals"], alpha=0.3, label=shorten_layername(layer))
plt.xlim([-25, 500])
plt.ylim([1E-4, 1E-2])
plt.legend()
plt.title(f"Network: {netname} \n Eigen Spectrum of Hessian matrix (Forward HVP_multiscale)")
saveallforms(sumdir, f"{netname}_spectrum_cmp_ForwardHVP_multiscale2")
plt.show()
#%%
"""Alternative ways to summarize the spectrum"""
plt.figure(figsize=(6, 6))
for layer, spect_col in spect_HVP_dict.items():
    spect_arr = np.array(spect_col)
    act_arr = np.array(peakact_dict[layer])
    z_arr = np.array(z_dict[layer])
    znorm = np.linalg.norm(z_arr, axis=1)
    spect_arr = np.sort(np.abs(spect_arr), axis=1)[:, ::-1]
    # norm_spect_arr = spect_arr / np.nanmax(spect_arr, axis=1, keepdims=True)# / act_arr  #
    norm_spect_arr = spect_arr / act_arr #/ znorm
    norm_spect_range = np.nanpercentile(norm_spect_arr, [25, 75], axis=0)
    plt.plot(np.nanmean(norm_spect_arr, axis=0),
                 label=shorten_layername(layer), linewidth=2, alpha=0.7)
    # print(f"{layer}: znorm {znorm} activation {act_arr}") # sanity check
    plt.fill_between(range(len(norm_spect_range[0])),
                     norm_spect_range[0],
                     norm_spect_range[1], alpha=0.2)
    # plt.plot(np.log10(np.nanmedian(norm_spect_arr, axis=0)), label=layer)

# plt.semilogy(data["eigvals"], alpha=0.3, label=shorten_layername(layer))
plt.xlim([-25, 500])
plt.ylim([1E-8, 0.005])
plt.legend()
plt.title(f"Network: {netname} \n Eigen Spectrum of Hessian matrix (Forward HVP_multiscale)")
saveallforms(sumdir, f"{netname}_spectrum_cmp_ForwardHVP_multiscale2_lin")
plt.show()


#%%
# netname = "resnet50_linf8"
# netname = "resnet50"
# netname = "densenet121"
netname = "vgg16"
scorer = TorchScorer(netname, rawlayername=False)
scorer.set_unit("score", '.layer3.Bottleneck4', unit=(55, 7, 7,), ingraph=True)
for chan in range(10, 20):
    for layer, unit_dict in VGG16_config.items():#RN50_config.items():
        if unit_dict["unit_pos"] is None:
            scorer.set_unit("score", layer, unit=(chan,), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            scorer.set_unit("score", layer, unit=(chan, unit_x, unit_y), ingraph=True)
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        #%%
        z, act, failed_try = grad_evol_unit(scorer, eps=0.5)
        if failed_try > 1500:
            print("failed evolution, stop")
            continue
        #%%
        activHVP = GANForwardHVPOperator(G, z[0].detach(),
                                         lambda x: scorer.score_tsr_wgrad(x).mean(),
                                         preprocess=lambda x: x, EPS=1E-2,)
        t0 = time()
        eigvals, eigvects = lanczos(activHVP, num_eigenthings=2000, use_gpu=True)
        print(time() - t0)  # 146sec for 2000 eigens
        eigvals = eigvals[::-1]
        eigvects = eigvects[::-1, :]
        np.savez(join(figdir, f"{unitstr}_Hess_data_ForwardHVP.pt"),{"z": z.detach().cpu().numpy(),
                   "act": act.detach().cpu().numpy(),
                  "eigvals": eigvals, "eigvects": eigvects, },
                   )
        #%%
        plt.figure(figsize=(5, 5))
        plt.semilogy(eigvals)
        plt.title(f"SVD of Hessian matrix (Forward HVP)\n{unitstr}")
        plt.savefig(join(figdir, f"{unitstr}_SVD_spectrum_ForwardHVP.png"))
        plt.show()
        #%%
        scorer.cleanup()

#%%
eigvals = eigvals[::-1]
plt.figure()
plt.semilogy(np.abs(eigvals), label="conv3_relu", linewidth=2, alpha=0.7)
plt.show()
# VGG16_config
#%%
netname = "vgg16"# "resnet50_linf8"
spect_HVP_dict = defaultdict(list)
peakact_dict   = defaultdict(list)
for layer, unit_dict in VGG16_config.items():#RN50_config.items():
    for chan in range(10, 20):
        if unit_dict["unit_pos"] is None:
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
        else:
            unit_x, unit_y = unit_dict["unit_pos"]
            unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
        try:
            data = np.load(join(figdir, f"{unitstr}_Hess_data_ForwardHVP.pt.npz"), allow_pickle=True)
            data = data['arr_0'].item()
            spect_HVP_dict[layer].append(data["eigvals"])
            peakact_dict[layer].append(data["act"])
        except FileNotFoundError:
            print(f"{unitstr} not found")

#%%
plt.figure(figsize=(6, 6))
for layer, spect_col in spect_HVP_dict.items():
    spect_arr = np.array(spect_col)
    act_arr = np.array(peakact_dict[layer])
    norm_spect_arr = spect_arr / np.nanmax(spect_arr, axis=1, keepdims=True)# / act_arr  #
    norm_spect_arr = spect_arr
    norm_spect_range = np.nanpercentile(norm_spect_arr, [25, 75], axis=0)
    plt.semilogy(np.nanmean(norm_spect_arr, axis=0),
                 label=shorten_layername(layer), linewidth=2, alpha=0.7)
    # plt.fill_between(range(len(norm_spect_range[0])),
    #                  norm_spect_range[0],
    #                  norm_spect_range[1], alpha=0.2)
    # plt.plot(np.log10(np.nanmedian(norm_spect_arr, axis=0)), label=layer)

# plt.semilogy(data["eigvals"], alpha=0.3, label=shorten_layername(layer))
plt.xlim([0, 300])
plt.ylim([1E-3, 5E-1])
plt.xlim([0, 2000])
plt.ylim([1E-7, 5E-1])
plt.legend()
plt.title(f"Network: {netname} \n Eigen Spectrum of Hessian matrix (Forward HVP)")
saveallforms(sumdir, f"{netname}_spectrum_cmp_ForwardHVP")
plt.show()







#%% Dev zone
#%% Experiment on the spatial scale of hessian
netname = "vgg16"
scorer = TorchScorer(netname, rawlayername=True)
scorer.set_unit("score", '.features.ReLU6', unit=(55, 56, 56,), ingraph=True) # .features.ReLU29
#%%
z, act, failed_try = grad_evol_unit(scorer, eps=0.5)
#%%
eigvals_dict = {}
eigvects_dict = {}
for EPS in [1E-1, 1E-2, 1E-3, 1E-4, 1E-5]:
    activHVP = GANForwardHVPOperator(G, z[0].detach(),
                     lambda x: scorer.score_tsr_wgrad(x).mean(),
                     preprocess=lambda x: x, EPS=EPS,)
    # activHVP.apply(1*torch.randn(4096).requires_grad_(False).cuda())
    t0 = time()
    try:
        eigvals, eigvects = lanczos(activHVP, num_eigenthings=2000, use_gpu=True)
        print("%.1e took  %.3f sec"%(EPS, time() - t0))  # 146sec for 2000 eigens
        eigvals_dict[EPS] = eigvals
        eigvects_dict[EPS] = eigvects
    except:
        print("%.1e failed"%(EPS))
#%%
plt.figure(figsize=(5, 5))
for EPS, eigvals in eigvals_dict.items():
    # if EPS != 1E-1:
    #     continue
    plt.semilogy(np.sort(np.abs(eigvals), )[::-1], label=f"EPS={EPS:.1e}")
plt.semilogy(np.abs(eva[::-1])[:2000], label=f"GAN")
plt.legend()
plt.title(f"SVD of Hessian matrix (Forward HVP)\n spatial scale comparison")
saveallforms(sumdir, f"{netname}_spectrum_ForwardHVP_spatial_comparison.png")
plt.show()
#%% Distribution of activation at a certain distance
pert_actdict = perturb_activation(G, scorer, z, EPSs=[1, 0.5, 0.2, 1E-1, 1E-2,], sample_size=50)
#%%
EPS = 2E-1
dist = z.norm(dim=1).detach() * EPS
dz = torch.randn([100, 4096]).cuda()
dz = dz / dz.norm(dim=1, keepdim=True) * dist.unsqueeze(1)
acts = scorer.score_tsr_wgrad(G.visualize(z.detach() + dz)).cpu()
print(acts.mean(), acts.std(), acts.min(), acts.max(), acts)
#%%
EPS = 1E-4
dist = z.norm(dim=1).detach() * EPS
dz = torch.randn([50, 4096]).cuda()
dz = dz / dz.norm(dim=1, keepdim=True) * dist.unsqueeze(1)
dz.requires_grad_(True)
acts = scorer.score_tsr_wgrad(G.visualize(z.detach() + dz)).cpu()
acts.sum().backward()
#%%
dzgrad = dz.grad
dzgrad.unique(dim=0).shape