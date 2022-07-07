"""
Estimating Hessian for ReLU neural network activations using forward HVP + Lanzosc.
"""
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
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from NN_sparseness.sparse_invariance_lib import shorten_layername
from NN_sparseness.insilico_manif_configs import RN50_config, VGG16_config, manifold_config, DenseNet169_config
from collections import defaultdict
from stats_utils import saveallforms
from time import time
from hessian_eigenthings.lanczos import lanczos
from Hessian.lanczos_generalized import lanczos_generalized
from Hessian.GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, \
    compute_hessian_eigenthings, NNForwardHVPOperator, \
    GANForwardHVPOperator_multiscale, NNForwardHVPOperator_multiscale
from NN_sparseness.insilico_manif_configs import VGG16_config, RN50_config
figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\tuning"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\summary"
# %%
"""Load data"""
savedir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\gradHess"
def scan_gradHess_data(netname, suffix="_ForwardHVP_multiscale.pt", unit_range=(10, 20)):
    spect_HVP_dict = defaultdict(list)
    peakact_dict   = defaultdict(list)
    z_dict         = defaultdict(list)
    tune_dict      = defaultdict(list)
    confg = manifold_config(netname)
    for layer, unit_dict in confg.items():
        for chan in range(*unit_range):
            if unit_dict["unit_pos"] is None:
                unitstr = f"{netname}-{shorten_layername(layer)}-unit%d" % (chan)
            else:
                unit_x, unit_y = unit_dict["unit_pos"]
                unitstr = f"{netname}-{shorten_layername(layer)}-unit%d-%d-%d" % (chan, unit_x, unit_y)
            try:
                data = np.load(join(savedir, f"{unitstr}_Hess_data{suffix}.npz"), allow_pickle=True)
                if "arr_0" in data:
                    data = data["arr_0"].item()
                spect_HVP_dict[layer].append(data["eigvals"])
                peakact_dict[layer].append(data["act"])
                z_dict[layer].append(data["z"])
                if 'pert_actdict' in data:
                    tune_dict[layer].append(data['pert_actdict'])
            except FileNotFoundError:
                print(f"{unitstr} not found")
            except KeyError:
                print(list(data.keys()))
    return spect_HVP_dict, peakact_dict, z_dict, tune_dict


def summarize_spectrum(spect_HVP_dict, peakact_dict, z_dict, log=True):
    figh = plt.figure(figsize=(6, 6))
    plt.rcParams['axes.prop_cycle'] = \
        plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(spect_HVP_dict))))
    for layer, spect_col in spect_HVP_dict.items():
        spect_arr = np.array(spect_col)
        act_arr = np.array(peakact_dict[layer])
        z_arr = np.array(z_dict[layer])
        znorm = np.linalg.norm(z_arr, axis=1)
        spect_arr = np.sort(np.abs(spect_arr), axis=1)[:, ::-1]
        # norm_spect_arr = spect_arr / np.nanmax(spect_arr, axis=1, keepdims=True)# / act_arr  #
        norm_spect_arr = spect_arr / act_arr  # / znorm
        norm_spect_range = np.nanpercentile(norm_spect_arr, [25, 75], axis=0)
        if log:
            plt.semilogy(np.nanmean(norm_spect_arr, axis=0),
                     label=shorten_layername(layer), linewidth=1.2, alpha=0.5)
        else:
            plt.plot(np.nanmean(norm_spect_arr, axis=0),
                     label=shorten_layername(layer), linewidth=1.2, alpha=0.5)
        # print(f"{layer}: znorm {znorm} activation {act_arr}")
        plt.fill_between(range(len(norm_spect_range[0])),
                         norm_spect_range[0],
                         norm_spect_range[1], alpha=0.2)
        # plt.plot(np.log10(np.nanmedian(norm_spect_arr, axis=0)), label=layer)

    # plt.semilogy(data["eigvals"], alpha=0.3, label=shorten_layername(layer))
    plt.xlim([-25, 500])
    plt.ylim([1E-4, 1E-2])
    plt.legend()
    plt.title(f"Network: {netname} \n Eigen Spectrum of Hessian matrix (Forward HVP_multiscale)")
    plt.show()
    return figh


def scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict, log=True):
    df_col = []
    for layer, spect_col in spect_HVP_dict.items():
        spect_arr = np.array(spect_col)  # (n_units, n_eigvals)
        act_arr = np.array(peakact_dict[layer])  # (n_units, 1)
        z_arr = np.array(z_dict[layer]).squeeze(axis=1)  # (n_units, 4096)
        znorm = np.linalg.norm(z_arr, axis=1)
        spect_arr = np.sort(np.abs(spect_arr), axis=1)[:, ::-1]
        # norm_spect_arr = spect_arr / np.nanmax(spect_arr, axis=1, keepdims=True)# / act_arr  #
        norm_spect_arr = spect_arr / act_arr  # / znorm
        norm_spect_range = np.nanpercentile(norm_spect_arr, [25, 75], axis=0)
        top_2 = (norm_spect_arr > 3E-2).sum(axis=1)
        top_3 = (norm_spect_arr > 1E-3).sum(axis=1)
        top_4 = (norm_spect_arr > 3E-4).sum(axis=1)
        top_5 = (norm_spect_arr > 1E-4).sum(axis=1)
        print(f"{layer}: znorm {znorm.mean():.1f} activation {act_arr.mean():.1f}\n"
  f"# top 2: {top_2.mean():.1f}  top 3: {top_3.mean():.1f} top 4: {top_4.mean():.1f} top 5: {top_5.mean():.1f}")
        # if log:
        #     plt.semilogy(np.nanmean(norm_spect_arr, axis=0),
        #              label=shorten_layername(layer), linewidth=1.2, alpha=0.5)
        # else:
        #     plt.plot(np.nanmean(norm_spect_arr, axis=0),
        #              label=shorten_layername(layer), linewidth=1.2, alpha=0.5)
        # # print(f"{layer}: znorm {znorm} activation {act_arr}")
        # plt.fill_between(range(len(norm_spect_range[0])),
        #                  norm_spect_range[0],
        #                  norm_spect_range[1], alpha=0.2)
        # # plt.plot(np.log10(np.nanmedian(norm_spect_arr, axis=0)), label=layer)

    # plt.semilogy(data["eigvals"], alpha=0.3, label=shorten_layername(layer))
    # plt.xlim([-25, 500])
    # plt.ylim([1E-4, 1E-2])
    # plt.legend()
    # plt.title(f"Network: {netname} \n Eigen Spectrum of Hessian matrix (Forward HVP_multiscale)")
    # plt.show()
    return
#%%
# set color sequence,
#%%
netname = "densenet169"  #"vgg16"
spect_HVP_dict, peakact_dict, z_dict, tune_dict = \
    scan_gradHess_data(netname, suffix="_ForwardHVP_multiscale.pt")
#%%
scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict)
#%%
netname = "resnet50_linf8"  #"vgg16"
spect_HVP_dict, peakact_dict, z_dict, tune_dict = \
    scan_gradHess_data(netname, suffix="_ForwardHVP_multiscale.pt")
#%%
scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict)
#%%
netname = "resnet50_linf8"  #"vgg16"
spect_HVP_dict, peakact_dict, z_dict, tune_dict = \
    scan_gradHess_data(netname, suffix="_ForwardHVP.pt")
#%%
scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict)
#%%
netname = "resnet50"  #"vgg16"
spect_HVP_dict, peakact_dict, z_dict, tune_dict = \
    scan_gradHess_data(netname, suffix="_ForwardHVP.pt")
#%%
scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict)

#%%

netname = "vgg16"  #"vgg16"
spect_HVP_dict, peakact_dict, z_dict, tune_dict = \
    scan_gradHess_data(netname, suffix="_ForwardHVP_multiscale.pt")
#%%
figh = summarize_spectrum(spect_HVP_dict, peakact_dict, z_dict, log=True)
saveallforms(sumdir, f"{netname}_spectrum_cmp_ForwardHVP_multiscale")
figh = summarize_spectrum(spect_HVP_dict, peakact_dict, z_dict, log=False)
saveallforms(sumdir, f"{netname}_spectrum_cmp_ForwardHVP_multiscale_lin")
#%%
scatter_spectrum(spect_HVP_dict, peakact_dict, z_dict)

