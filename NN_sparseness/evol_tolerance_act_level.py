
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr
from stats_utils import saveallforms
from NN_sparseness.visualize_sparse_inv_example import *
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.sparse_plot_utils import scatter_density_grid
from torchvision.transforms.functional import rotate, affine, resize, center_crop
#%%
expdir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer4-Btn2-10_rf\summary_max_abinit"
#%%
from insilico_Exp_torch import TorchScorer
netname = "resnet50_linf8"
# model, model_full = load_featnet("resnet50_linf8")
# model.eval().cuda()
# exampledir = join(rootdir, f"tuning_map_examples_{netname}")
#%%
"""
how the unit level invariance depend on the activation level of the image
"""
def measure_invariance(model, layer_long, unit_id, imgtsrs, netname=netname, plot=True):
    if isinstance(imgtsrs, list):
        imgtsr = torch.stack(imgtsrs)
    else:
        imgtsr = imgtsrs
    imgstr_sml = affine(imgtsr, 0, [0, 0], scale=0.75, shear=[0, 0], fill=0.5)
    imgstr_lrg = affine(imgtsr, 0, [0, 0], scale=1.5, shear=[0, 0], fill=0.5)
    imgstr_lft = affine(imgtsr, 0, [45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rgt = affine(imgtsr, 0, [-45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rot = rotate(imgtsr, 30, expand=False, center=None, fill=0.5)
    imgstr_rot_rev = rotate(imgtsr, -30, expand=False, center=None, fill=0.5)
    augimgtsr = torch.stack([imgtsr,
                             imgstr_lrg,
                             imgstr_sml,
                             imgstr_rgt,
                             imgstr_lft,
                             imgstr_rot,
                             imgstr_rot_rev,
                             ])
    augimgtsr = augimgtsr.permute([1, 0, 2, 3, 4])
    augimgtsr = augimgtsr.reshape(-1, *augimgtsr.shape[2:], )
    augimgtsr_pp = normalize(resize(augimgtsr, [256, 256]))
    invfeattsrs_new = record_imgtsrs_dataset(model, [layer_long], augimgtsr_pp)[layer_long]
    # feattsrs = torch.load(join(outdir, "resnet50_linf8_INvalid_feattsrs.pt"))
    invrespvec = invfeattsrs_new[:, unit_id]
    tuningmap = invrespvec.reshape(-1, 7)
    # invar_cc = calc_invariance(invrespvec)
    if plot:
        mtgtsr = make_grid(augimgtsr,
                       nrow=7, padding=2, pad_value=0)
        figh = plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(mtgtsr.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"{netname} {layer_long} unit {unit_id}")
        plt.subplot(1, 2, 2)
        plt.imshow(tuningmap.numpy(), cmap="inferno")
        plt.title(f"Invariance ") #{invar_cc.item():.2f}
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    else:
        figh = None
    return augimgtsr, tuningmap, figh


def sample_idxs_by_bins(INresps, normthresh=1.0, bins=None, idx_per_bin=5):
    if bins is None:
        bins = [(0.0, 0.2),
                (0.2, 0.4),
                (0.4, 0.6),
                (0.6, 0.8),
                (0.8, 1.0),
                (1.0, 1.3),
                (1.3, 100),
                ]
    idx_all = []
    for LB, UB in bins:
        msk = (INresps > LB * normthresh) * (INresps < UB * normthresh)
        if msk.sum() == 0.0:
            continue
        elif msk.sum() < idx_per_bin:
            idx_bin = msk.nonzero()[:, 0]
        else:
            idx_bin = torch.multinomial(msk.float(), idx_per_bin)
        idx_all.extend(idx_bin)
    idx_all.append(INresps.argmax())
    idx_all = torch.stack(idx_all, )
    return idx_all

def augment_imgtsr(imgtsr, normalize=False, imgpix=256):
    imgstr_sml = affine(imgtsr, 0, [0, 0], scale=0.75, shear=[0, 0], fill=0.5)
    imgstr_lrg = affine(imgtsr, 0, [0, 0], scale=1.5, shear=[0, 0], fill=0.5)
    imgstr_lft = affine(imgtsr, 0, [45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rgt = affine(imgtsr, 0, [-45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rot = rotate(imgtsr, 30, expand=False, center=None, fill=0.5)
    imgstr_rot_rev = rotate(imgtsr, -30, expand=False, center=None, fill=0.5)
    augimgtsr = torch.stack([imgtsr,
                             imgstr_lrg,
                             imgstr_sml,
                             imgstr_rgt,
                             imgstr_lft,
                             imgstr_rot,
                             imgstr_rot_rev,
                             ])
    augimgtsr = augimgtsr.permute([1, 0, 2, 3, 4])
    augimgtsr = augimgtsr.reshape(-1, *augimgtsr.shape[2:], )
    return augimgtsr


def evol_sampling(G, scorer, batch=10, n_iter=75, noise_std=0.5,):
    dz = torch.randn(batch, 4096).cuda()
    dz.requires_grad_()
    optimizer = Adam([dz], lr=0.05)
    z_col = []
    resp_col = []
    gen_col = []
    pbar = tqdm(range(n_iter))
    for i in pbar:
        optimizer.zero_grad()
        curimg = G.visualize(dz)
        resp_new = scorer.score_tsr_wgrad(curimg, )
        z_col.append((dz).detach().cpu())
        resp_col.append(resp_new.detach().cpu())
        gen_col.append(torch.ones_like(resp_new.cpu()) * i)
        loss = - resp_new.mean()
        loss.backward()
        optimizer.step()
        nonactiv_zmsk = torch.isclose(resp_new, torch.zeros_like(resp_new))
        dz.data = dz.data + nonactiv_zmsk[:, None].float() * torch.randn_like(dz) * noise_std
        pbar.set_description(f"{i}: {resp_new.mean().item():.2f} ")
        # print(")#new score {resp_new.detach()}

    z_tsr = torch.concat(z_col, dim=0)
    resp_tsr = torch.concat(resp_col, dim=0)
    gen_tsr = torch.concat(gen_col, dim=0)
    return z_tsr, resp_tsr, gen_tsr
#%%
"""
1. Evolution process 
2. Filter images that reside in different ranges 
3. Measure the invariance of the images
"""
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
netname = "resnet50_linf8"
# Invfeatdata = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
#%%
G = upconvGAN().eval().cuda()
G.requires_grad_(False)
#%%
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\actlevel_tolerence_evol"
INdataset = create_imagenet_valid_dataset(normalize=False)
#%%
def compute_tolerance(tuningmap):
    Nobj, NTfm = tuningmap.shape
    centact = tuningmap[:, 0]
    meanact = tuningmap.mean(dim=1)
    maxact = tuningmap.max(dim=1).values
    obj_toler = ((tuningmap / maxact[:, None]).sum(dim=1) - 1) / (NTfm - 1)

    df = pd.DataFrame(
        {"centact": centact, "meanact": meanact, "maxact": maxact, "obj_toler": obj_toler})
    # df.to_csv(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"), index=False)
    return df

#%%
from NN_sparseness.insilico_manif_configs import RN50_config
layerlist = ['.layer1.Bottleneck1',
            '.layer2.Bottleneck0',
            '.layer2.Bottleneck3',
            '.layer3.Bottleneck2',
            '.layer3.Bottleneck4',
            '.layer4.Bottleneck0',
            '.layer4.Bottleneck2',]
for unit_id in range(0, 250, 5):
    for layer_long in layerlist:
        layer_short = shorten_layername(layer_long)
        unit_tup = (netname, layer_long, unit_id, *RN50_config[layer_long]["unit_pos"])
        scorer = TorchScorer(netname)
        scorer.select_unit(unit_tup, allow_grad=True)
        #%% ImageNet responses.
        INrespvec = feattsrs[unit_tup[1]][:, unit_tup[2]]
        INmax = INrespvec.max()
        IN99 = torch.quantile(INrespvec, 0.99)
        IN999 = torch.quantile(INrespvec, 0.999)
        #%%
        INthresh = torch.quantile(INrespvec, 0.998)
        INsampidx = sample_idxs_by_bins(INrespvec, normthresh=INthresh, idx_per_bin=15)
        INimgtsrs = torch.stack([INdataset[idx][0] for idx in INsampidx])
        INaugimgtsr = augment_imgtsr(INimgtsrs)
        # augimgtsr_pp = normalize(resize(augimgtsr, [256, 256]))
        with torch.no_grad():
            tune_scores = scorer.score_tsr_wgrad(INaugimgtsr, B=50, ).cpu()
        INtuningmap = tune_scores.reshape(-1, 7)
        #%%
        IN_tol_df = compute_tolerance(INtuningmap)
        IN_tol_df["imgidx"] = INsampidx.numpy()
        IN_tol_df.to_csv(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}_INet_toler.csv"), index=False)

        #%% Evol GAN
        z_tsr, resp_tsr, gen_tsr = evol_sampling(G, scorer,
                                                 batch=10, n_iter=75, noise_std=0.4,)
        # plt.figure(figsize=(5, 5))
        # plt.scatter(gen_tsr, resp_tsr, s=16, c="k", alpha=0.5)
        # plt.show()
        # raise Exception("Stop")
        #%%
        normthresh = torch.quantile(resp_tsr, 0.99)
        sampidx = sample_idxs_by_bins(resp_tsr, normthresh=normthresh, idx_per_bin=15)
        #%%
        imgtsr = G.visualize(z_tsr[sampidx, :].cuda())#.cpu()
        augimgtsr = augment_imgtsr(imgtsr)
        # augimgtsr_pp = normalize(resize(augimgtsr, [256, 256]))
        with torch.no_grad():
            tune_scores = scorer.score_tsr_wgrad(augimgtsr, B=50,).cpu()
        tuningmap = tune_scores.reshape(-1, 7)

        #%%
        evol_tol_df = compute_tolerance(tuningmap)
        evol_tol_df.to_csv(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_toler.csv"), index=False)
        #%%
        figh, ax = plt.subplots(figsize=[5, 5])
        plt.scatter(evol_tol_df.centact, evol_tol_df.obj_toler, alpha=0.6, label="GAN evol")
        plt.scatter(IN_tol_df.centact, IN_tol_df.obj_toler, alpha=0.6, label="ImageNet")
        ax.set(xlabel="act at center", ylabel="Absl. tolerance",
               title=f"Tolerance to transformation ~ activation level\n"
                     f"%s, %s, %d" % (unit_tup[0], shorten_layername(unit_tup[1]), unit_tup[2]))
        YLIM = ax.get_ylim()
        ax.vlines(INmax, YLIM[0], YLIM[1], color="r", label="ImageNet max")
        ax.vlines(IN99,  YLIM[0], YLIM[1], color="b", label="ImageNet 99%ile")
        ax.vlines(IN999, YLIM[0], YLIM[1], color="g", label="ImageNet 99.9%ile")
        # ax.vlines([INmax, IN99, IN999], *YLIM, linestyles="dashed", color="r", label=["max", "99%", "99.9%"])
        plt.legend()
        saveallforms(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}_evol_tolerance", figh)
        plt.show()
        #%%
        scorer.cleanup()
        # raise Exception("Stop")
#%%
#%%
