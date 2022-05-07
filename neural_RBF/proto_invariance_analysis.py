import pandas as pd
import torch
import os
import re
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict
from lpips import LPIPS
from GAN_utils import upconvGAN
from torch_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from scipy.stats import pearsonr, spearmanr
#%% Rename folder structure.
def _rename_folder_structure(root):
    # root = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
    dirnms = os.listdir(root)
    for dirnm in dirnms:
        if os.path.isdir(join(root, dirnm)) and dirnm.startswith("layer3-Btn5-5_rf_"):
            os.rename(join(root, dirnm), join(root, dirnm.replace("layer3-Btn5-5_rf_", "")))
#%% Post hoc sorting of the results.
def sweep_folder(root, dirnm_pattern=".*_max_abinit$", sum_sfx="summary"):
    sumdir = join(root, sum_sfx)
    os.makedirs(sumdir, exist_ok=True)
    repatt = re.compile(dirnm_pattern)  # (".*_max$")
    dirnms = os.listdir(root)
    sumdict = EasyDict({'imdist': [], 'score': [], 'z': []})
    for dirnm in dirnms:
        if not re.match(repatt, dirnm):
            continue
        saveD = EasyDict(torch.load(join(root, dirnm, "diversity_dz_score.pt")))
        z_final = saveD.dz_final + saveD.z_base.cpu()  # apply perturbation to `z_base`
        sumdict['z'].append(z_final)
        for k in ['imdist', 'score']:
            sumdict[k].append(saveD[k])
        for k in ["z_base", "score_base", "rfmaptsr"]:
            sumdict[k] = saveD[k].cpu()

    for k in sumdict:
        if isinstance(sumdict[k], list):
            sumdict[k] = torch.cat(sumdict[k], dim=0)

    torch.save(sumdict, join(sumdir, "diversity_z_summary.pt"))
    return sumdict, sumdir


def visualize_proto_by_level(sumdict, sumdir, bin_width=0.10, relwidth=0.25,
                             sampimgN=6, show=False):
    rfmaptsr = sumdict.rfmaptsr.cuda()
    proto_all_tsr = []
    for bin_c in np.arange(0.0, 1.10, 0.10):
        bin_r = bin_c + relwidth * bin_width
        bin_l = bin_c - relwidth * bin_width
        idx_mask = (sumdict.score >= bin_l * sumdict.score_base) * \
              (sumdict.score < bin_r * sumdict.score_base)
        # idx = idx_mask.nonzero().squeeze()
        z_bin = sumdict.z[idx_mask][:sampimgN]
        imgtsrs_rf = (G.visualize(z_bin.cuda()) * rfmaptsr.cuda()).cpu()
        if z_bin.shape[0] < sampimgN:
            padimgN = sampimgN - imgtsrs_rf.shape[0]
            imgtsrs_rf = torch.cat((imgtsrs_rf,
                    torch.zeros(padimgN, *imgtsrs_rf.shape[1:])), dim=0)
        if show:
            show_imgrid(imgtsrs_rf, nrow=1,)
        save_imgrid(imgtsrs_rf, join(sumdir, "proto_in_range_%0.2f-%.2f.png" % (bin_l, bin_r)),
                    nrow=1,)
        proto_all_tsr.append(imgtsrs_rf)

    save_imgrid(torch.cat(proto_all_tsr, dim=0),
                join(sumdir, "proto_level_progression.png", ),
                nrow=sampimgN, rowfirst=False)
    return proto_all_tsr


def visualize_score_imdist(sumdict, sumdir, ):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rval, pval = spearmanr(sumdict.score, sumdict.imdist)
    rval_pos, pval_pos = spearmanr(sumdict.score[sumdict.score > 0.0],
                                   sumdict.imdist[sumdict.score > 0.0])
    ax.plot(sumdict.score / sumdict.score_base, sumdict.imdist, '.', alpha=0.5)
    plt.xlabel("score / max score")
    plt.ylabel("LPIPS imdist")
    plt.title("Image distance to prototype vs score\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "imdist_vs_score.png"))
    plt.show()
    #%
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.hist(sumdict.score.numpy(), bins=40)
    plt.vlines(sumdict.score_base.item(), *plt.ylim(), color="red", linestyles="dashed")
    plt.xlabel("unit score")
    plt.ylabel("count")
    plt.title("unit score marginal distribution")
    plt.savefig(join(sumdir, "unit_score_dist_max_abinit.png"))
    plt.show()

#% Bin the results and summarize the prototypes within the bin.
def calc_proto_diversity_per_bin(sumdict, sumdir, bin_width=0.10, distsampleN=40):
    rfmaptsr = sumdict.rfmaptsr.cuda()
    pixdist_dict = {}
    lpipsdist_dict = {}
    lpipsdistmat_dict = {}
    df = pd.DataFrame()
    for bin_c in np.arange(0.0, 1.10, 0.10):
        bin_r = bin_c + 0.4 * bin_width
        bin_l = bin_c - 0.4 * bin_width
        idx_mask = (sumdict.score >= bin_l * sumdict.score_base) * \
              (sumdict.score < bin_r * sumdict.score_base)

        # idx = idx_mask.nonzero().squeeze()
        imdist_bin = sumdict.imdist[idx_mask]
        score_bin = sumdict.score[idx_mask]
        z_bin = sumdict.z[idx_mask]
        print("%0.2f-%0.2f: %d imdist %.3f+-%.3f  score %.3f+-%.3f" % (bin_l, bin_r, idx_mask.sum(),
              imdist_bin.mean().item(), imdist_bin.std().item(),
              score_bin.mean().item(), score_bin.std().item()))
        #%
        if z_bin.shape[0] > 1: # cannot compute pairwise distance with one sample;
            imgtsrs = G.visualize_batch(z_bin[:distsampleN, :].cuda())
            imgtsrs_rf = imgtsrs * rfmaptsr.cpu()
            pixdist = torch.pdist(imgtsrs_rf.flatten(start_dim=1))
            print("pairwise pixel dist %.3f+-%.3f N=%d" % (pixdist.mean(), pixdist.std(), len(pixdist)))
            pixdist_dict[bin_c] = pixdist
            # calculate lpips distance matrix row by row.
            # TODO: this is slow. Can we do it in batch?
            # distmat_bin = []
            # for i in range(imgtsrs.shape[0]):
            #     dist_in_bin = Dist(imgtsrs.cuda() * rfmaptsr, imgtsrs[i:i + 1].cuda() * rfmaptsr).cpu().squeeze()
            #     distmat_bin.append(dist_in_bin)
            #
            # distmat_bin = torch.stack(distmat_bin, dim=0)
            # Batch processing version, much faster!
            distmat_bin = Dist.forward_distmat(imgtsrs_rf.cuda(), None).cpu().squeeze()
            mask = torch.triu(torch.ones(*distmat_bin.shape, dtype=torch.bool), diagonal=1, )
            pairwise_dist = distmat_bin[mask]
            print("pairwise dist %.3f+-%.3f N=%d" % (pairwise_dist.mean(), pairwise_dist.std(), len(pairwise_dist)))
            lpipsdist_dict[bin_c] = pairwise_dist
            lpipsdistmat_dict[bin_c] = distmat_bin
            df_part = pd.DataFrame({"bin_c": bin_c, "bin_l": bin_l,"bin_r": bin_r,
                          "pixdist": pixdist, "lpipsdist": pairwise_dist})
            df = df_part if df.empty else pd.concat((df, df_part), axis=0)
        else:
            print("cannot compute pairwise distance with one or zero sample;")

    torch.save({"pixdist": pixdist_dict, "lpipsdist": lpipsdist_dict, "lpipsdistmat": lpipsdistmat_dict},
               join(sumdir, "imgdist_by_bin_dict.pt"))
    df.to_csv(join(sumdir, "imgdist_by_bin.csv"))
    df["bin_label"] = df.bin_c.apply(lambda c: "%0.1f" % c)
    return df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict


def visualize_diversity_by_bin(df, sumdir):
    rval, pval = spearmanr(df.bin_c, df.lpipsdist)
    rval_pos, pval_pos = spearmanr(df.bin_c[df.bin_c > 0.0], df.lpipsdist[df.bin_c > 0.0])
    figh, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.violinplot(x="bin_label", y="lpipsdist", data=df)
    ax.set_xlabel("activation level (bin center)")
    ax.set_ylabel("lpips dist among prototypes")
    ax.set_title("LPIPS diversity ~ activation level\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "pairwise_lpips_dist_by_bin.png"))
    plt.show()
    rval, pval = spearmanr(df.bin_c, df.pixdist)
    rval_pos, pval_pos = spearmanr(df.bin_c[df.bin_c > 0.0], df.pixdist[df.bin_c > 0.0])
    figh, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.violinplot(x="bin_label", y="pixdist", data=df)
    ax.set_xlabel("activation level (bin center)")
    ax.set_ylabel("pixel dist among prototypes")
    ax.set_title("pixel diversity ~ activation level\n"
                 "Spearman r=%.3f p=%.1e\n"
                 "Spearman (excld 0) r=%.3f p=%.1e" % (rval, pval, rval_pos, pval_pos))
    plt.savefig(join(sumdir, "pairwise_pixel_dist_by_bin.png"))
    plt.show()

#%%
#%% New experiments
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
# root = r"E:\insilico_exps\proto_diversity\resnet50\layer4-Btn0-5_rf"
# root = r"E:\insilico_exps\proto_diversity\resnet50\layer2-Btn3-10_rf"
# root = r"E:\insilico_exps\proto_diversity\resnet50\layer2-Btn3-5_rf"
# root = r"E:\insilico_exps\proto_diversity\resnet50\layer4-Btn2-5_rf"
# root = r"E:\insilico_exps\proto_diversity\resnet50\layer3-Btn5-5_rf"
root = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer4-Btn2-10_rf"
for suffix in ["min", "max", "max_abinit"]:
    sumdict, sumdir = sweep_folder(root, dirnm_pattern=f"fix.*_{suffix}$",
                                   sum_sfx=f"summary_{suffix}")
    visualize_proto_by_level(sumdict, sumdir, bin_width=0.10, relwidth=0.25,)
    visualize_score_imdist(sumdict, sumdir, )
    df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(sumdict, sumdir, bin_width=0.10, distsampleN=40)
    visualize_diversity_by_bin(df, sumdir)
#%%
rootdir = r"E:\insilico_exps\proto_diversity\resnet50"
subdirlist = [fdn for fdn in os.listdir(rootdir) if "layer" in fdn]
for subdir in subdirlist:
    root = join(rootdir, subdir)
    for suffix in ["min", "max", "max_abinit"]:
        sumdict, sumdir = sweep_folder(root, dirnm_pattern=f"fix.*_{suffix}$",
                                       sum_sfx=f"summary_{suffix}")
        visualize_proto_by_level(sumdict, sumdir, bin_width=0.10, relwidth=0.25,)
        visualize_score_imdist(sumdict, sumdir, )
        df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(sumdict, sumdir, bin_width=0.10, distsampleN=40)
        visualize_diversity_by_bin(df, sumdir)
#%%

#%% Dev zone for debugging
#%%
distmat_tmp = Dist.forward_distmat(imgtsrs_rf.cuda(), ).squeeze()
#%%
def torch_cosine_mat(X, Y=None):
    if Y is None:
        Y = X
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    return (X @ Y.T) / torch.norm(X, dim=1, keepdim=True) / torch.norm(Y, dim=1, keepdim=True).T

torch_cosine_mat(z_bin[:50, :])
#%%
imgtsrs = G.visualize_batch(z_bin.cuda())
imgtsrs_rf = G.visualize_batch(z_bin.cuda())*rfmaptsr
distmat_bin = []
for i in range(z_bin.shape[0]):
    dist_in_bin = Dist(imgtsrs.cuda(), imgtsrs[i:i+1].cuda()*rfmaptsr).cpu().squeeze()
    distmat_bin.append(dist_in_bin)

distmat_bin = torch.stack(distmat_bin, dim=0)
mask = torch.triu(torch.ones(*distmat_bin.shape, dtype=bool), diagonal=1,)
pairwise_dist = distmat_bin[mask]
print("mean pairwise dist %.3f+-%.3f N=%d"%(pairwise_dist.mean(), pairwise_dist.std(), len(pairwise_dist)))
#%%
# trial_dir = join(outrf_dir, "tmp")
# opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.0,
#             score_obj="fix", score_fixval=20, alpha_score=1.0, noise_std=0.3,)
# S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, trial_dir, repn=1)
# #%%
# failmask = torch.isclose(S.score, torch.zeros(1))
# z = S.dz_final + noisestd * torch.randn(5, 4096)*failmask.unsqueeze(1)
# regresslayer = ".layer3.Bottleneck5"
# featnet, net = load_featnet("resnet50_linf8")
# featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
#                              device="cuda", print_module=False)
# featFetcher.record(regresslayer,)
#%%
distmat_bin = []
for i in range(imgtsrs.shape[0]):
    dist_in_bin = Dist(imgtsrs.cuda(), imgtsrs[i:i + 1].cuda()).cpu().squeeze()
    distmat_bin.append(dist_in_bin)

distmat_bin = torch.stack(distmat_bin, dim=0)
# Batch processing version, much faster!
distmat_bin = Dist.forward_distmat(imgtsrs_rf.cuda(), None).cpu().squeeze()
