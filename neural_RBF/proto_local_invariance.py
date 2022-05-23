"""Find RBF approximation of CNN activations
Use optimizations to find the images with a certain activation level while maximizing or minimizing the distance to a target.

"""
import re
import os
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.models as models
from lpips import LPIPS
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from ZO_HessAware_Optimizers import CholeskyCMAES
from torch_utils import show_imgrid, save_imgrid, save_imgrid_by_row
from featvis_lib import load_featnet
from easydict import EasyDict
from layer_hook_utils import featureFetcher, get_module_names, register_hook_by_module_names, layername_dict
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from neural_RBF.proto_invariance_lib import sweep_folder, visualize_proto_by_level, visualize_score_imdist, \
        calc_proto_diversity_per_bin, visualize_diversity_by_bin
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
from neural_regress.regress_lib import compare_activation_prediction, sweep_regressors, \
    resizer, normalizer, denormalizer, PoissonRegressor, RidgeCV, Ridge, KernelRidge
#%
def latent_diversity_explore(G, Dist, scorer, z_base, dzs=None, alpha=10.0, dz_sigma=3.0,
                      batch_size=5, steps=150, lr=0.1, midpoint=True):
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        score_loss = (resp_base - resp_news)
        if midpoint:
            resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
            score_mid_loss = (resp_base - resp_news_mid)
        else:
            score_mid_loss = torch.zeros_like(score_loss)
        img_dists = Dist(img_base, curimgs)[:, 0, 0, 0]
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(),\
           curimgs.detach().cpu(), resp_news.detach().cpu()


def latent_diversity_explore_wRF(G, Dist, scorer, z_base, rfmaptsr, dzs=None, alpha=10.0, dz_sigma=3.0,
                      batch_size=5, steps=150, lr=0.1, midpoint=True):
    """

    :param G:
    :param scorer:
    :param z_base: base latent vector
    :param rfmaptsr: We assume its shape is (1, 1, 256, 256), and its values are in [0, 1]
    :param alpha: The weight of the distance term VS the score term
    :param dz_sigma: The initial std of dz.
    :param dzs: The initial dz.
            If None, it will be sampled from a Gaussian distribution with std dz_sigma.
    :param batch_size:
    :param steps:
    :param lr:
    :param midpoint: If True, the activation of midpoint is also computed.
    :return:
    """
    Dist.spatial = False
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        score_loss = (resp_base - resp_news)
        if midpoint:
            resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
            score_mid_loss = (resp_base - resp_news_mid)
        else:
            score_mid_loss = torch.zeros_like(score_loss)
        img_dists = Dist(img_base * rfmaptsr, curimgs * rfmaptsr)[:, 0, 0, 0]
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} RF img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()


def latent_diversity_explore_wRF_fixval(G, Dist, scorer, z_base, rfmaptsr, dzs=None, dz_sigma=3.0,
                                        imgdist_obj="max", imgdist_fixval=None, alpha_img=1.0,
                                        score_obj="max", score_fixval=None, alpha_score=1.0,
                                        batch_size=5, steps=150, lr=0.1, noise_std=0.3, midpoint=True):
    """ Most recent version of latent diversity explore.
    Setting 1, fix the score at half maximum. Maximize image distance to prototype
        latent_diversity_explore_wRF_fixval(G, scorer, z_base, rfmaptsr,
                        imgdist_obj="max", score_obj="fix", score_fixval=score_base * 0.5, alpha_score=10.0)
    Setting 2, fix the image distance to prototype. minimize or maximize score
        latent_diversity_explore_wRF_fixval(G, scorer, z_base, rfmaptsr,
                        imgdist_obj="fix", imgdist_fixval=0.1, score_obj="max", alpha_img=10.0)
    :param G:
    :param scorer:
    :param z_base: base latent vector
    :param rfmaptsr: We assume its shape is (1, 1, 256, 256), and its values are in [0, 1]
    :param alpha: The weight of the distance term VS the score term
    :param dz_sigma: The initial std of dz.
    :param dzs: The initial dz.
            If None, it will be sampled from a Gaussian distribution with std dz_sigma.
    :param batch_size:
    :param steps:
    :param lr:
    :param midpoint: If True, the activation of midpoint is also computed.
    :return:
    """
    Dist.spatial = False
    if dzs is None:
        dzs = dz_sigma * torch.randn(batch_size, 4096).cuda()
    dzs_init = dzs.clone().cpu()
    dzs.requires_grad_()
    optimizer = Adam([dzs], lr=lr)
    for i in tqdm(range(steps)):
        optimizer.zero_grad()
        curimgs = G.visualize(z_base + dzs)
        resp_news = scorer.score_tsr_wgrad(curimgs, )
        if score_obj is "max":
            score_loss = (resp_base - resp_news)
        elif score_obj is "min":
            score_loss = - (resp_base - resp_news)
        elif score_obj is "fix":
            score_loss = torch.abs(resp_news - score_fixval)
        else:
            raise ValueError(f"Unknown score_obj {score_obj}")

        img_dists = Dist(img_base * rfmaptsr, curimgs * rfmaptsr)[:, 0, 0, 0]
        if imgdist_obj is "max":
            dist_loss = - img_dists
        elif imgdist_obj is "min":
            dist_loss = img_dists
        elif imgdist_obj is "fix":
            dist_loss = torch.abs(img_dists - imgdist_fixval)
        else:
            raise ValueError(f"Unknown imgdist_obj {imgdist_obj}")
        # if midpoint:
        #     resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
        #     score_mid_loss = (resp_base - resp_news_mid)
        # else:
        #     score_mid_loss = torch.zeros_like(score_loss)
        failmask = torch.isclose(resp_news.detach(), torch.zeros(1, device="cuda"))
        loss = (alpha_score * score_loss + alpha_img * dist_loss * (~failmask)).mean()
        # loss = (alpha_score * score_loss + alpha_img * dist_loss).mean()
        loss.backward()
        optimizer.step()
        if failmask.any():
            dzs.data = dzs.data + noise_std * \
                torch.randn(batch_size, 4096, device="cuda") * failmask.unsqueeze(1)
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} "
            f"old score {resp_base.item():.2f} RF img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), img_dists.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()


def search_peak_evol(G, scorer, nstep=100):
    resp_all = []
    z_all = []
    optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
    z_arr = np.zeros((1, 4096))  # optimizer.init_x
    pbar = tqdm(range(nstep))
    for i in pbar:
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        z_arr_new = optimizer.step_simple(resp, z_arr)
        z_arr = z_arr_new
        resp_all.append(resp)
        z_all.append(z_arr)
        print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")

    resp_all = np.concatenate(resp_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    z_base = torch.tensor(z_all.mean(axis=0, keepdims=True)).float().cuda()
    img_base = G.visualize(z_base)
    resp_base = scorer.score(img_base, )
    return z_base, img_base, resp_base, resp_all, z_all


def search_peak_gradient(G, scorer, z_base, resp_base, nstep=200):
    dz = 0.1 * torch.randn(1, 4096).cuda()
    dz.requires_grad_()
    optimizer = Adam([dz], lr=0.1)
    for i in tqdm(range(nstep)):
        optimizer.zero_grad()
        curimg = G.visualize(z_base + dz)
        resp_new = scorer.score_tsr_wgrad(curimg, )
        # img_dist = Dist(img_base, curimg)
        loss = - resp_new
        loss.backward()
        optimizer.step()
        print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
              f"old score {resp_base.item():.2f}")
    # show_imgrid(curimgs)
    z_base = z_base + dz.detach().clone()
    z_base.detach_()
    img_base = G.visualize(z_base)
    resp_base = scorer.score(img_base, )
    return z_base, img_base, resp_base

#%%
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
# cent_pos = (6, 6)
def calc_rfmap(scorer, rf_dir, label=None, use_fit=True, device="cuda",):
    if label is None:
        label = "%s-%d"%(scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
    gradAmpmap = grad_RF_estimate(scorer.model, scorer.layer, (slice(None), scorer.unit_x, scorer.unit_y),
                                  input_size=scorer.inputsize, device=device, show=False, reps=200, batch=4)
    show_gradmap(gradAmpmap, )
    fitdict = fit_2dgauss(gradAmpmap, label, outdir=rf_dir, plot=True)
    rfmap = fitdict.fitmap if use_fit else fitdict.gradAmpmap
    rfmap /= rfmap.max()
    rfmaptsr = torch.from_numpy(rfmap).float().cuda().unsqueeze(0).unsqueeze(0)
    rfmaptsr = F.interpolate(rfmaptsr, (256, 256), mode="bilinear", align_corners=True)
    rfmap_full = rfmaptsr.cpu()[0, 0, :].unsqueeze(2).numpy()
    return rfmaptsr, rfmap_full, fitdict

#%
def latent_explore_batch(z_base, rfmaptsr, opts, outdir, repn=20, ):
    os.makedirs(outdir, exist_ok=True)
    dz_init_col = []
    dz_col = []
    score_col = []
    imdist_col = []
    for i in range(repn):
        dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF_fixval(G, Dist, scorer,
                            z_base, rfmaptsr, **opts)
        save_imgrid(curimgs, join(outdir, f"proto_divers_{i}.png"))
        save_imgrid(curimgs * rfmaptsr.cpu(), join(outdir, f"proto_divers_wRF_{i}.png"))
        dz_init_col.append(dzs_init)
        dz_col.append(dzs)
        score_col.append(scores)
        imdist_col.append(img_dists)

    dz_init_tsr = torch.cat(dz_init_col, dim=0)
    dz_final_tsr = torch.cat(dz_col, dim=0)
    score_vec = torch.cat(score_col, dim=0)
    imdist_vec = torch.cat(imdist_col, dim=0)
    savedict = {"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
                "z_base": z_base, "score_base": resp_base, "rfmaptsr": rfmaptsr, "opts": opts, }
    torch.save(savedict, join(outdir, "diversity_dz_score.pt"))
    return EasyDict(savedict)


def pick_goodimages(S, rfmaptsr, thresh=2.5):
    imdist_good = S.imdist[S.score > thresh]
    imdist_bad = S.imdist[S.score < thresh]
    print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
          f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")

    show_imgrid(G.visualize_batch(S.dz_final[S.score > thresh, :].cuda()).cpu()*rfmaptsr.cpu())


def filter_visualize_codes(outdir, thresh=2.5, err=None, subdir="sorted", abinit=True):
    """Filter out codes with certain score and only present these codees."""
    S = EasyDict(torch.load(join(outdir, "diversity_dz_score.pt")))
    imdist_good = S.imdist[S.score > thresh]
    imdist_bad = S.imdist[S.score < thresh]
    score_good = S.score[S.score > thresh]
    score_bad = S.score[S.score < thresh]

    print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
          f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")
    print(f"score good {score_good.mean():.2f}+-{score_good.std():.2f}\t"
          f"{score_bad.mean():.2f}+-{score_bad.std():.2f}")
    rho, pval = pearsonr(S.score, S.imdist)
    print(f"Corr between score and im dist to proto {rho:.3f} P={pval:.3f} (All samples)")

    sortidx = torch.argsort(- S.score)
    score_sort = S.score[sortidx]
    msk = score_sort > thresh
    if err is not None:
        msk = torch.abs(score_sort - thresh) < err
    score_sort = score_sort[msk]
    imdist_sort = S.imdist[sortidx][msk]
    dz_final_sort = S.dz_final[sortidx, :][msk, :]
    zs = dz_final_sort if abinit else dz_final_sort + S.z_base.cpu()

    imgs = G.visualize_batch(zs)
    os.makedirs(join(outdir, subdir), exist_ok=True)
    save_imgrid(imgs, join(outdir, subdir, "proto_divers.png"))
    save_imgrid(imgs * S.rfmaptsr.cpu(), join(outdir, subdir, "proto_divers_wRF.png"))
    save_imgrid_by_row(imgs * S.rfmaptsr.cpu(), join(outdir, subdir, "proto_divers_wRF.png"), n_row=5)
    S_new = S
    S_new.score = score_sort
    S_new.imdist = imdist_sort
    S_new.dz_final = dz_final_sort
    S_new.dz_init = S.dz_init[sortidx, :][msk, :]
    torch.save(S_new, join(outdir, subdir, "diversity_dz_score.pt"))
    df = pd.DataFrame({"score": S_new.score, "imdist2proto": S_new.imdist})
    df.to_csv(join(outdir, subdir, "score_dist_summary.csv"))
    rho, pval = pearsonr(S_new.score, S_new.imdist)
    print(f"Corr between score and im dist to proto {rho:.3f} P={pval:.3f} (After filter)")
    # torch.save(dict(score=score_sort, imdist=imdist_sort, dz_final=dz_final_sort,
    #                 dz_init=S.dz_init[sortidx, :][msk, :], z_base=S.z_base,
    #                 score_base=S.score_base, rfmaptsr=S.rfmaptsr, opts=S.opts),
    #            join(outdir, "sorted", "diversity_dz_score.pt"))
    return S_new

#%% New experiments
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
scorer = TorchScorer("resnet50")
# scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 7, 7), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer4.Bottleneck2", 5, 3, 3), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer2.Bottleneck3", 5, 13, 13), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer2.Bottleneck3", 10, 13, 13), allow_grad=True)
# scorer.select_unit(("resnet50", ".layer4.Bottleneck0", 5, 3, 3), allow_grad=True)
scorer.select_unit(("resnet50", ".layer4.Bottleneck1", 5, 3, 3), allow_grad=True)
#%%
unitlist = [("resnet50", ".layer4.Bottleneck2", 10, 3, 3),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50", ".layer4.Bottleneck1", 5, 3, 3),
            ("resnet50", ".layer4.Bottleneck0", 10, 3, 3),
            ("resnet50", ".layer2.Bottleneck1", 5, 13, 13),
            ("resnet50", ".layer2.Bottleneck3", 15, 13, 13),
            ]
#%%
unitlist = [#("resnet50_linf8", ".layer4.Bottleneck2", 10, 3, 3),
            #("resnet50_linf8", ".layer4.Bottleneck1", 5, 3, 3),
            ("resnet50_linf8", ".layer3.Bottleneck5", 5, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50_linf8", ".layer2.Bottleneck3", 5, 13, 13),
            ("resnet50_linf8", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50_linf8", ".layer3.Bottleneck5", 10, 7, 7),
            ("resnet50_linf8", ".layer4.Bottleneck0", 10, 3, 3),
            ("resnet50_linf8", ".layer2.Bottleneck1", 5, 13, 13),
            ]

repn = 4
for unit_tup in unitlist:
    netname = unit_tup[0]
    scorer = TorchScorer(netname)
    scorer.select_unit(unit_tup, allow_grad=True)
    unitlabel = "%s-%d" % (scorer.layer.replace(".Bottleneck", "-Btn").strip("."), scorer.chan)
    outroot = join(r"E:\insilico_exps\proto_diversity", netname)
    outrf_dir = join(outroot, unitlabel+"_rf")
    os.makedirs(outrf_dir, exist_ok=True)
    rfmaptsr, rfmapnp, fitdict = calc_rfmap(scorer, outrf_dir, label=unitlabel, use_fit=True, )
    #%%
    z_evol, img_evol, resp_evol, resp_all, z_all = search_peak_evol(G, scorer, nstep=100)
    z_base, img_base, resp_base = search_peak_gradient(G, scorer, z_evol, resp_evol, nstep=100)
    resp_base = torch.tensor(resp_base).float().cuda()
    save_imgrid(img_base, join(outrf_dir, "proto_peak.png"))
    save_imgrid(img_base*rfmaptsr, join(outrf_dir, "proto_peak_rf.png"))
    torch.save(dict(z_base=z_base, img_base=img_base, resp_base=resp_base,
                    z_evol=z_evol, img_evol=img_evol, resp_evol=resp_evol,
                    unit_tuple=unit_tup, unitlabel=unitlabel),
               join(outrf_dir, "proto_optim.pt"))
    #%%
    for ratio in np.arange(0.10, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, noise_std=0.3,)
        S = latent_explore_batch(z_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * ratio * 0.2)

    for ratio in np.arange(0.1, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=3, imgdist_obj="max", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, noise_std=0.3,)
        S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=True,
                                       err=resp_base.item() * ratio * 0.2)

    for ratio in np.arange(0.10, 1.1, 0.1):
        trial_dir = join(outrf_dir, "fix%s_min"%(("%.2f"%ratio).replace(".", "")))
        opts = dict(dz_sigma=1.5, imgdist_obj="min", alpha_img=5.0,
                    score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, noise_std=0.3,)
        S = latent_explore_batch(z_base, rfmaptsr, opts, trial_dir, repn=repn)
        S_sel = filter_visualize_codes(trial_dir, thresh=resp_base.item() * ratio, abinit=False,
                                       err=resp_base.item() * ratio * 0.2)

    for suffix in ["min", "max", "max_abinit"]:
        sumdict, sumdir = sweep_folder(outrf_dir, dirnm_pattern=f"fix.*_{suffix}$",
                                       sum_sfx=f"summary_{suffix}")
        visualize_proto_by_level(G, sumdict, sumdir, bin_width=0.10, relwidth=0.25, )
        visualize_score_imdist(sumdict, sumdir, )
        df, pixdist_dict, lpipsdist_dict, lpipsdistmat_dict = calc_proto_diversity_per_bin(G, Dist, sumdict, sumdir,
                                                                   bin_width=0.10, distsampleN=40)
        visualize_diversity_by_bin(df, sumdir)


#%%

#%%

#%%


#%% Development zone
#%%
feattsr_all = []
resp_all = []
z_all = []
optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
z_arr = np.zeros((1, 4096))  # optimizer.init_x
pbar = tqdm(range(100))
for i in pbar:
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score(imgs, )
    z_arr_new = optimizer.step_simple(resp, z_arr)
    z_arr = z_arr_new
    resp_all.append(resp)
    z_all.append(z_arr)
    print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    # with torch.no_grad():
    #     featnet(scorer.preprocess(imgs, input_scale=1.0))
    #
    # del imgs
    # pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    # feattsr = featFetcher[regresslayer][:, :, 6, 6]
    # feattsr_all.append(feattsr.cpu().numpy())

resp_all = np.concatenate(resp_all, axis=0)
z_all = np.concatenate(z_all, axis=0)
# feattsr_all = np.concatenate(feattsr_all, axis=0)

z_base = torch.tensor(z_all.mean(axis=0, keepdims=True)).float().cuda()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%% Maximize invariance / Diversity
#%%
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=True)
#%%
dz = 0.1 * torch.randn(1, 4096).cuda()
dz.requires_grad_()
optimizer = Adam([dz], lr=0.1)
for i in tqdm(range(200)):
    optimizer.zero_grad()
    curimg = G.visualize(z_base + dz)
    resp_new = scorer.score_tsr_wgrad(curimg, )
    score_loss = (resp_base - resp_new)
    img_dist = Dist(img_base, curimg)
    loss = score_loss
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dist.item():.2f}")
# show_imgrid(curimgs)
#%%
z_base = z_base + dz.detach().clone()
z_base.detach_()
#%%
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%%
out_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5"
opts = dict(alpha=10.0, dz_sigma=2.0, batch_size=5, steps=150, lr=0.1, )
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(100):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore(G, Dist, scorer, z_base,
                  **opts)
    save_imgrid(curimgs, join(out_dir, f"proto_divers_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "opts": opts},
           join(out_dir, "diversity_dz_score.pt"))
#%% Calculate RF mask of the unit.
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
cent_pos = (6, 6)
outrf_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
gradAmpmap = grad_RF_estimate(scorer.model, scorer.layer, (slice(None), cent_pos[0], cent_pos[1]),
                          input_size=(3, 227, 227), device="cuda", show=False, reps=200, batch=4)
show_gradmap(gradAmpmap, )
fitdict = fit_2dgauss(gradAmpmap, "layer3-Btn5-5", outdir=outrf_dir, plot=True)
#%%
rfmap = fitdict.fitmap
rfmap = fitdict.gradAmpmap
rfmap /= rfmap.max()
rfmaptsr = torch.from_numpy(rfmap).float().cuda().unsqueeze(0).unsqueeze(0)
rfmaptsr = torch.nn.functional.interpolate(rfmaptsr,
               (256, 256), mode="bilinear", align_corners=True)
rfmap_full = rfmaptsr.cpu()[0,0,:].unsqueeze(2).numpy()


#%%
# out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
# alpha=10.0; dz_sigma=2.0
out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf_far200"
alpha = 200.0; dz_sigma = 3.0
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(100):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF(G, Dist, scorer, z_base,
                  rfmaptsr, alpha=alpha, dz_sigma=dz_sigma, batch_size=5, steps=150, lr=0.1, )
    save_imgrid(curimgs, join(out_dir_rf, f"proto_divers_{i}.png"))
    save_imgrid(curimgs*rfmaptsr.cpu(), join(out_dir_rf, f"proto_divers_wRF_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "alpha": alpha, "dz_sigma": dz_sigma},
           join(out_dir_rf, "diversity_dz_score.pt"))
# 4.54
#%%
out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf_fix05_min"
os.makedirs(out_dir_rf, exist_ok=True)
opts = dict(dz_sigma=1.0, score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,
        imgdist_obj="min", alpha_img=0.2)
dz_init_col = []
dz_col = []
score_col = []
imdist_col = []
for i in range(5):
    dzs_init, dzs, img_dists, curimgs, scores = latent_diversity_explore_wRF_fixval(G, Dist, scorer,
                        z_base, rfmaptsr, **opts)
    save_imgrid(curimgs, join(out_dir_rf, f"proto_divers_{i}.png"))
    save_imgrid(curimgs*rfmaptsr.cpu(), join(out_dir_rf, f"proto_divers_wRF_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
    imdist_col.append(img_dists)

dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
imdist_vec = torch.cat(imdist_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec, "imdist": imdist_vec,
            "z_base": z_base, "score_base": resp_base, "rfmaptsr": rfmaptsr, "opts": opts, },
           join(out_dir_rf, "diversity_dz_score.pt"))

#%% New interface
#%% ab initio generation of images matching
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_min")
opts = dict(dz_sigma=3.0, score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,
            imgdist_obj="min", alpha_img=0.2)
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_min_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="min", alpha_img=0.2,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0,)
S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.1,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0, )
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=20)
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix05_max_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.05,
            score_obj="fix", score_fixval=resp_base * 0.5, alpha_score=1.0, )
S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=2.7, )

#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix08_max_abinit")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=0.05,
            score_obj="fix", score_fixval=resp_base * 0.8, alpha_score=1.0, )
S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=4.4, )
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix08_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.8, alpha_score=1.0, )
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=4.4, abinit=False)
#%%
#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix03_max")
opts = dict(dz_sigma=3.0, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.3, alpha_score=1.0, )
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=1.4, abinit=False)

#%%
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix015_max")
opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.00,
            score_obj="fix", score_fixval=resp_base * 0.15, alpha_score=1.0, )
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=20)
S_sel = filter_visualize_codes(out_dir_rf, thresh=0.65, abinit=False)
#%%
for ratio in np.arange(0.10, 1, 0.05):
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max"%(("%.2f"%ratio).replace(".", "")))
    opts = dict(dz_sigma=1.5, imgdist_obj="max", alpha_img=5.0,
                score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
    S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=40)
    S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=False,
                                   err=resp_base.item() * ratio * 0.2)
#%
for ratio in np.arange(0.05, 1, 0.05):
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max_abinit"%(("%.2f"%ratio).replace(".", "")))
    opts = dict(dz_sigma=3, imgdist_obj="max", alpha_img=5.0,
                score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
    S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, out_dir_rf, repn=40)
    S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                                   err=resp_base.item() * ratio * 0.2)
#%%
ratio = 0.5
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_min"%(("%.2f"%ratio).replace(".", "")))
opts = dict(dz_sigma=2, imgdist_obj="min", alpha_img=5.0,
            score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
S = latent_explore_batch(z_base, rfmaptsr, opts, out_dir_rf, repn=40)
S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                               err=resp_base.item() * ratio * 0.2)
ratio = 0.5
out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_min_abinit"%(("%.2f"%ratio).replace(".", "")))
opts = dict(dz_sigma=2, imgdist_obj="min", alpha_img=5.0,
            score_obj="fix", score_fixval=resp_base * ratio, alpha_score=1.0, )
S = latent_explore_batch(torch.zeros_like(z_base), rfmaptsr, opts, out_dir_rf, repn=40)
S_sel = filter_visualize_codes(out_dir_rf, thresh=resp_base.item() * ratio, abinit=True,
                               err=resp_base.item() * ratio * 0.2)
#%%


#%%
S_sel = filter_visualize_codes(join(outroot, "layer3-Btn5-5_rf_fix05_min_abinit"),
                               thresh=2.4, abinit=True,)
#%%
S_sel = filter_visualize_codes(join(outroot, "layer3-Btn5-5_rf_fix05_min"),
                               thresh=2.4, abinit=False,)
# pick_goodimages(S, rfmaptsr, thresh=2.5)
#%% Copy to a summary folder
import shutil
abinit = True
os.makedirs(join(outroot, "summary"), exist_ok=True)
for ratio in np.arange(0.05, 1, 0.05):
    ratio_str = ("%.2f"%ratio).replace(".", "")
    out_dir_rf = join(outroot, "layer3-Btn5-5_rf_fix%s_max%s"%
                      (ratio_str, "_abinit" if abinit else ""))
    outfn = "%sproto_divers_%s.png"%("abinit_" if abinit else "", ratio_str)
    shutil.copy2(join(out_dir_rf, "sorted", "proto_divers_wRF_0.png"),
                 join(outroot, "summary", outfn))



#%%
imdist_good = imdist_vec[score_vec > 2.8]
imdist_bad = imdist_vec[score_vec < 2.8]
print(f"imdist good {imdist_good.mean():.2f}+-{imdist_good.std():.2f}\t"
      f"{imdist_bad.mean():.2f}+-{imdist_bad.std():.2f}")
#%%
show_imgrid(G.visualize(dz_final_tsr[score_vec > 2.7, :].cuda()).cpu()*rfmaptsr.cpu())
#%%
cos_angle_good = (dz_final_tsr[score_vec>0.05,:] @ z_base.cpu().T) \
    / dz_final_tsr[score_vec>0.05,:].norm(dim=1, keepdim=True) / z_base.cpu().norm(dim=1)
cos_angle_bad = (dz_final_tsr[score_vec<0.05,:] @ z_base.cpu().T) \
    / dz_final_tsr[score_vec<0.05,:].norm(dim=1, keepdim=True) / z_base.cpu().norm(dim=1)
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_base.cpu().detach().numpy()[0].transpose(1, 2, 0))
axs[1].imshow(curimg.cpu().detach().numpy()[0].transpose(1, 2, 0))
plt.show()
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_base.cpu().detach().numpy()[0].transpose(1, 2, 0) * rfmap_full)
axs[1].imshow(curimg.cpu().detach().numpy()[0].transpose(1, 2, 0) * rfmap_full)
plt.show()
#%%
show_imgrid(curimgs * rfmaptsr.cpu())
#%%
#%% New iteration
z_base = z_base + dz.detach()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%% Dev Zone
#%% Single thread
dz = 0.1 * torch.randn(1, 4096).cuda()
dz.requires_grad_()
optimizer = Adam([dz], lr=0.05)
for i in tqdm(range(100)):
    optimizer.zero_grad()
    curimg = G.visualize(z_base + dz)
    resp_new = scorer.score_tsr_wgrad(curimg, )
    score_loss = (resp_base - resp_new)
    img_dist = Dist(img_base, curimg)
    loss = score_loss - img_dist
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_new.item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dist.item():.2f}")
# %% Multi thread exploration
alpha = 10.0
dz_sigma = 3  # 0.4
dzs = dz_sigma * torch.randn(5, 4096).cuda()
dzs.requires_grad_()
optimizer = Adam([dzs], lr=0.1)
for i in tqdm(range(150)):
    optimizer.zero_grad()
    curimgs = G.visualize(z_base + dzs)
    resp_news = scorer.score_tsr_wgrad(curimgs, )
    score_loss = (resp_base - resp_news)
    resp_news_mid = scorer.score_tsr_wgrad(G.visualize(z_base + dzs / 2), )
    score_mid_loss = (resp_base - resp_news_mid)
    img_dists = Dist(img_base, curimgs)
    loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
#%%


#%% md
# Trying different spatial weighted Distance computation
#%%
Dist.spatial = False
Dimg_origin = Dist(img_base, curimgs.cuda())
#%% Mask the image before computing the distance
Dist.spatial = False
dist_mask = Dist(img_base * rfmaptsr, curimgs.cuda() * rfmaptsr)
#%% with spatial weighting of feature distance
Dist.spatial = True
distmaps = Dist(img_base, curimgs.cuda())
Dimg_weighted = (distmaps * rfmaptsr).sum(dim=[1,2,3]) / rfmaptsr.sum(dim=[1,2,3])
#%%