"""Find RBF approximation of CNN activations"""
import torch
import torchvision.models as fit_models
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from featvis_lib import load_featnet
from layer_hook_utils import featureFetcher
from ZO_HessAware_Optimizers import CholeskyCMAES
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
from collections import defaultdict
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
from neural_regress.insilico_modelling_lib import compare_activation_prediction, sweep_regressors, \
    resizer, normalizer, denormalizer, PoissonRegressor, RidgeCV, Ridge, KernelRidge
#%%
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=False)
#%%
regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer,)
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
    # with torch.no_grad():
    #     featnet(scorer.preprocess(imgs, input_scale=1.0))
    #
    # del imgs
    # pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    # feattsr = featFetcher[regresslayer][:, :, 6, 6]
    # feattsr_all.append(feattsr.cpu().numpy())
    resp_all.append(resp)
    z_all.append(z_arr)
    print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")

resp_all = np.concatenate(resp_all, axis=0)
z_all = np.concatenate(z_all, axis=0)
# feattsr_all = np.concatenate(feattsr_all, axis=0)

z_base = torch.tensor(z_all.mean(axis=0, keepdims=True)).float().cuda()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )
#%% Maximize invariance
from torch.optim import Adam
from lpips import LPIPS
Dist = LPIPS(net="squeeze", ).cuda()
Dist.requires_grad_(False)
#%%
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=True)

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
from torch_utils import show_imgrid, save_imgrid
# show_imgrid(curimgs)
#%%
def latent_diversity_explore(G, scorer, z_base, dzs=None, alpha=10.0, dz_sigma=3.0,
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
        img_dists = Dist(img_base, curimgs)
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()
#%%
out_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5"
dz_init_col = []
dz_col = []
score_col = []
for i in range(200):
    dzs_init, dzs, curimgs, scores = latent_diversity_explore(G, scorer, z_base,
                  alpha=10.0, dz_sigma=2.0, batch_size=5, steps=150, lr=0.1, )
    save_imgrid(curimgs, join(out_dir, f"proto_divers_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
#%%
dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec},
           join(out_dir, "diversity_dz_score.pt"))
#%%
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, show_gradmap
cent_pos = (6, 6)
outrf_dir = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
gradAmpmap = grad_RF_estimate(scorer.model, scorer.layer, (slice(None), cent_pos[0], cent_pos[1]),
                          input_size=(3, 227, 227), device="cuda", show=False, reps=200, batch=4)
show_gradmap(gradAmpmap, )
fitdict = fit_2dgauss(gradAmpmap, "layer3-Btn5-5", outdir=outrf_dir, plot=True)
#%%
rfmap = fitdict.fitmap
rfmap /= rfmap.max()
rfmaptsr = torch.from_numpy(rfmap).float().cuda().unsqueeze(0).unsqueeze(0)
rfmaptsr = torch.nn.functional.interpolate(rfmaptsr,
               (256, 256), mode="bilinear", align_corners=True)
rfmap_full = rfmaptsr.cpu()[0,0,:].unsqueeze(2).numpy()
#%% Trying different spatial weighted Distance computation
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

def latent_diversity_explore_wRF(G, scorer, z_base, rfmaptsr, dzs=None, alpha=10.0, dz_sigma=3.0,
                      batch_size=5, steps=150, lr=0.1, midpoint=True):
    """

    :param G:
    :param scorer:
    :param z_base:
    :param rfmaptsr: We assume its shape is (1, 1, 256, 256), and its values are in [0, 1]
    :param dzs:
    :param alpha: The weight of the distance term VS the score term
    :param dz_sigma: The initial std of dz.
    :param batch_size:
    :param steps:
    :param lr:
    :param midpoint:
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
        img_dists = Dist(img_base * rfmaptsr, curimgs * rfmaptsr)
        loss = (score_loss + score_mid_loss - alpha * img_dists).mean()
        loss.backward()
        optimizer.step()
        print(
            f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} mid score {resp_news_mid.mean().item():.2f} "
            f"old score {resp_base.item():.2f} RF img dist{img_dists.mean().item():.2f}")
    return dzs_init, dzs.detach().cpu(), \
           curimgs.detach().cpu(), resp_news.detach().cpu()

#%%
out_dir_rf = r"E:\insilico_exps\proto_diversity\resnet50_linf8\layer3-Btn5-5_rf"
dz_init_col = []
dz_col = []
score_col = []
for i in range(200):
    dzs_init, dzs, curimgs, scores = latent_diversity_explore_wRF(G, scorer, z_base,
                  rfmaptsr, alpha=10.0, dz_sigma=2.0, batch_size=5, steps=150, lr=0.1, )
    save_imgrid(curimgs, join(out_dir_rf, f"proto_divers_{i}.png"))
    save_imgrid(curimgs*rfmaptsr.cpu(), join(out_dir_rf, f"proto_divers_wRF_{i}.png"))
    dz_init_col.append(dzs_init)
    dz_col.append(dzs)
    score_col.append(scores)
#%%
dz_init_tsr = torch.cat(dz_init_col, dim=0)
dz_final_tsr = torch.cat(dz_col, dim=0)
score_vec = torch.cat(score_col, dim=0)
torch.save({"dz_init": dz_init_tsr, "dz_final": dz_final_tsr, "score": score_vec},
           join(out_dir_rf, "diversity_dz_score.pt"))


#%%
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
# optimizer = Adam([dz], lr=0.01)
# pbar = tqdm(range(100))
# for i in pbar:
#     with torch.no_grad():
#
#     loss =
#%%
#%% New iteration
z_base = z_base + dz.detach()
img_base = G.visualize(z_base)
resp_base = scorer.score_tsr_wgrad(img_base, )