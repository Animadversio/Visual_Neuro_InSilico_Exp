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
alpha = 5.0
dz_sigma = 2.1 # 0.4
dzs = dz_sigma * torch.randn(5, 4096).cuda()
dzs.requires_grad_()
optimizer = Adam([dzs], lr=0.05)
for i in tqdm(range(100)):
    optimizer.zero_grad()
    curimgs = G.visualize(z_base + dzs)
    resp_news = scorer.score_tsr_wgrad(curimgs, )
    score_loss = (resp_base - resp_news)
    img_dists = Dist(img_base, curimgs)
    loss = (score_loss - alpha * img_dists).mean()
    loss.backward()
    optimizer.step()
    print(f"{i}: {loss.item():.2f} new score {resp_news.mean().item():.2f} "
          f"old score {resp_base.item():.2f} img dist{img_dists.mean().item():.2f}")
#%%
from torch_utils import show_imgrid
show_imgrid(curimgs)
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_base.cpu().detach().numpy()[0].transpose(1, 2, 0))
axs[1].imshow(curimg.cpu().detach().numpy()[0].transpose(1, 2, 0))
plt.show()
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