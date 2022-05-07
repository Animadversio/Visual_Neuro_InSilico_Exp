import os
import torch
from os.path import join
from hdf5storage import loadmat
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from alexnet_model import load_alexnet_matlab
from layer_hook_utils import featureFetcher
from torch_utils import show_imgrid, save_imgrid
import json, yaml
#%%
from tqdm import tqdm
from torch.optim import Adam
from ZO_HessAware_Optimizers import CholeskyCMAES
from Cosine.cosine_evol_lib import run_evol, set_objective, set_objective_grad
#%%
net = load_alexnet_matlab()
G = upconvGAN("fc6").eval().cuda()
G.requires_grad_(False)
#%%
fc6name = ".Linearfc6"
scorer = TorchScorer(net, imgpix=256)
scorer.set_recording([fc6name], allow_grad=True)
# fetcher = featureFetcher(net, )
# fetcher.record(fc6name)
#%% Load in data
projroot = r"E:\OneDrive - Harvard University\GAC_reconstruct"
matfile = "activations-alexnet-fc6.mat"
activ_matrix = loadmat(join(projroot, matfile))['activations_all']
activ_matrix = activ_matrix.T
activ_tsr = torch.tensor(activ_matrix).float()
imgnum = activ_matrix.shape[0]
assert activ_matrix.shape[1] == 4096
#%%
run_name = r"GAN_mapping"
os.makedirs(join(projroot, run_name), exist_ok=True)
for imgi in range(imgnum):
    img = G.visualize(activ_tsr[imgi, :].cuda()/4)
    save_imgrid(img, join(projroot, run_name, "img{:04d}.png".format(imgi)))
    # activ_matrix[imgi, :] = activ_matrix[imgi, :] / activ_matrix[imgi, :].sum()
#%%
run_name = r"GAN_evol_MSE"
expdir = join(projroot, run_name)
os.makedirs(expdir, exist_ok=True)
imgi = 61
optimizer = CholeskyCMAES(4096, init_sigma=3.0, maximize=True)
objective = set_objective("cosine", activ_matrix[imgi:imgi+1, :],
              popul_mask=slice(None), popul_m=None, popul_s=None, normalize=False)
codes_all, scores_all, actmat_all, generations, RND = \
    run_evol(scorer, objective, optimizer, G, reckey=fc6name, steps=100,
         label=f"img_{imgi+1}", savedir=expdir,
         RFresize=False, corner=(0, 0), imgsize=(227, 227), init_code=None)
#%%
scorer = TorchScorer(net, imgpix=256)
scorer.set_recording([fc6name], allow_grad=True)
#%%
ImageNet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
#%%
def cosine_evol_alexnet(G, net, activ_vector, scoretype, label="", expdir="",):
    objective = set_objective(scoretype, activ_vector,
                            popul_mask=None, popul_m=None, popul_s=None,
                            normalize=False, )
    optimizer = CholeskyCMAES(4096, init_sigma=1.0, maximize=True)
    zs = torch.randn(40, 4096, device="cuda")
    pbar = tqdm(range(100))
    for _ in pbar:
        imgs = G.visualize(zs)
        ppimgs = (imgs - ImageNet_mean) * 255
        layerout = net(ppimgs, output_layers=["fc6"])
        scores = objective(layerout["fc6"].cpu().numpy())
        zs_new = optimizer.step_simple(scores, zs.cpu().numpy())
        zs = torch.tensor(zs_new).float().to("cuda")
        pbar.set_description(f"score: {scores.mean():.3f}+-{scores.std():.3f}")

    scores = torch.tensor(scores)
    _, sortidx = scores.sort(descending=True)
    save_imgrid(imgs[sortidx, :], join(expdir, f"{label}_lastgen.png"), nrow=5,)
    save_imgrid(imgs[sortidx[0], :], join(expdir, f"{label}_bestimg.png"), nrow=5,)
    torch.save({'zs': zs.cpu(), "scores":scores.cpu()}, join(expdir, f"{label}_zs.pt"))
    return imgs.cpu(), scores, zs.cpu()


def grad_cosine_evol_alexnet(G, net, activ_vector, scoretype, label="", expdir="",):
    gradobj = set_objective_grad(scoretype, activ_vector,
                            popul_mask=None, popul_m=None, popul_s=None,
                            normalize=False, device="cuda")
    zs = torch.randn(10, 4096, device="cuda")
    zs.requires_grad_(True)
    optimizer = Adam([zs], lr=0.08)
    pbar = tqdm(range(250))
    for _ in pbar:
        optimizer.zero_grad()
        imgs = G.visualize(zs)
        ppimgs = (imgs - ImageNet_mean) * 255
        layerout = net(ppimgs, output_layers=["fc6"])
        scores = gradobj(layerout["fc6"])
        loss = - scores.mean()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"score: {scores.mean():.3f}+-{scores.std():.3f}")

    _, sortidx = scores.sort(descending=True)
    save_imgrid(imgs[sortidx, :], join(expdir, f"{label}_lastgen.png"), nrow=5,)
    save_imgrid(imgs[sortidx[0], :], join(expdir, f"{label}_bestimg.png"), nrow=5,)
    torch.save({'zs': zs.detach().cpu(), "scores":scores.detach().cpu()}, join(expdir, f"{label}_zs.pt"))
    return imgs.detach().cpu(), scores.detach().cpu(), zs.detach().cpu()

#%%
#%
run_name = r"GAN_Evol_cosine"
expdir = join(projroot, run_name)
os.makedirs(expdir, exist_ok=True)
for imgi in range(imgnum):
    cosine_evol_alexnet(G, net, activ_matrix[imgi:imgi+1, :], scoretype="cosine",
                         label=f"img_{imgi+1:02d}", expdir=expdir,)
#%
run_name = r"GAN_Evol_MSE"
expdir = join(projroot, run_name)
os.makedirs(expdir, exist_ok=True)
for imgi in range(imgnum):
    cosine_evol_alexnet(G, net, activ_matrix[imgi:imgi+1, :], scoretype="MSE",
                         label=f"img_{imgi+1:02d}", expdir=expdir,)

run_name = r"GAN_Adam_cosine"
expdir = join(projroot, run_name)
os.makedirs(expdir, exist_ok=True)
for imgi in range(imgnum):
    grad_cosine_evol_alexnet(G, net, activ_tsr[imgi:imgi+1, :], scoretype="cosine",
                         label=f"img_{imgi+1:02d}", expdir=expdir,)
#%
run_name = r"GAN_Adam_MSE"
expdir = join(projroot, run_name)
os.makedirs(expdir, exist_ok=True)
for imgi in range(imgnum):
    grad_cosine_evol_alexnet(G, net, activ_tsr[imgi:imgi+1, :], scoretype="MSE",
                         label=f"img_{imgi+1:02d}", expdir=expdir,)

#%%
# _, recording = scorer.score_tsr_wgrad(imgs)
# scores = gradobj(recording[fc6name][0])
import yaml
for obj in ["cosine", "MSE"]:
    run_name = r"GAN_Adam_" + obj
    config_dict = dict(author="BXW",
                       vision_model="alexnet_py",
                       image_space="FC6GAN",
                       image_space_info=dict(training_dataset="ImageNet",),
                       optimizer="Adam",
                       optimizer_info=dict(lr=0.08, iterations=250, init_z_sigma=1.0),
                       distance=obj,
                       description="",)
    yaml.dump(config_dict, open(join(projroot, run_name, "config.yml"), "w"))

for obj in ["cosine", "MSE"]:
    run_name = r"GAN_evol_" + obj
    config_dict = dict(author="BXW",
                       vision_model="alexnet_py",
                       image_space="FC6GAN",
                       image_space_info=dict(training_dataset="ImageNet",),
                       optimizer="CholeskyCMAES",
                       optimizer_info=dict(init_sigma=1.0, iterations=100, ),
                       distance=obj,
                       description="",)
    yaml.dump(config_dict, open(join(projroot, run_name, "config.yml"), "w"))

