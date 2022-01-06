"""
PCA of population activation, using incremental PCA
"""
import numpy as np
from tqdm import tqdm
import torch
from imageio import imread
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader  #, ImageDataset
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage, ToTensor, CenterCrop
from sklearn.decomposition import PCA, IncrementalPCA
from layer_hook_utils import featureFetcher, get_module_names, get_layer_names, register_hook_by_module_names
from featvis_lib import load_featnet
from os.path import join
from collections import defaultdict
outdir = "H:\CNN-PCs"
#%%
def create_imagenet_valid_dataset():
    RGB_mean = torch.tensor([0.485, 0.456, 0.406]) #.view(1,-1,1,1).cuda()
    RGB_std  = torch.tensor([0.229, 0.224, 0.225]) #.view(1,-1,1,1).cuda()
    preprocess = Compose([ToTensor(),
                          Resize(256, ),
                          CenterCrop((256, 256), ),
                          Normalize(RGB_mean, RGB_std),
                          ])
    dataset = ImageFolder(r"E:\Datasets\imagenet-valid", transform=preprocess)
    return dataset
#%

def slice_center_col(tsr, ingraph=False):
    if tsr.ndim == 4:
        B, C, H, W = tsr.shape
        Hi, Wi = int(H//2), int(W//2)
        return tsr[:, :, Hi, Wi].clone()
    elif tsr.ndim == 2:
        return tsr
    else:
        raise ValueError


def record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=125, num_workers=8):
    """Record the response for population of neurons to a dataset of images
    Higher level API

    Benchmark
    # 48:58 for 50K images on 1 GPU 1 worker loader.
    # 05:32 for 50K images on 1 GPU 5 worker loader.

    :param model:
    :param reclayers:
    :param dataset:
    :param return_input:
    :param batch_size:
    :param num_workers:
    :return: feattsrs
        format a dict with layer as keys, and
            feattsrs[layer] = feattsr # shape (imageN, featN)
    """
    loader = DataLoader(dataset, shuffle=False, drop_last=False,
                        batch_size=batch_size, num_workers=num_workers)

    fetcher = featureFetcher(model, device="cuda")
    for layer in reclayers:
        fetcher.record(layer, return_input=return_input, ingraph=False)
    feat_col = defaultdict(list)
    feattsrs = {}
    for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
        with torch.no_grad():
            model(imgtsr.cuda())

        for layer in reclayers:
            if return_input:
                feats_full = fetcher[layer][0].cpu()
            else:
                feats_full = fetcher[layer].cpu()
            feats = slice_center_col(feats_full, ingraph=False)
            feat_col[layer].append(feats)

    for layer in reclayers:
        feattsrs[layer] = torch.cat(tuple(feat_col[layer]), dim=0)

    return feattsrs


def feattsr_svd(feattsrs, device="cpu"):
    """
    :param feattsrs: output dict of `record_dataset`
    :param device:
    :return: tsr_svds
        format a dict with layer as keys, and
            feattsrs[layer] = feat_mean, U, S, V
        All are tensors with following shapes:
            feat_mean (featN,)
            U  (imageN, featN)
            S  (featN,)
            V  (featN, featN)
    """
    tsr_svds = {}
    for layer, feattsr in feattsrs.items():
        feat_mean = feattsr.to(device).mean(dim=0)
        U, S, V = (feattsr.to(device) - feat_mean).svd()
        tsr_svds[layer] = (feat_mean.to("cpu"), U.to("cpu"), S.to("cpu"), V.to("cpu"))
    return tsr_svds


#%% Load network
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
netname = "resnet50_linf8"
#%% Process images
dataset = create_imagenet_valid_dataset()
reclayers = [".layer2.Bottleneck2", ".layer3.Bottleneck2", ".layer4.Bottleneck2"]
feattsrs = record_dataset(model, reclayers, dataset, return_input=False,
                   batch_size=125, num_workers=8)
torch.save(feattsrs, join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))
#%% SVD of image tensor
tsr_svds = feattsr_svd(feattsrs, device="cuda")
torch.save(tsr_svds, join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))

#%% Load up svd tensor
netname = "resnet50_linf8"
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*tsr_svds.keys()]
#%%
feat_mean, U, S, V = tsr_svds[reclayers[0]]
#%%
targdir = V[:, 0]

#%%

import time
from os.path import join
import matplotlib.pylab as plt
import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, ToPILImage
from layer_hook_utils import get_module_names, get_layer_names
from torch.optim import Adam, SGD
import torch.nn.functional as F
# from insilico_Exp_torch import TorchScorer, visualize_trajectory, resize_and_pad_tsr

#%% Visualize a target direction at a given layer
def save_imgtsr(finimgs, figdir:str ="", savestr:str =""):
    """
    finimgs: a torch tensor on cpu with shape B,C,H,W.
    """
    B = finimgs.shape[0]
    for imgi in range(B):
        ToPILImage()(finimgs[imgi,:,:,:]).save(join(figdir, "%s_%02d.png"%(savestr, imgi)))


def featdir_GAN_visualize(G, CNNnet, layername, objfunc, tfms=[], imgfullpix=256, maximize=True, 
    use_adam=True, lr=0.01, langevin_eps=0.0, MAXSTEP=100, Bsize=5, saveImgN=None,
    savestr="", figdir="", imshow=False, PILshow=False, verbose=True, saveimg=False):
    """ Visualize the features carried by the scorer.  """
    # scorer.mode = score_mode

    return_input = False
    fetcher = featureFetcher(CNNnet, device="cuda", print_module=False)
    fetcher.record(layername, return_input=return_input, ingraph=True)
    score_sgn = -1 if maximize else 1
    z = 0.5*torch.randn([Bsize, 4096]).cuda()
    z.requires_grad_(True)
    optimizer = Adam([z], lr=lr) if use_adam else SGD([z], lr=lr)
    tfms_f = Compose(tfms)
    score_traj = []
    pbar = tqdm(range(MAXSTEP))
    for step in pbar:
        x = G.visualize(z, scale=1.0)
        ppx = tfms_f(x)
        ppx = F.interpolate(x, [imgfullpix, imgfullpix], mode="bilinear", align_corners=True)
        optimizer.zero_grad()
        CNNnet(ppx)
        if return_input:
            feats_full = fetcher[layername][0] # .cpu()
        else:
            feats_full = fetcher[layername] # .cpu()
        feats = slice_center_col(feats_full, )
        score = score_sgn * objfunc(feats, )
        score.sum().backward()
        z.grad = z.norm(dim=1, keepdim=True) / z.grad.norm(dim=1, keepdim=True) * z.grad  # this is a gradient normalizing step 
        optimizer.step()
        score_traj.append(score.detach().clone().cpu())
        if langevin_eps > 0.0:
            # if > 0 then add noise to become Langevin gradient descent jump minimum
            z.data.add_(torch.randn(z.shape, device="cuda") * langevin_eps)
        if verbose and step % 10 == 0:
            print("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))
        pbar.set_description("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))

    final_score = score_sgn * score.detach().clone().cpu()
    del score
    torch.cuda.empty_cache()
    if maximize:
        idx = torch.argsort(final_score, descending=True)
    else:
        idx = torch.argsort(final_score, descending=False)
    score_traj = score_sgn * torch.stack(tuple(score_traj))[:, idx]
    finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
    print("Final scores %s"%(" ".join("%.2f" % s for s in final_score[idx])))
    mtg = ToPILImage()(make_grid(finimgs))
    if PILshow: mtg.show()
    mtg.save(join(figdir, "%s_G_%s.png"%(savestr, layername)))
    np.savez(join(figdir, "%s_G_%s.npz"%(savestr, layername)), z=z.detach().cpu().numpy(), score_traj=score_traj.numpy())
    if imshow:
        plt.figure(figsize=[Bsize*2, 2.3])
        plt.imshow(mtg)
        plt.axis("off")
        plt.show()
    if saveimg:
        os.makedirs(join(figdir, "img"), exist_ok=True)
        if saveImgN is None:
            save_imgtsr(finimgs, figdir=join(figdir, "img"), savestr="%s"%(savestr))
        else:
            save_imgtsr(finimgs[:saveImgN,:,:,:], figdir=join(figdir, "img"), savestr="%s"%(savestr))
            mtg_sel = ToPILImage()(make_grid(finimgs[:saveImgN,:,:,:]))
            mtg_sel.save(join(figdir, "%s_G_%s_best.png" % (savestr, layername)))
    return finimgs, mtg, score_traj


def set_objective_torch(score_method, targdir, featmean=None, normalize=False):
    if featmean is None:
        centralize = False 
    else:
        centralize = True

    def objfunc(feats):
        featmat = feats.view([-1, targdir.size(0)]) # [1 by feat num]
        targmat = targdir.unsqueeze(0) # [1 by feat num]
        # if normalize:
        #     actmat_msk = (actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
        #     targmat_msk = (targmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
        if centralize:
            featmat = featmat - featmean
            targmat = targmat - featmean

        if score_method == "L1":
            scores = - (featmat - targmat).abs().mean(dim=1)
        elif score_method == "MSE":
            scores = - (featmat - targmat).pow(2).mean(dim=1)
        elif score_method == "corr":
            featmat = featmat - featmat.mean(dim=1, keepdim=True)
            targmat = targmat - targmat.mean(dim=1, keepdim=True)
            feat_norm = featmat.norm(dim=1, keepdim=True)
            targ_norm = targmat.norm(dim=1, keepdim=True)
            scores = ((featmat @ targmat.T) / feat_norm / targ_norm).squeeze(dim=1)
        elif score_method == "cosine":
            feat_norm = featmat.norm(dim=1, keepdim=True)
            targ_norm = targmat.norm(dim=1, keepdim=True)
            scores = ((featmat @ targmat.T) / feat_norm / targ_norm).squeeze(dim=1)
        elif score_method == "dot":
            scores = (featmat @ targmat.T).squeeze(dim=1)
        else:
            raise ValueError
        return scores # (Nimg, ) 1d array
    # return an array / tensor of scores for an array of activations
    # Noise form
    return objfunc
#%%
from GAN_utils import upconvGAN
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
model.requires_grad_(False)
netname = "resnet50_linf8"
#%%
def shorten_layername(lname):
    return lname.replace(".Bottleneck", '-Btn').replace(".layer", "layer")
#%%
score_method = "cosine"
for PCi in range(50, 100):
    for layeri in range(3):
        layername = reclayers[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=outdir,
                                             savestr="%s_%s_PC%03d_pos_%s"%(netname, layersn, PCi, score_method),
                                             lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8,)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=outdir,
                                             savestr="%s_%s_PC%03d_neg_%s"%(netname, layersn, PCi, score_method),
                                             lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8,)

#%% Dev zone, working pipeline, process dataset to get
#
# loader = DataLoader(dataset, batch_size=125, shuffle=False, drop_last=False, num_workers=8)
# reclayers = [".layer2.Bottleneck2", ".layer3.Bottleneck2", ".layer4.Bottleneck2"]
# return_input = False
# fetcher = featureFetcher(model, device="cuda")
# for layer in reclayers:
#     fetcher.record(layer, return_input=return_input, ingraph=False)
# # fetcher.record(".Linearfc", return_input=True, ingraph=False)
# feat_col = defaultdict(list)
# feattsrs = {}
# for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
#     with torch.no_grad():
#         model(imgtsr.cuda())
#
#     for layer in reclayers:
#         if return_input:
#             feats_full = fetcher[layer][0].cpu()
#         else:
#             feats_full = fetcher[layer].cpu()
#         feats = slice_center_col(feats_full, ingraph=False)
#         feat_col[layer].append(feats)
#
# for layer in reclayers:
#     feattsrs[layer] = torch.cat(tuple(feat_col[layer]), dim=0)
#%%

#%% Obsolete, Incremental PCA is not useful in this number of images....
#%% batch svd / PCA
# n_components = 50
# ipca = IncrementalPCA(n_components=n_components, batch_size=400)
# feat_col = []
# update_freq = 5
# for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
#     with torch.no_grad():
#         model(imgtsr.cuda())
#     feats = fetcher[".Linearfc"][0].cpu()
#     feat_col.append(feats)
#     if (ibatch + 1) % update_freq == 0:
#         feattsr = torch.cat(tuple(feat_col), dim=0)
#         feats_ipca = ipca.partial_fit(feattsr)
#         feat_col = []

# feattsr = torch.cat(tuple(feat_col), dim=0)
# feats_ipca = ipca.partial_fit(feattsr)  # 1250it [36:21,  1.75s/it] for 50K images
# #%%
# # netname = "resnet50_swsl"
# netname = "resnet50_linf8"
# np.savez(join(outdir,"IN-valid-%s-PC50.npz"%netname), components=feats_ipca.components_,
#         expvar_r=feats_ipca.explained_variance_ratio_,
#         noise_var=feats_ipca.noise_variance_, SV=feats_ipca.singular_values_,
#         batch_size=feats_ipca.batch_size, N_sample=feats_ipca.n_samples_seen_)
# #%%
# components = feats_ipca.components_
# #%%
# data = np.load(join(outdir, "IN-valid-resnet50_swsl-PC50.npz"))
# components = data["components"]

#%%
from Cosine.cosine_evol_lib import run_evol, sample_center_column_units_idx, set_objective, set_random_population_recording
from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
from insilico_Exp_torch import TorchScorer, visualize_trajectory, resize_and_pad_tsr
import time
# def run_evol_lowmem(scorer, objfunc, optimizer, G, reckey=None, steps=100, label="obj-target-G", savedir="",
#             RFresize=True, corner=(0, 0), imgsize=(224, 224), init_code=None):
#     if init_code is None:
#         init_code = np.zeros((1, G.codelen))
#     RND = np.random.randint(1E5)
#     new_codes = init_code
#     # new_codes = init_code + np.random.randn(25, 256) * 0.06
#     scores_all = []
#     actmat_all = []
#     generations = []
#     # codes_all = []
#     best_imgs = []
#     for i in range(steps,):
#         # codes_all.append(new_codes.copy())
#         T0 = time.time() #process_
#         imgs = G.visualize_batch_np(new_codes)  # B=1
#         latent_code = torch.from_numpy(np.array(new_codes)).float()
#         T1 = time.time() #process_
#         if RFresize: imgs = resize_and_pad_tsr(imgs, imgsize, corner, )
#         T2 = time.time() #process_
#         _, recordings = scorer.score_tsr(imgs)
#         actmat = recordings[reckey]
#         T3 = time.time() #process_
#         scores = objfunc(actmat, )  # targ_actmat
#         T4 = time.time() #process_
#         new_codes = optimizer.step_simple(scores, new_codes, )
#         T5 = time.time() #process_
#         if "BigGAN" in str(G.__class__):
#             print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
#                 i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
#                 latent_code[:, :128].norm(dim=1).mean()))
#         else:
#             print("step %d score %.3f (%.3f) (norm %.2f )" % (
#                 i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
#         print(f"GANvis {T1-T0:.3f} RFresize {T2-T1:.3f} CNNforw {T3-T2:.3f}  "
#             f"objfunc {T4-T3:.3f}  optim {T5-T4:.3f} total {T5-T0:.3f}")
#         scores_all.extend(list(scores))
#         generations.extend([i] * len(scores))
#         best_imgs.append(imgs[scores.argmax(),:,:,:].detach().clone()) # debug at
#         if i < steps - 1:
#             del imgs
#     # codes_all = np.concatenate(tuple(codes_all), axis=0)
#     scores_all = np.array(scores_all)
#     generations = np.array(generations)
#     mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
#     mtg_exp.save(join(savedir, "besteachgen_%s_%05d.jpg" % (label, RND,)))
#     mtg = ToPILImage()(make_grid(imgs, nrow=7))
#     mtg.save(join(savedir, "lastgen_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
#     np.savez(join(savedir, "scores_%s_%05d.npz" % (label, RND)), generations=generations,
#              scores_all=scores_all, actmat_all=actmat_all, codes_fin=new_codes)#, codes_all=codes_all)
#     visualize_trajectory(scores_all, generations, title_str=label).savefig(
#         join(savedir, "traj_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
#     return new_codes, scores_all, generations, RND
#%%
score_method = "cosine"
popul_mask = np.ones((2048,), dtype=bool)
popul_m = np.zeros((1, 2048,),)
popul_s = np.ones((1, 2048,),)

scorer = TorchScorer("resnet50_linf8")
module_names, module_types, module_spec = get_module_names(scorer.model, (3, 256, 256), "cuda", False)
unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [".layer4.Bottleneck2"], randomize=False)
#
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
code_length = G.codelen
objfunc = set_objective(score_method, components[1:2, :], popul_mask, popul_m, popul_s)
optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                maximize=True, random_seed=None, optim_params={})
codes_all, scores_all, actmat_all, generations, RND = run_evol(scorer, objfunc, optimizer, G,
                                   reckey=".layer4.Bottleneck2",
                               label="PC2-cosine", savedir=r"H:\CNN-PCs\resnet50-PC1",
                               steps=100, RFresize=False, corner=(0, 0), imgsize=(256, 256))
# codes_fin, scores_all, generations, RND = run_evol_lowmem
#corner=(20, 20), imgsize=(187, 187))