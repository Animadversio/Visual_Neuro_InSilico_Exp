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
from os.path import join
outdir = "H:\CNN-PCs"
#%%
# RGB_mean = torch.tensor([0.485, 0.456, 0.406])#.view(1,-1,1,1).cuda()
# RGB_std  = torch.tensor([0.229, 0.224, 0.225])#.view(1,-1,1,1).cuda()
# preprocess = Compose([ToTensor(),
#                       Resize(256, ),
#                       CenterCrop((256, 256), ),
#                       Normalize(RGB_mean, RGB_std),
#                       ])
# dataset = ImageFolder(r"E:\Datasets\imagenet-valid", transform=preprocess)
# loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
# #%%
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
# #%%
# fetcher = featureFetcher(model, device="cuda")
# fetcher.record(".Linearfc", return_input=True, ingraph=False)
# #%%
# n_components = 50
# ipca = IncrementalPCA(n_components=n_components, batch_size=40)
# #%%
# model.eval().cuda()
# feat_col = []
# update_freq = 20
# for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
#     with torch.no_grad():
#         model(imgtsr.cuda())
#     feats = fetcher[".Linearfc"][0].cpu()
#     feat_col.append(feats)
#     if (ibatch + 1) % update_freq == 0:
#         feattsr = torch.cat(tuple(feat_col), dim=0)
#         feats_ipca = ipca.partial_fit(feattsr)
#         feat_col = []
#
# feattsr = torch.cat(tuple(feat_col), dim=0)
# feats_ipca = ipca.partial_fit(feattsr)  # 1250it [36:21,  1.75s/it] for 50K images
# #%%
# np.savez(join(outdir,"IN-valid-resnet50_swsl-PC50.npz"), components=feats_ipca.components_,
#         expvar_r=feats_ipca.explained_variance_ratio_,
#         noise_var=feats_ipca.noise_variance_, SV=feats_ipca.singular_values_,
#         batch_size=feats_ipca.batch_size, N_sample=feats_ipca.n_samples_seen_)
# #%%
# components = feats_ipca.components_
#%%
data = np.load(join(outdir, "IN-valid-resnet50_swsl-PC50.npz"))
components = data["components"]

#%%
from Cosine.cosine_evol_lib import run_evol, sample_center_column_units_idx, set_objective, set_random_population_recording
from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
from insilico_Exp_torch import TorchScorer, visualize_trajectory, resize_and_pad_tsr
import time
def run_evol_lowmem(scorer, objfunc, optimizer, G, reckey=None, steps=100, label="obj-target-G", savedir="",
            RFresize=True, corner=(0, 0), imgsize=(224, 224), init_code=None):
    if init_code is None:
        init_code = np.zeros((1, G.codelen))
    RND = np.random.randint(1E5)
    new_codes = init_code
    # new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    actmat_all = []
    generations = []
    # codes_all = []
    best_imgs = []
    for i in range(steps,):
        # codes_all.append(new_codes.copy())
        T0 = time.time() #process_
        imgs = G.visualize_batch_np(new_codes)  # B=1
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        T1 = time.time() #process_
        if RFresize: imgs = resize_and_pad_tsr(imgs, imgsize, corner, )
        T2 = time.time() #process_
        _, recordings = scorer.score_tsr(imgs)
        actmat = recordings[reckey]
        T3 = time.time() #process_
        scores = objfunc(actmat, )  # targ_actmat
        T4 = time.time() #process_
        new_codes = optimizer.step_simple(scores, new_codes, )
        T5 = time.time() #process_
        if "BigGAN" in str(G.__class__):
            print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                latent_code[:, :128].norm(dim=1).mean()))
        else:
            print("step %d score %.3f (%.3f) (norm %.2f )" % (
                i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
        print(f"GANvis {T1-T0:.3f} RFresize {T2-T1:.3f} CNNforw {T3-T2:.3f}  "
            f"objfunc {T4-T3:.3f}  optim {T5-T4:.3f} total {T5-T0:.3f}")
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))
        best_imgs.append(imgs[scores.argmax(),:,:,:].detach().clone()) # debug at
        if i < steps - 1:
            del imgs
    # codes_all = np.concatenate(tuple(codes_all), axis=0)
    scores_all = np.array(scores_all)
    generations = np.array(generations)
    mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
    mtg_exp.save(join(savedir, "besteachgen_%s_%05d.jpg" % (label, RND,)))
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    mtg.save(join(savedir, "lastgen_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
    np.savez(join(savedir, "scores_%s_%05d.npz" % (label, RND)), generations=generations,
             scores_all=scores_all, actmat_all=actmat_all, codes_fin=new_codes)#, codes_all=codes_all)
    visualize_trajectory(scores_all, generations, title_str=label).savefig(
        join(savedir, "traj_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
    return new_codes, scores_all, generations, RND
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