""" Cluster version of BigGAN Evol """
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings, get_full_hessian
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from skimage.io import imsave
from build_montages import build_montages, color_framed_montages
import torchvision.models as tv
from torchvision.utils import make_grid
from pytorch_pretrained_biggan.model import BigGAN, BigGANConfig
from pytorch_pretrained_biggan.utils import truncated_noise_sample, save_as_images, one_hot_from_names
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
import os
import tqdm
from cma import CMAEvolutionStrategy
from ZO_HessAware_Optimizers import CholeskyCMAES
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN
#%
def visualize_trajectory(scores_all, generations, codes_arr=None, show=False, title_str=""):
    """ Visualize the Score Trajectory """
    gen_slice = np.arange(min(generations), max(generations) + 1)
    AvgScore = np.zeros_like(gen_slice)
    MaxScore = np.zeros_like(gen_slice)
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh, ax = plt.subplots()
    ax1 = ax.twinx()
    ax1.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    ax1.plot(gen_slice, AvgScore, color='black', label="Average score")
    ax1.plot(gen_slice, MaxScore, color='red', label="Max score")
    ax1.set_xlabel("generation #")
    ax1.set_ylabel("CNN unit score")
    plt.legend()
    if codes_arr is not None:
        ax2 = ax.twinx()
        if codes_arr.shape[1] == 256: # BigGAN
            nos_norm = np.linalg.norm(codes_arr[:, :128], axis=1)
            cls_norm = np.linalg.norm(codes_arr[:, 128:], axis=1)
            ax2.plot(generations, nos_norm, color="orange", label="noise", alpha=0.7)
            ax2.plot(generations, cls_norm, color="magenta", label="class", alpha=0.7)
        elif codes_arr.shape[1] == 4096: # FC6GAN
            norms_all = np.linalg.norm(codes_arr[:, :], axis=1)
            ax2.plot(generations, norms_all, color="magenta", label="all", alpha=0.7)
        ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
        plt.legend()
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    if show:
        plt.show()
    return figh
#%%
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["CholCMA"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=5, help="")
args = parser.parse_args([])
#%% Select GAN
from GAN_utils import BigGAN_wrapper, upconvGAN
from insilico_Exp import TorchScorer, ExperimentEvolve
if args.G == "BigGAN":
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    BGAN.eval().cuda()
    for param in BGAN.parameters():
        param.requires_grad_(False)
    G = BigGAN_wrapper(BGAN)
elif args.G == "fc6":
    G = upconvGAN("fc6")
    G.eval().cuda()
    for param in G.parameters():
        param.requires_grad_(False)
#%%
# net = tv.alexnet(pretrained=True)
scorer = TorchScorer(args.net)
scorer.select_unit(("alexnet", "fc6", 2))
#%%
imgs = G.visualize(torch.randn(3, 256).cuda()).cpu()
scores = scorer.score_tsr(imgs)
#%%
if sys.platform == "linux":
    rootdir = r"/scratch/binxu/BigGAN_Optim_Tune"
else:
    rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune"

for unit_id in range(args.chans[0], args.chans[1]):
    unit = (args.net, args.layer, unit_id)
    scorer.select_unit(unit)
    savedir = join(rootdir, r"%s_%s_%d"%unit[:3])
    os.makedirs(savedir, exist_ok=True)
    for triali in range(args.reps):
        if args.G == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
        RND = np.random.randint(1E5)
        for optim in :
            optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)
            new_codes = init_code + np.random.randn(25, 256) * 0.06
            scores_all = []
            generations = []
            for i in range(args.steps,):
                imgs = G.visualize_batch_np(new_codes, B=10)
                latent_code = torch.from_numpy(np.array(new_codes)).float()
                scores = scorer.score_tsr(imgs)
                if args.G == "BigGAN":
                    print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                        i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                        latent_code[:, :128].norm(dim=1).mean()))
                else:
                    print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                        i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                        latent_code[:, :128].norm(dim=1).mean()))
                new_codes = optim_cust.step_simple(scores, new_codes, )
                scores_all.extend(list(scores))
                generations.extend([i] * len(scores))

            scores_all = np.array(scores_all)
            generations = np.array(generations)
            mtg = ToPILImage()(make_grid(imgs, nrow=7))
            mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
            np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=latent_code.cpu().numpy())
            visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
                join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))











#%%
cmasteps = 100
for unit_id in range(30):
    unit = ("alexnet", "fc6", unit_id)
    scorer.select_unit(unit)
    savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"%unit
    os.makedirs(savedir, exist_ok=True)
    for triali in range(5):
        fixnoise = 0.7 * truncated_noise_sample(1, 128)
        RND = np.random.randint(1E5)
        #%%
        methodlab = "CMA_prod"
        optim_noise = CMAEvolutionStrategy(fixnoise, 0.4)#0.4)  # 0.2
        optim_class = CMAEvolutionStrategy(128 * [0.0], 0.2)
        scores_all = []
        generations = []
        for i in tqdm.trange(cmasteps, desc="CMA steps"):
            class_codes = optim_class.ask()
            noise_codes = optim_noise.ask()
            codes_tsr = torch.from_numpy(np.array(class_codes)).float()
            noise_tsr = torch.from_numpy(np.array(noise_codes)).float()
            latent_code = torch.cat((noise_tsr, codes_tsr), dim=1).cuda()  # this initialize inner loop
            with torch.no_grad():
                imgs = G.visualize(latent_code).cpu()
            scores = scorer.score_tsr(imgs)
            print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), codes_tsr.norm(dim=1).mean(), noise_tsr.norm(dim=1).mean()))
            optim_class.tell(class_codes, -scores)
            optim_noise.tell(noise_codes, -scores)
            scores_all.extend(list(scores))
            generations.extend([i]*len(scores))

        scores_all = np.array(scores_all)
        generations = np.array(generations)
        mtg = ToPILImage()(make_grid(imgs,nrow=6))
        mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))
        np.savez(join(savedir, "scores%s_%05d.npz"%(methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=latent_code.cpu().numpy())
        visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(join(savedir, "traj%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))
        #%%
        methodlab = "CMA_class"
        optim_class = CMAEvolutionStrategy(128 * [0.0], 0.2)
        scores_all = []
        generations = []
        for i in tqdm.trange(cmasteps, desc="CMA steps"):
            class_codes = optim_class.ask()
            codes_tsr = torch.from_numpy(np.array(class_codes)).float()
            noise_tsr = torch.from_numpy(fixnoise).repeat(codes_tsr.shape[0], 1).float()
            latent_code = torch.cat((noise_tsr, codes_tsr), dim=1).cuda()  # this initialize inner loop
            with torch.no_grad():
                imgs = G.visualize(latent_code).cpu()
            scores = scorer.score_tsr(imgs)
            print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), codes_tsr.norm(dim=1).mean(), noise_tsr.norm(dim=1).mean()))
            optim_class.tell(class_codes, -scores)
            scores_all.extend(list(scores))
            generations.extend([i]*len(scores))

        scores_all = np.array(scores_all)
        generations = np.array(generations)
        mtg = ToPILImage()(make_grid(imgs,nrow=6))
        mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))
        np.savez(join(savedir, "scores%s_%05d.npz"%(methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=latent_code.cpu().numpy())
        visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(join(savedir, "traj%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))
        #%%
        methodlab = "CMA_all"
        optim_all = CMAEvolutionStrategy(list(fixnoise[0]) + 128 * [0.0], 0.2)
        scores_all = []
        generations = []
        for i in tqdm.trange(cmasteps, desc="CMA steps"):
            all_codes = optim_all.ask()
            latent_code = torch.from_numpy(np.array(all_codes)).float().cuda()
            with torch.no_grad():
                imgs = G.visualize(latent_code).cpu()
            scores = scorer.score_tsr(imgs)
            print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),  latent_code[:, :128].norm(dim=1).mean()))
            optim_all.tell(all_codes, -scores)
            scores_all.extend(list(scores))
            generations.extend([i]*len(scores))

        scores_all = np.array(scores_all)
        generations = np.array(generations)
        mtg = ToPILImage()(make_grid(imgs,nrow=7))
        mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))
        np.savez(join(savedir, "scores%s_%05d.npz"%(methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=latent_code.cpu().numpy())
        visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(join(savedir, "traj%s_%05d_score%.1f.jpg"%(methodlab, RND, scores.mean())))

        #%%
        methodlab = "CholCMA"
        init_code = np.concatenate((fixnoise, np.zeros((1, 128))),axis=1)
        optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2)
        new_codes = init_code + np.random.randn(25, 256)*0.06
        scores_all = []
        generations = []
        for i in tqdm.trange(cmasteps, desc="CMA steps"):
            imgs = G.visualize_batch_np(new_codes, B=10)
            latent_code = torch.from_numpy(np.array(new_codes)).float()
            scores = scorer.score_tsr(imgs)
            print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),  latent_code[:, :128].norm(dim=1).mean()))
            new_codes = optim_cust.step_simple(scores, new_codes, )
            scores_all.extend(list(scores))
            generations.extend([i] * len(scores))

        scores_all = np.array(scores_all)
        generations = np.array(generations)
        mtg = ToPILImage()(make_grid(imgs, nrow=7))
        mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
        np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all,
                 codes_fin=latent_code.cpu().numpy())
        visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
            join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
        #%%
        methodlab = "CholCMA_noA"
        init_code = np.concatenate((fixnoise, np.zeros((1, 128))),axis=1)
        optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)
        new_codes = init_code + np.random.randn(25, 256)*0.06
        scores_all = []
        generations = []
        for i in tqdm.trange(cmasteps, desc="CMA steps"):
            imgs = G.visualize_batch_np(new_codes, B=10)
            latent_code = torch.from_numpy(np.array(new_codes)).float()
            scores = scorer.score_tsr(imgs)
            print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),  latent_code[:, :128].norm(dim=1).mean()))
            new_codes = optim_cust.step_simple(scores, new_codes, )
            scores_all.extend(list(scores))
            generations.extend([i] * len(scores))

        scores_all = np.array(scores_all)
        generations = np.array(generations)
        mtg = ToPILImage()(make_grid(imgs, nrow=7))
        mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
        np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all,
                 codes_fin=latent_code.cpu().numpy())
        visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
            join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
#%%
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
for param in G.parameters():
    param.requires_grad_(False)
G.eval().cuda()
# net = tv.alexnet(pretrained=True)
from insilico_Exp import TorchScorer, ExperimentEvolve
scorer = TorchScorer("alexnet")
scorer.select_unit(("alexnet", "fc6", 2))
from ZO_HessAware_Optimizers import CholeskyCMAES
#%% FC6 Evolution.
import os
unit = ("alexnet", "fc6", 2)
scorer.select_unit(unit)
savedir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune\%s_%s_%d"%unit
os.makedirs(savedir, exist_ok=True)
cmasteps= 100
methodlab = "CholCMA_noA_fc6"
init_code = np.zeros((1, 4096))
RND = np.random.randint(1E5)
optim_cust = CholeskyCMAES(space_dimen=4096, init_code=init_code, init_sigma=3, Aupdate_freq=102)
new_codes = init_code + np.random.randn(30, 4096) * 3
scores_all = []
generations = []
for i in tqdm.trange(cmasteps, desc="CMA steps"):
    imgs = G.visualize_batch_np(new_codes, B=42)
    latent_code = torch.from_numpy(np.array(new_codes)).float()
    scores = scorer.score_tsr(imgs)
    print("step %d dsim %.3f (%.3f) (norm %.2f)" % (
        i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
    new_codes = optim_cust.step_simple(scores, new_codes, )
    scores_all.extend(list(scores))
    generations.extend([i] * len(scores))

scores_all = np.array(scores_all)
generations = np.array(generations)
mtg = ToPILImage()(make_grid(imgs, nrow=7))
mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all,
         codes_fin=latent_code.cpu().numpy())
visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
    join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))


#%%
def BigGAN_evol_exp(scorer, optimizer, G, steps=100, RND=None, label="", init_code=None, batchsize=20):
    init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
    # optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2)
    new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    generations = []
    for i in tqdm.trange(steps, desc="CMA steps"):
        imgs = G.visualize_batch_np(new_codes, B=batchsize)
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        scores = scorer.score_tsr(imgs)
        print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
            i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
            latent_code[:, :128].norm(dim=1).mean()))
        new_codes = optimizer.step_simple(scores, new_codes, )
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))

    scores_all = np.array(scores_all)
    generations = np.array(generations)
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
    np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all,
             codes_fin=latent_code.cpu().numpy())
    visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
        join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))