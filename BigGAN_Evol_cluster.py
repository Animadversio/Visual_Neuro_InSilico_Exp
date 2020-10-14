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
from ZO_HessAware_Optimizers import CholeskyCMAES, HessCMAES
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN
#%%
def visualize_trajectory(scores_all, generations, codes_arr=None, show=False, title_str=""):
    """ Visualize the Score Trajectory """
    gen_slice = np.arange(min(generations), max(generations) + 1)
    AvgScore = np.zeros_like(gen_slice)
    MaxScore = np.zeros_like(gen_slice)
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh, ax1 = plt.subplots()
    ax1.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    ax1.plot(gen_slice, AvgScore, color='black', label="Average score")
    ax1.plot(gen_slice, MaxScore, color='red', label="Max score")
    ax1.set_xlabel("generation #")
    ax1.set_ylabel("CNN unit score")
    plt.legend()
    if codes_arr is not None:
        ax2 = ax1.twinx()
        if codes_arr.shape[1] == 256: # BigGAN
            nos_norm = np.linalg.norm(codes_arr[:, :128], axis=1)
            cls_norm = np.linalg.norm(codes_arr[:, 128:], axis=1)
            ax2.scatter(generations, nos_norm, s=5, color="orange", label="noise", alpha=0.2)
            ax2.scatter(generations, cls_norm, s=5, color="magenta", label="class", alpha=0.2)
        elif codes_arr.shape[1] == 4096: # FC6GAN
            norms_all = np.linalg.norm(codes_arr[:, :], axis=1)
            ax2.scatter(generations, norms_all, s=5, color="magenta", label="all", alpha=0.2)
        ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
        plt.legend()
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    if show:
        plt.show()
    return figh
#%%
if sys.platform == "linux":
    rootdir = r"/scratch/binxu/BigGAN_Optim_Tune_new" # _new
    Hdir_BigGAN = r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
else:
    # rootdir = r"E:\OneDrive - Washington University in St. Louis\BigGAN_Optim_Tune_tmp"
    rootdir = r"E:\Monkey_Data\BigGAN_Optim_Tune_tmp"
    Hdir_BigGAN = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN\H_avg_1000cls.npz"
    Hdir_fc6 = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default="fc6", help="Network model to use for Image distance computation")
parser.add_argument("--chans", type=int, nargs='+', default=[0, 25], help="")
parser.add_argument("--G", type=str, default="BigGAN", help="")
parser.add_argument("--optim", type=str, nargs='+', default=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
parser.add_argument("--steps", type=int, default=100, help="")
parser.add_argument("--reps", type=int, default=5, help="")
parser.add_argument("--RFresize", type=bool, default=False, help="")
args = parser.parse_args() # ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']
if args.G == "BigGAN":
    Hdata = np.load(Hdir_BigGAN)
elif args.G == "fc6":
    Hdata = np.load(Hdir_fc6)
#%%
"""with a correct cmaes or initialization, BigGAN can match FC6 activation."""
#%% Select GAN
from GAN_utils import BigGAN_wrapper, upconvGAN
from insilico_Exp import TorchScorer, ExperimentEvolve
if args.G == "BigGAN":
    if sys.platform == "linux":
        BGAN = get_BigGAN(version="biggan-deep-256")
    else:
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
if args.RFresize:
    from torch_net_utils import receptive_field, receptive_field_for_unit, layername_dict
    rf_dict = receptive_field(scorer.model.features, (3, 227, 227), device="cuda")
    layername = layername_dict[args.net]
    layer_name_map = {layer: str(i+1) for i, layer in enumerate(layername)}
# scorer.select_unit(("alexnet", "fc6", 2))
# imgs = G.visualize(torch.randn(3, 256).cuda()).cpu()
# scores = scorer.score_tsr(imgs)
#%% Wrap the optimizers, like concatenate 2 or concatenate one with a fixed code
class concat_wrapper:
    def __init__(self, optim1, optim2):
        self.optim1 = optim1
        self.optim2 = optim2
        self.sep = self.optim1.space_dimen

    def step_simple(self, scores, codes):
        new_codes1 = self.optim1.step_simple(scores, codes[:,:self.sep])
        new_codes2 = self.optim2.step_simple(scores, codes[:,self.sep:])
        return np.concatenate((new_codes1, new_codes2), axis=1)

class fix_param_wrapper:
    def __init__(self, optim, fixed_code, pre=True):
        """pre True means fix the initial part of code, False means last part"""
        self.optim = optim
        self.fix_code = fixed_code
        self.pre = pre  # if the fix code in in the first part
        self.sep = fixed_code.shape[1]

    def step_simple(self, scores, codes):
        if self.pre:
            new_codes1 = self.optim.step_simple(scores, codes[:, self.sep:])
            return np.concatenate((np.repeat(self.fix_code, new_codes1.shape[0], axis=0), new_codes1), axis=1)
        else:
            new_codes1 = self.optim.step_simple(scores, codes[:, :-self.sep])
            return np.concatenate((new_codes1, np.repeat(self.fix_code, new_codes1.shape[0], axis=0)), axis=1)

#%% Optimizer from label, Use this to translate string labels to optimizer
def label2optimizer(methodlabel, init_code, GAN="BigGAN", ): # TODO add default init_code
    if GAN == "BigGAN":
        if methodlabel == "CholCMA":
            optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2,)
        elif methodlabel == "CholCMA_class":
            optim = CholeskyCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.06,)
            optim_cust = fix_param_wrapper(optim, init_code[:, :128], pre=True)
        elif methodlabel == "CholCMA_noise":
            optim = CholeskyCMAES(space_dimen=128, init_code=init_code[:, :128], init_sigma=0.3,)
            optim_cust = fix_param_wrapper(optim, init_code[:, 128:], pre=False)
        elif methodlabel == "CholCMA_prod":
            optim1 = CholeskyCMAES(space_dimen=128, init_code=init_code[:, :128], init_sigma=0.1,)
            optim2 = CholeskyCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.06,)
            optim_cust = concat_wrapper(optim1, optim2)
        elif methodlabel == "CholCMA_noA":
            optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)
        elif methodlabel == "HessCMA":
            eva = Hdata['eigvals_avg'][::-1]
            evc = Hdata['eigvects_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
        elif methodlabel == "HessCMA_noA":
            eva = Hdata['eigvals_avg'][::-1]
            evc = Hdata['eigvects_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2, Aupdate_freq=102)
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
        elif methodlabel == "HessCMA_class":
            eva = Hdata['eigvals_clas_avg'][::-1]
            evc = Hdata['eigvects_clas_avg'][:, ::-1]
            optim_hess = HessCMAES(space_dimen=128, init_code=init_code[:, 128:], init_sigma=0.2, )
            optim_hess.set_Hessian(eigvals=eva, eigvects=evc, expon=1 / 2.5)
            optim_cust = fix_param_wrapper(optim_hess, init_code[:, :128], pre=True)
    elif GAN == "fc6":
        if methodlabel == "CholCMA":
            optim_cust = CholeskyCMAES(space_dimen=4096, init_code=init_code, init_sigma=3,)
        elif methodlabel == "CholCMA_noA":
            optim_cust = CholeskyCMAES(space_dimen=4096, init_code=init_code, init_sigma=3, Aupdate_freq=102)
        elif methodlabel == "HessCMA800":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=800, init_code=init_code, init_sigma=0.8, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=800, expon=1 / 5)
        elif methodlabel == "HessCMA500":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=500, init_code=init_code, init_sigma=0.8, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=500, expon=1 / 5)
        elif methodlabel == "HessCMA500_1":
            eva = Hdata['eigv_avg'][::-1]
            evc = Hdata['eigvect_avg'][:, ::-1]
            optim_cust = HessCMAES(space_dimen=4096, cutoff=500, init_code=init_code, init_sigma=0.4, )
            optim_cust.set_Hessian(eigvals=eva, eigvects=evc, cutoff=500, expon=1 / 4)
    return optim_cust
#%%
method_col = args.optim

#%%
def resize_and_pad(imgs, corner, size):
    pad_img = torch.ones_like(imgs) * 0.5
    rsz_img = F.interpolate(imgs, size=size, align_corners=True, mode="bilinear")
    pad_img[:,:,corner[0]:corner[0]+size[0],corner[1]:corner[1]+size[1]] = rsz_img
    return pad_img

# optimizer_col = [label2optimizer(methodlabel, np.random.randn(1, 256), GAN=args.G) for methodlabel in method_col]
#%%
pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}

for unit_id in range(args.chans[0], args.chans[1]):
    if "fc" in args.layer:
        unit = (args.net, args.layer, unit_id)
    else:
        unit = (args.net, args.layer, unit_id, *pos_dict[args.layer])
    scorer.select_unit(unit)
    if args.RFresize:
        if "fc" in args.layer:
            imgsize = (256, 256)
            corner = (0, 0)
        else:
            rf_pos = receptive_field_for_unit(rf_dict, (3, 227, 227), layer_name_map[args.layer], pos_dict[args.layer])
            imgsize = (int((rf_pos[0][1] - rf_pos[0][0]) / 227 * 256 + 1), int((rf_pos[1][1] - rf_pos[1][0]) / 227 * 256 + 1))
            corner = (int(rf_pos[0][0] / 227 * 256 - 1), int(rf_pos[1][0] / 227 * 256 - 1))
    # Save directory named after the unit. Add RFrsz as suffix if resized
    savedir = join(rootdir, r"%s_%s_%d" % unit[:3]) 
    if args.RFresize: savedir+="_RFrsz"
    os.makedirs(savedir, exist_ok=True)
    for triali in range(args.reps):
        if args.G == "BigGAN":
            fixnoise = 0.7 * truncated_noise_sample(1, 128)
            init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        elif args.G == "fc6":
            init_code = np.random.randn(1, 4096)
        RND = np.random.randint(1E5)
        np.save(join(savedir, "init_code_%05d.npy"%RND), init_code)
        optimizer_col = [label2optimizer(methodlabel, init_code, args.G) for methodlabel in method_col]
        for methodlab, optimizer in zip(method_col, optimizer_col):
            if args.G == "fc6":  methodlab += "_fc6"  # add space notation as suffix to optimizer
            # One single evolution! 
            new_codes = init_code
            # new_codes = init_code + np.random.randn(25, 256) * 0.06
            scores_all = []
            generations = []
            codes_all = []
            best_imgs = []
            for i in range(args.steps,):
                codes_all.append(new_codes.copy())
                imgs = G.visualize_batch_np(new_codes) # B=1
                latent_code = torch.from_numpy(np.array(new_codes)).float()
                if args.RFresize: imgs = resize_and_pad(imgs, corner, imgsize)
                scores = scorer.score_tsr(imgs)
                if args.G == "BigGAN":
                    print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                        i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                        latent_code[:, :128].norm(dim=1).mean()))
                else:
                    print("step %d score %.3f (%.3f) (norm %.2f )" % (
                        i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
                new_codes = optimizer.step_simple(scores, new_codes, )
                scores_all.extend(list(scores))
                generations.extend([i] * len(scores))
                best_imgs.append(imgs[scores.argmax(),:,:,:])

            codes_all = np.concatenate(tuple(codes_all), axis=0)
            scores_all = np.array(scores_all)
            generations = np.array(generations)
            mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
            mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
            mtg = ToPILImage()(make_grid(imgs, nrow=7))
            mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
            if args.G == "fc6":
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:,:])
            else:
                np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all, codes_all=codes_all)
            visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
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