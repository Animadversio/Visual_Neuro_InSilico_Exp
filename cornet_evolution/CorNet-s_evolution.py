"""
Evolution using specific time step from CorNet-S units.
Run from cluster win Command Line Inferface in large scale.
Binxu
Feb.6th, 2022
"""
import os, sys, argparse, time, glob, pickle, subprocess, shlex, io, pprint
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
import cornet
from PIL import Image
from easydict import EasyDict
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# sys.path.append("E:\Github_Projects\ActMax-Optimizer-Dev")                 #Binxu local
sys.path.append(r"D:\Github\ActMax-Optimizer-Dev")                           #Binxu office
# sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\ActMax-Optimizer-Dev")   #Victoria local
#sys.path.append(r"\data\Victoria\UCSD_projects\ActMax-Optimizer-Dev")       #Victoria remote
from core.GAN_utils import upconvGAN
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid
from core.layer_hook_utils import featureFetcher, get_module_names, get_layer_names
from core.montage_utils import ToPILImage, make_grid, show_tsrbatch, PIL_tsrbatch
from collections import defaultdict


def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  # remove DataParallel
    return model


class featureFetcher_recurrent:
    """ Light weighted modular feature fetcher, simpler than TorchScorer. """
    def __init__(self, model, input_size=(3, 224, 224), device="cuda", print_module=True):
        self.model = model.to(device)
        module_names, module_types, module_spec = get_module_names(model, input_size, device=device, show=print_module)
        self.module_names = module_names
        self.module_types = module_types
        self.module_spec = module_spec
        self.activations = defaultdict(list)
        self.hooks = {}
        self.device = device

    def record(self, module, submod, key="score", return_input=False, ingraph=False):
        """
        submod:
        """
        hook_fun = self.get_activation(key, ingraph=ingraph, return_input=return_input)
        if submod is not None:
            hook_h = getattr(getattr(self.model, module), submod).register_forward_hook(hook_fun)
        else:
            hook_h = getattr(self.model, module).register_forward_hook(hook_fun)
        #register_hook_by_module_names(target_name, hook_fun, self.model, device=self.device)
        self.hooks[key] = hook_h
        return hook_h

    def remove_hook(self):
        for name, hook in self.hooks.items():
            hook.remove()
        print("Deconmissioned all the hooks")
        return

    def __del__(self):
        for name, hook in self.hooks.items():
            hook.remove()
        print("Deconmissioned all the hooks")
        return

    def __getitem__(self, key):
        try:
            return self.activations[key]
        except KeyError:
            raise KeyError

    def get_activation(self, name, ingraph=False, return_input=False):
        """If returning input, it may return a list or tuple of things """
        if return_input:
            def hook(model, input, output):
                self.activations[name].append(input if ingraph else [inp.detach().cpu() for inp in input])
        else:
            def hook(model, input, output):
                # print("get activation hook")
                self.activations[name].append(output if ingraph else output.detach().cpu())

        return hook
#%%
"""
Actually, if you use higher version of pytorch, the torch transform could work...
Lower version you need to manually write the preprocessing function. 
"""
# imsize = 224
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])
# preprocess_fun = torchvision.transforms.Compose([
#                     torchvision.transforms.Resize((imsize, imsize)),
#                     normalize,
#                 ])
RGBmean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).cuda()
RGBstd  = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).cuda()
def preprocess_fun(imgtsr, imgsize=224, ):
    """Manually write some version of preprocessing"""
    imgtsr = nn.functional.interpolate(imgtsr, [imgsize,imgsize])
    return (imgtsr - RGBmean) / RGBstd
#%%
import matplotlib.pylab as plt
def visualize_trajectory(scores_all, generations, show=True):
    gen_slice = np.arange(min(generations), max(generations)+1)
    AvgScore = np.zeros_like(gen_slice).astype("float64")
    MaxScore = np.zeros_like(gen_slice).astype("float64")
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh = plt.figure(figsize=[6,5])
    plt.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    plt.plot(gen_slice, AvgScore, color='black', label="Average score")
    plt.plot(gen_slice, MaxScore, color='red', label="Max score")
    plt.xlabel("generation #")
    plt.ylabel("CNN unit score")
    plt.title("Optimization Trajectory of Score\n")# + title_str)
    plt.legend()
    if show:
        plt.show()
    return figh


def visualize_image_trajectory(G, codes_all, generations, show=True):
    meancodes = [np.mean(codes_all[generations == i, :], axis=0)
                 for i in range(int(generations.min()), int(generations.max())+1)]
    meancodes = np.array(meancodes)
    imgtsrs = G.visualize_batch_np(meancodes)
    mtg = PIL_tsrbatch(imgtsrs, nrow=10)
    if show:mtg.show()
    return mtg


def visualize_best(G, codes_all, scores_all, show=True):
    bestidx = np.argmax(scores_all)
    bestcodes = np.array([codes_all[bestidx,:]])
    bestimgtsrs = G.visualize_batch_np(bestcodes)
    mtg = PIL_tsrbatch(bestimgtsrs, nrow=1)
    if show:mtg.show()
    return mtg


def calc_meancodes(codes_all, generations):
    meancodes = [np.mean(codes_all[generations == i, :], axis=0)
                 for i in range(int(generations.min()), int(generations.max()) + 1)]
    meancodes = np.array(meancodes)
    return meancodes


#%% Prepare model
G = upconvGAN("fc6")
G.eval().cuda().requires_grad_(False)

model = get_model(pretrained=True)
model.eval().requires_grad_(False)
#%% Evolution parameters and Optimzer
def run_evolution(model, area, sublayer, time_step, channum, pos="autocenter"):
    if pos is "autocenter":
        findcenter = True
    else:
        findcenter = False
        assert len(pos) == 2 and type(pos) in [list, tuple]
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    if findcenter:
        with torch.no_grad():
            model(preprocess_fun(G.visualize(torch.zeros(1,4096).float().cuda())))
        tsr = fetcher["target"][time_step]
        _, C, H, W = tsr.shape
        pos = (H // 2, W // 2)

    print("Evolve from {}_{}_Time {}_ Channel {} Position {}".format(area, sublayer, str(time_step), str(channum), str(pos)))
    optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
    # optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
    #                                   lr=1.5, sphere_norm=300)
    # optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33), )
    codes_col = []
    gen_col = []
    scores_col = []
    codes = optim.get_init_pop()
    for i in range(100):
        # get score
        fetcher.activations["target"] = []
        with torch.no_grad():
            ppx = preprocess_fun(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")))
            model(ppx)
        scores = np.array(fetcher["target"][time_step][:, channum, pos[0], pos[1]])
        # optimizer update
        newcodes = optim.step_simple(scores, codes)
        gen_col.append(i * np.ones_like(scores, dtype=np.int))
        scores_col.append(scores)
        codes_col.append(codes)
        # print(f"Gen {i:d} {scores.mean():.3f}+-{scores.std():.3f}")
        codes = newcodes
        del newcodes
    generations = np.concatenate(tuple(gen_col), axis=0)
    scores_all = np.concatenate(tuple(scores_col), axis=0)
    codes_all = np.concatenate(tuple(codes_col), axis=0)
    fetcher.remove_hook()
    del fetcher
    return codes, scores, \
           EasyDict(generations=generations,scores_all=scores_all,codes_all=codes_all,)


dataroot = r"F:\insilico_exps\CorNet-recurrent-evol"
#%%
area = "IT"
sublayer = "output"  # None
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(5):
    for channum in range(50, 100):
        for time_step in [0, 1]:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = PIL_tsrbatch(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                     scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg

# ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
# Final activation 20.48+-1.08 time 64.520 sec
# Final activation 83.59+-5.11 time 64.792 sec
# Final activation 16.68+-0.97 time 55.456 sec
# Final activation 96.97+-6.94 time 57.806 sec
#%%
area = "V4"
sublayer = "output"  # None
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(5):
    for channum in range(25, 50):
        for time_step in [0, 1, 2, 3]:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = PIL_tsrbatch(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                     scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg

#%%
area = "V2"
sublayer = "output"  # None
outdir = join(dataroot, "%s-%s"%(area, sublayer))
os.makedirs(outdir, exist_ok=True)
for runnum in range(5):
    for channum in range(25, 50):
        for time_step in [0, 1]:
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runnum:02d}"
            meta = EasyDict(area=area, sublayer=sublayer, channum=channum, time_step=time_step,
                            runnum=runnum, explabel=explabel,)
            t0 = time.time()
            codes, scores, datadict = run_evolution(model, area, sublayer, time_step, channum, pos="autocenter")
            t1 = time.time()
            print(f"Final activation {scores.mean():.2f}+-{scores.std():.2f} time {t1 - t0:.3f} sec")
            meancodes = calc_meancodes(datadict.codes_all, datadict.generations)
            figh = visualize_trajectory(datadict.scores_all, datadict.generations, False)
            figh.savefig(join(outdir, "score_traj_%s.png" % (explabel)))
            plt.close(figh)
            # mtg = visualize_image_trajectory(G, datadict.codes_all, datadict.generations, False)
            mtg = PIL_tsrbatch(G.visualize_batch_np(meancodes), nrow=10)
            mtg.save(join(outdir, "evol_img_traj_%s.jpg" % (explabel)))
            bestmtg = visualize_best(G, datadict.codes_all, datadict.scores_all, False)
            bestmtg.save(join(outdir, "bestimg_%s.jpg" % (explabel)))
            np.savez(join(outdir, "exp_data_%s.png" % (explabel)),
                   meancodes=meancodes, generations=datadict.generations,
                     scores_all=datadict.scores_all, **meta,)
            t2 = time.time()
            print(f"Finish saving time {t2 - t0:.3f} sec")
            # del mtg, bestmtg
#%%
# #%%
# time_steps = [0, 1]
# area = "IT"
# sublayer = "conv3"
#
# import random
# C = 512 # np.shape(fetcher["target"][time_step])[1]
# channums = random.sample(range(C), 200)
# for channum in channums:
#     for time_step in time_steps:
#         for i in range(3):
#             fetcher, codes, scores = run_evolution(model, area, sublayer, time_step, channum)
#             pil_image = ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
#             # filename = "N:\\Users\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(
#             #     area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(), ".2f"))
#
#             filename = "D:\\Ponce-Lab\\Victoria\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(),".2f"))
#             pil_image.save(filename)
#             del codes, pil_image



