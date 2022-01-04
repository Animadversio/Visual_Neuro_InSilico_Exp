"""
Library of functions useful for Recording population response and do Cosine Evolution.
"""
import time
from os.path import join
import matplotlib.pylab as plt
import torch
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from insilico_Exp_torch import TorchScorer, visualize_trajectory, resize_and_pad_tsr
from layer_hook_utils import get_module_names, get_layer_names

# def select_popul_record(model, layer, size=50, chan="rand", x=None, y=None):
#     return popul_idxs

def run_evol(scorer, objfunc, optimizer, G, reckey=None, steps=100, label="obj-target-G", savedir="",
            RFresize=True, corner=(0, 0), imgsize=(224, 224), init_code=None):
    if init_code is None:
        init_code = np.zeros((1, G.codelen))
    RND = np.random.randint(1E5)
    new_codes = init_code
    # new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    actmat_all = []
    generations = []
    codes_all = []
    best_imgs = []
    for i in range(steps,):
        codes_all.append(new_codes.copy())
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
        best_imgs.append(imgs[scores.argmax(),:,:,:].detach().clone())
        # debug @ jan.3rd. Before there is serious memory leak `.detach().clone()` solve the reference issue.
        actmat_all.append(actmat)
    codes_all = np.concatenate(tuple(codes_all), axis=0)
    scores_all = np.array(scores_all)
    actmat_all = np.concatenate(tuple(actmat_all), axis=0)
    generations = np.array(generations)
    mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
    mtg_exp.save(join(savedir, "besteachgen_%s_%05d.jpg" % (label, RND,)))
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    mtg.save(join(savedir, "lastgen_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
    if codes_all.shape[1] == 4096: # then subsample the codes
        np.savez(join(savedir, "scores_%s_%05d.npz" % (label, RND)), generations=generations, scores_all=scores_all, actmat_all=actmat_all, codes_fin=codes_all[-80:,:])
    else:
        np.savez(join(savedir, "scores_%s_%05d.npz" % (label, RND)), generations=generations, scores_all=scores_all, actmat_all=actmat_all, codes_all=codes_all)
    visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=label).savefig(
        join(savedir, "traj_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
    return codes_all, scores_all, actmat_all, generations, RND

#%%
def sample_center_units_idx(tsrshape, samplenum=500, single_col=True, resample=False):
    """

    :param tsrshape: shape of the tensor to be sampled
    :param samplenum: total number of unit to sample
    :param single_col: restrict the sampling to be from a single column
    :param resample: allow the same unit to be sample multiple times or not?
    :return:
        flat_idx_samp: a integer array to sample the flattened feature tensor
    """
    msk = np.zeros(tsrshape, dtype=np.bool) # the viable units in the center of the featuer map
    if len(tsrshape)==3:
        C, H, W = msk.shape
        if single_col: # a single column
            msk[:, int(H//2), int(W//2)] = True
        else: # a area in the center
            msk[:,
                int(H/4):int(3*H/4),
                int(W/4):int(3*W/4)] = True
    else:
        msk[:] = True
    center_idxs = np.where(msk.flatten())[0]
    flat_idx_samp = np.random.choice(center_idxs, samplenum, replace=resample)
    flat_idx_samp.sort()
    #     np.unravel_index(flat_idx_samp, outshape)
    return flat_idx_samp


def sample_center_column_units_idx(tsrshape, single_col=True):
    """ Return index of center column or the center columns.

    :param tsrshape: shape of the tensor to be sampled
    :param single_col: restrict the sampling to be from a single column
    :return:
        flat_idx_samp: a integer array to sample the flattened feature tensor
    """
    msk = np.zeros(tsrshape, dtype=np.bool) # the viable units in the center of the featuer map
    if len(tsrshape) == 3:
        C, H, W = msk.shape
        if single_col: # a single column
            msk[:, int(H//2), int(W//2)] = True
        else: # a area in the center
            msk[:,
                int(H/4):int(3*H/4),
                int(W/4):int(3*W/4)] = True
    else:
        msk[:] = True
    center_idxs = np.where(msk.flatten())[0]
    center_idxs.sort()
    return center_idxs


def set_random_population_recording(scorer, targetnames, randomize=True, popsize=500, single_col=True, resample=False,
                                    seed=None):
    """ Main effect is to set the recordings for the scorer object.
    (additional method for scorer)

    :param scorer:
    :param targetnames:
    :param popsize:
    :param single_col: restrict the sampling to be from a single column
    :param resample: allow the same unit to be sample multiple times or not?
    :return:

    """
    np.random.seed(seed) # set a seed for reproducing population selection
    unit_mask_dict = {}
    unit_tsridx_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    try:
        for layer in targetnames:
            inshape = module_spec[invmap[layer]]["inshape"]
            outshape = module_spec[invmap[layer]]["outshape"]
            if randomize:
                flat_idx_samp = sample_center_units_idx(outshape, popsize, single_col=single_col, resample=resample)
            else:
                flat_idx_samp = sample_center_column_units_idx(outshape, single_col=True)
                popsize = len(flat_idx_samp)

            tsr_idx_samp = np.unravel_index(flat_idx_samp, outshape)
            unit_mask_dict[layer] = flat_idx_samp
            unit_tsridx_dict[layer] = tsr_idx_samp
            scorer.set_popul_recording(layer, flat_idx_samp, )
            print(f"Layer {layer} Sampled {popsize} units from feature tensor of shape {outshape}")
    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return unit_mask_dict, unit_tsridx_dict
#%%

# 
def set_objective(score_method, targmat, popul_mask, popul_m, popul_s, grad=False, normalize=True):
    def objfunc(actmat):
        actmat_msk = actmat[:, popul_mask]
        targmat_msk = targmat[:, popul_mask] # [1 by masksize]
        if normalize:
            actmat_msk = (actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
            targmat_msk = (targmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
        
        if score_method == "L1":
            scores = - np.abs(actmat_msk - targmat_msk).mean(axis=1)
        elif score_method == "MSE":
            scores = - np.square(actmat_msk - targmat_msk).mean(axis=1)
        elif score_method == "corr":
            actmat_msk = actmat_msk - actmat_msk.mean()
            targmat_msk = targmat_msk - targmat_msk.mean()
            popact_norm = np.linalg.norm(actmat_msk, axis=1, keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1, keepdims=True)
            scores = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)
        elif score_method == "cosine":
            popact_norm = np.linalg.norm(actmat_msk, axis=1,keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1,keepdims=True)
            scores = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)
        elif score_method == "dot":
            scores = (actmat_msk @ targmat_msk.T).squeeze(axis=1)
        else:
            raise ValueError
        return scores # (Nimg, ) 1d array
    # return an array / tensor of scores for an array of activations
    # Noise form
    return objfunc


def encode_image(scorer, imgtsr, key=None,
                 RFresize=True, corner=None, imgsize=None):
    """return a 2d array / tensor of activations for a image tensor
    imgtsr: (Nimgs, C, H, W)
    actmat: (Npop, Nimages) torch tensor

    :return
        if key is None then return a dict of all actmat of all layer
        if key is in the dict, then return a single actmat of shape (imageN, unitN)
    """
    #TODO: make this work for larger image dataset
    if RFresize: imgtsr = resize_and_pad_tsr(imgtsr, imgsize, corner, )
    _, recordings = scorer.score_tsr(imgtsr)
    if key is None:
        return recordings
    else:
        return recordings[key]


def set_popul_mask(ref_actmat):
    img_var = ref_actmat.var(axis=0) # (unitN, )
    popul_mask = ~np.isclose(img_var, 0.0) 
    # if these inactive units are not excluded, then the normalization will be nan. 
    print(popul_mask.sum(), " units still active in the mask.")
    return popul_mask


def set_normalizer(ref_actmat):
    """ Get normalizer for activation. by default return mean and std.

    :param ref_actmat: torch tensor of shape (imageN, unitN)
    :return:
    """
    return ref_actmat.mean(axis=0, keepdims=True), ref_actmat.std(axis=0, keepdims=True),



from cycler import cycler
from matplotlib.cm import jet
def visualize_popul_act_evol(actmat_all, generations, targ_actmat):
    """
    # figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
    # figh.savefig(join(expdir, "popul_act_evol_%s_%d.png" % (explabel, RND)))

    :param actmat_all:
    :param generations:
    :param targ_actmat:
    :return:
    """
    Ngen = generations.max() + 1
    actmat_tr_avg = np.array([actmat_all[generations==gi, :].mean(axis=0) for gi in range(Ngen)])
    sortidx = targ_actmat.argsort()[0]
    figh= plt.figure(figsize=[10, 8])
    ax = plt.gca()
    ax.set_prop_cycle(cycler(color=[jet(k) for k in np.linspace(0,1,Ngen)]))
    plt.plot(actmat_tr_avg[:,sortidx].T, alpha=0.3, lw=1.5)
    plt.plot(targ_actmat[:,sortidx].T, color='k', alpha=0.8,lw=2.5)
    plt.xlabel("populatiion unit (sorted by target pattern)")
    plt.ylabel("activation")
    plt.title("Neural Pattern Evolution")
    plt.tight_layout()
    # plt.show()
    return figh


import os
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize
def load_ref_imgs(imgdir, preprocess=Compose([Resize((224, 224)), ToTensor()]), Nlimit=200):
    imgs = []
    imgnms = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(imgdir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(preprocess(Image.open(os.path.join(imgdir, f)).convert('RGB')))
        imgnms.append(f)
        if Nlimit is not None:
            if len(imgs) > Nlimit: 
                break
    imgtsr = torch.stack(imgs)
    return imgnms, imgtsr

if __name__=="__main__":
    #%%
    from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
    from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
    refimgdir = r"E:\Network_Data_Sync\Stimuli\2019-Selectivity\2019-Selectivity-Big-Set-01"
    exproot = r"E:\Cluster_Backup\Cosine_insilico"
    Optimizer = ["CholCMA", "HessCMA", "Adam"]
    Glist = ["FC6", "BigGAN"]
    score_methodlist = ["cos", "MSE", "L1", "dot", "corr"]
    GANname = "FC6"
    # Set population recording
    scorer = TorchScorer("resnet50")
    module_names, module_types, module_spec = get_module_names(scorer.model, (3, 227, 227), "cuda", False)
    unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [".layer3.Bottleneck0"], popsize=500)#
    # Encode a population of images to set the normalizer and mask. 
    refimgnms, refimgtsr = load_ref_imgs(imgdir=refimgdir, preprocess=Compose([Resize((227, 227)), ToTensor()]))
    ref_actmat = encode_image(scorer, refimgtsr, key=".layer3.Bottleneck0")
    popul_m, popul_s = set_normalizer(ref_actmat)
    popul_mask = set_popul_mask(ref_actmat)
    #%%
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    code_length = G.codelen
    expdir = os.path.join(exproot,"cosine")
    #%%
    for imgid in range(len(refimgnms)):
        # Select target image and add target vector. 
        targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
        targ_actmat = encode_image(scorer, target_imgtsr, key=".layer3.Bottleneck0")  # 1, unitN
        targlabel = os.path.splitext(targnm)[0]
        # organize data with the targetlabel
        expdir = os.path.join(exproot, "rec_%s"%targlabel)
        os.makedirs(expdir, exist_ok=True)
        for score_method in ["cosine", "corr", "MSE", "dot"]:
            explabel = "%s-%s-%s"%(targlabel, score_method, GANname)
            objfunc = set_objective(score_method, targ_actmat, popul_mask, popul_m, popul_s)
            optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                            init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                            maximize=True, random_seed=None, optim_params={})
            codes_all, scores_all, actmat_all, generations, RND = run_evol(scorer, objfunc, optimizer, G, reckey=".layer3.Bottleneck0", label=explabel, savedir=expdir,
                        steps=100, RFresize=True, corner=(20, 20), imgsize=(187, 187))
            ToPILImage()(target_imgtsr[0]).save(join(expdir, "targetimg_%s_%d.png"%(explabel, RND)))
            figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
            figh.savefig(join(expdir, "popul_act_evol_%s_%d.png" % (explabel, RND)))
    #%%

    # figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
    # figh.savefig(join(expdir, "popul_act_evol_%s_%d.png"%(explabel, RND)))
