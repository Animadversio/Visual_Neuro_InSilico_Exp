import cv2
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import os
from os.path import join
from tqdm import tqdm
from easydict import EasyDict
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from insilico_Exp_torch import TorchScorer
from layer_hook_utils import get_module_names
#%% video input
def open_video(vid_path=r'E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4'):
    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_Fnum = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video opened: %s "%vid_path)
    print("FPS %.1f Total frame number %d"%(fps, video_Fnum))
    return vidcap


def read_framebatch(vidcap, batch=40):
    imgs = []
    for cnt in range(batch):
        success, image = vidcap.read() 
        imgs.append(image)
        if not success:
            print("Video ends!")
            return []
    image_stack = np.stack(imgs, axis=0)
    return image_stack

#%% Helper function to find the index for center unit.
def sample_center_units_idx(tsrshape, samplenum=500):
    msk = np.zeros(tsrshape, dtype=np.bool)
    if len(tsrshape)==3:
        C, H, W = msk.shape
        msk[:,
            int(H/4):int(3*H/4),
            int(W/4):int(3*W/4)] = True
    else:
        msk[:] = True
    center_idxs = np.where(msk.flatten())[0]
    flat_idx_samp = np.random.choice(center_idxs, samplenum,)
    flat_idx_samp.sort()
    #     np.unravel_index(flat_idx_samp, outshape)
    return flat_idx_samp

def set_random_population_recording(scorer, targetnames, popsize=500):
    unit_mask_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    for layer in targetnames:
        inshape = module_spec[invmap[layer]]["inshape"]
        outshape = module_spec[invmap[layer]]["outshape"]
        flat_idx_samp = sample_center_units_idx(outshape, popsize)
        unit_mask_dict[layer] = flat_idx_samp
        scorer.set_popul_recording(layer, flat_idx_samp)
    return unit_mask_dict

#%% extend record to existing ones. 
def extend_record(recordings_all, recordings):
    if type(recordings_all) is dict:
        for k, v in recordings.items():
            if not k in recordings_all:
                recordings_all[k] = v
            else:
                recordings_all[k] = np.concatenate([recordings_all[k], v], axis=0)
    elif type(recordings_all) is np.ndarray:
        recordings_all = np.concatenate([recordings_all, recordings], axis=0)
    return recordings_all

#%% Estimate Auto-correlation
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

from torch.fft import irfft, rfft
_NEXT_FAST_LEN = {}
def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1

def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Implementation copied form `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


def calc_acf_all(recordings_all):
    acf_arr_dict = {}
    for layer in recordings_all:
        # numpy backend
        # acf_arr = []
        # popsize = recordings_all[layer].shape[1]
        # for ni in range(popsize):
        #     acf = estimated_autocorrelation(recordings_all[layer][:,ni])
        #     acf_arr.append(acf)
        # acf_arr = np.array(acf_arr)
        # torch backend
        acfs_tsr = autocorrelation(torch.tensor(recordings_all[layer]), dim=0)  # time lag by popsize
        acf_arr = acfs_tsr.numpy().T  # popsize by time lag
        acf_arr_dict[layer] = acf_arr
    return acf_arr_dict


def exp_offset(t, tau, A, B):
    return A * (np.exp(-t / tau) + B)


def fit_acf_expoffset(targetnames, acf_arr_dict, fps=30, max_ticks=180):
    popt_dict = {}
    for layer in targetnames:
        acf_arr = acf_arr_dict[layer]
        tticks = np.arange(acf_arr.shape[1]) / fps * 1000
        popt_dict[layer] = []
        for ni in tqdm(range(acf_arr.shape[0])):
            try:
                popt, pcov = curve_fit(exp_offset, tticks[:max_ticks], acf_arr[ni,:max_ticks], p0=[10, 1, 0])
            except:
                print("fitting failed")
                continue
            popt_dict[layer].append(popt)
        popt_dict[layer] = np.stack(popt_dict[layer])

    print("Formatting fit parameter into table...")
    df_col = []
    for layer, popt_arr in popt_dict.items():
        for i in range(popt_arr.shape[0]):
            df_col.append((layer, popt_arr[i,0], popt_arr[i,1], popt_arr[i,2]))

    df = pd.DataFrame(df_col, columns=["layer", "tau", "A", "B"])
    targinvmap = {layer: i for i,layer in enumerate(targetnames)}
    layerid = df.layer.apply(targinvmap.__getitem__)
    df["layer_id"] = layerid
    return df, popt_dict


def shadedErrorbar(x, ymean, ysem, lw=None, c=None, label=None):
    if x is None:
        x = np.arange(len(ymean))
    plt.plot(x, ymean, c=c, lw=lw, label=label)
    plt.fill_between(x, ymean-ysem, ymean+ysem, color=c, alpha=0.3)

def average_acf_compare(acf_arr_dict, fps=30, xlim=10000, \
            video_id="", run_frames=-1):
    figh = plt.figure(figsize=[7,6])
    for layer in acf_arr_dict:
        acf_arr = acf_arr_dict[layer]
        popsize = acf_arr.shape[0]
        tticks = np.arange(acf_arr.shape[1]) / fps * 1000
        tick_lim = (tticks < xlim).sum()
        acf_mean = np.nanmean(acf_arr, axis=0)
        acf_sem = np.nanstd(acf_arr, axis=0) / np.sqrt(popsize)
        shadedErrorbar(tticks[:tick_lim], acf_mean[:tick_lim], acf_sem[:tick_lim], label=layer)
    plt.xlabel("time (ms)")
    plt.xlim(0, xlim)
    plt.ylabel("auto correlation")
    plt.legend()
    plt.title("Comparison of ACF curves across layers\nVideo %s Frames # %d" \
              % (video_id, run_frames))
    return figh


def visualize_fit_df(df, video_id="", run_frames=-1, varnm="tau"):
    cval, pval = spearmanr(df.layer_id, df.tau)
    figh = plt.figure(figsize=[8, 6])
    sns.violinplot(x="layer", y=varnm, data=df)
    plt.title("Trend of ACF timescale and depth of layer\ncorr coef %.3f (P=%.1e,N=%d)\nVideo %s Frames # %d" \
              % (cval, pval, df.shape[0], video_id, run_frames))
    plt.xticks(rotation=40)
    return figh


def timescale_analysis_pipeline(scorer, targetnames, popsize=500, run_frames=20000, seglen=1000, batch=80,
    video_path=r'E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4', video_id="goprorun",
    savedir="", savenm="resnet"):
    os.makedirs(savedir, exist_ok=True)
    # scorer = TorchScorer(netname)
    print("Setting up the recording units in neural network... ")
    unit_mask_dict = set_random_population_recording(scorer, targetnames, popsize=popsize)

    vidcap = open_video(vid_path=video_path) 
    print("Start streaming and recording...")
    recordings_all = {}
    for i in tqdm(range(int(run_frames/seglen))):
        image_stack = read_framebatch(vidcap, seglen)
        if len(image_stack) == 0:
            break
        _, recordings = scorer.score_tsr(image_stack[:, :, 280:-280, :], input_scale=255.0, B=batch)
        # scores_all = extend_record(scores_all, scores)
        recordings_all = extend_record(recordings_all, recordings)
    pkl.dump(EasyDict({"recordings_all": recordings_all, "unit_mask_dict":unit_mask_dict, "video_path":video_path}),
        open(join(savedir, "pop_recordings_%s.pkl"%savenm), "wb"))

    print("Start visualizing the recording traces...")
    for layer in targetnames:
        plt.figure(figsize=[12, 3])
        plt.plot(recordings_all[layer], alpha=0.1, lw=0.5)
        plt.xlim([0, 10000])
        plt.title("Population Response in %s\n Video %s" \
                  % (layer, video_id))
        plt.title(layer)
        plt.savefig(join(savedir, "pop_resp_demo_%s-%s.png"%(savenm, layer)))
    # plt.savefig(join(savedir, "pop_resp_demo_%s-%s.pdf"%(savenm, layer)))

    print("Start auto correlation calculation (ACF)...")
    acf_arr_dict = calc_acf_all(recordings_all)
    pkl.dump(EasyDict({"acf_arr_dict": acf_arr_dict, "unit_mask_dict": unit_mask_dict, "video_path":video_path}),
             open(join(savedir, "pop_acf_%s.pkl" % savenm), "wb"))

    print("Visualizing auto correlation calculation (ACF)...")
    figh = average_acf_compare(acf_arr_dict, fps=30, xlim=10000,
                        video_id=video_id, run_frames=run_frames)
    figh.savefig(join(savedir, "resp_acf_curv_cmp_%s.png" % savenm))
    figh.savefig(join(savedir, "resp_acf_curv_cmp_%s.pdf" % savenm))
    plt.show()

    print("Fitting Auto correlation function with Exp decay...")
    df, popt_dict = fit_acf_expoffset(targetnames, acf_arr_dict, fps=30, max_ticks=180)
    pkl.dump(EasyDict({"popt_dict": popt_dict, "unit_mask_dict": unit_mask_dict, "video_path":video_path}),
             open(join(savedir, "resp_acf_fit_%s.pkl" % savenm), "wb"))
    df.to_csv(join(savedir, "resp_acf_fit_params_%s.csv" % savenm))

    print("Progression of acf time scale along visual hierarchy...")
    figh2 = visualize_fit_df(df, video_id, run_frames)
    figh2.savefig(join(savedir, "resp_acf_fit_tau_violin_%s.png" % savenm))
    figh2.savefig(join(savedir, "resp_acf_fit_tau_violin_%s.pdf" % savenm))
    plt.show()
    return recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2

if __name__=='__main__':
    scorer = TorchScorer("resnet50")
    targetnames = [".Relurelu",
                   ".layer1.Bottleneck0",
                   ".layer1.Bottleneck2",
                   ".layer2.Bottleneck0",
                   ".layer2.Bottleneck2",
                   ".layer3.Bottleneck0",
                   ".layer3.Bottleneck2",
                   ".layer3.Bottleneck4",
                   ".layer4.Bottleneck0",
                   ".layer4.Bottleneck2",
                   ".Linearfc"
                   ]
    recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=500, run_frames=20000, seglen=1000, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_bike", savenm="resnet_bike")

    # r'E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4',
    # r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4"
    #%%
    # scorer.select_unit(("resnet50", ".layer3.Bottleneck0", 6, 7, 7)) # dummy unit
    # netname = "resnet50"
    # targetnames = [".MaxPool2dmaxpool",
    #            ".layer1.Bottleneck0",
    #            ".layer1.Bottleneck2",
    #            ".layer2.Bottleneck0",
    #            ".layer2.Bottleneck2",
    #            ".layer3.Bottleneck0",
    #            ".layer3.Bottleneck2",
    #            ".layer3.Bottleneck4",
    #            ".layer4.Bottleneck0",
    #            ".layer4.Bottleneck2",
    #            ".Linearfc"
    #           ]
    #
    # scorer = TorchScorer(netname)
    # print("Setting up the recording units in neural network... ")
    # unit_mask_dict = set_random_population_recording(scorer, targetnames, popsize=500)
    #
    # vidcap = open_video(vid_path=r'E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4')
    # print("Start streaming and recording...")
    # scores_all, recordings_all = [], {}
    # for i in tqdm(range(20)):
    #     image_stack = read_framebatch(vidcap,1000)
    #     if len(image_stack) == 0:
    #         break
    #     _, recordings = scorer.score_tsr(image_stack[:,:,280:-280,:], input_scale=255.0)
    #     # scores_all = extend_record(scores_all, scores)
    #     recordings_all = extend_record(recordings_all, recordings)
    #
    # print("Start visualizing the recording traces...")
    # for layer in recordings:
    #     plt.figure(figsize=[12,3])
    #     plt.plot(recordings_all[layer], alpha=0.3, lw=0.5)
    #     plt.xlim([0,10000])
    #     plt.title(layer)
    #
    # print("Start auto correlation calculation (ACF)...")
    # acf_arr_dict = calc_acf_all(recordings_all)
    #
    # print("Fitting Auto correlation function with Exp decay...")
    # df, popt_dict = fit_acf_expoffset(targetnames, acf_arr_dict, fps=30, max_ticks=180)
    #
    # print("Progression of acf time scale along visual hierarchy...")
    # cval, pval = spearmanr(df.layerid, df.tau)
    # sns.violinplot(x="layer", y="tau", data=df)
    # plt.title("Trend of ACF timescale and depth of layer\ncorr coef %.3f (P=%.1e,N=%d)"%(cval,pval,df.shape[0]))
    #
