from tqdm import tqdm
from os.path import join
from core.montage_utils import ToPILImage, make_grid, make_grid_np, show_tsrbatch, PIL_tsrbatch
import matplotlib.pyplot as plt
import seaborn as sns

dataroot = r"F:\insilico_exps\CorNet-recurrent-evol"
outdir = r"F:\insilico_exps\CorNet-recurrent-evol\proto_summary"
runnum = 5
sublayer = "output"  # None


# %%
def sweep_proto_merge(area, sublayer, chanrng, timestepN, runnum=5, outdir=outdir):
    """ sweep through folders and collect the prototype images and montage them into one """
    datadir = join(dataroot, "%s-%s" % (area, sublayer))
    for channum in tqdm(range(chanrng[0], chanrng[1])):
        outlabel = f"{area}-{sublayer}-Ch{channum:03d}_allproto_mtg"
        imgcol = []
        for time_step in range(timestepN):
            for runi in range(runnum):
                explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runi:02d}"
                img = plt.imread(join(datadir, "bestimg_%s.jpg" % (explabel)))
                imgcol.append(img)
        imgmtg = make_grid_np(imgcol, nrow=runnum, padding=2, pad_value=0)
        plt.imsave(join(outdir, outlabel + ".jpg"), imgmtg, )


sweep_proto_merge("V2", "output", (0, 50), 2, runnum=5, outdir=outdir)
sweep_proto_merge("V4", "output", (0, 50), 4, runnum=5, outdir=outdir)
sweep_proto_merge("IT", "output", (0, 100), 2, runnum=5, outdir=outdir)

# %%
# %%
import cornet
import numpy as np
import torch
import torch.nn.functional as F
from core.layer_hook_utils import featureFetcher_recurrent


def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  # remove DataParallel
    return model


RGBmean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).cuda()
RGBstd = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).cuda()


def preprocess_fun(imgtsr, imgsize=224, ):
    """Manually write some version of preprocessing"""
    imgtsr = F.interpolate(imgtsr, [imgsize, imgsize])
    return (imgtsr - RGBmean) / RGBstd


def preprocess_fun_np(imgarr, imgsize=224, device="cuda"):
    """Manually write some version of preprocessing"""
    if imgarr.dtype == np.uint8:
        imgarr = imgarr / 255.0
    imgtsr = torch.from_numpy(imgarr).float().permute([0, 3, 1, 2])
    imgtsr = F.interpolate(imgtsr, [imgsize, imgsize]).to(device)
    return (imgtsr - RGBmean.to(device)) / RGBstd.to(device)


def get_act_dynamics(model, area, sublayer, channum, imgarr):
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    with torch.no_grad():
        model(preprocess_fun(torch.randn(1, 3, 224, 224).cuda()))

    tsr = fetcher["target"][0]
    total_Tstep = len(fetcher["target"])
    _, C, H, W = tsr.shape
    pos = (H // 2, W // 2)

    imgarr = np.array(imgcol)
    fetcher.activations["target"] = []
    with torch.no_grad():
        imgbatch = preprocess_fun_np(imgarr)
        model(imgbatch)
    scores_trace = np.array([
        fetcher["target"][time_step][:, channum, pos[0], pos[1]].numpy()
        for time_step in range(total_Tstep)])

    fetcher.remove_hook()
    del fetcher
    return scores_trace


def plot_score_traces(scores_trace, chanlabel, total_Tstep, runnum=5):
    lstyles = ["-", "--", "-.", ":"]
    colorcyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    figh = plt.figure()
    for T in range(total_Tstep):
        plt.plot(scores_trace[:, T * runnum],
                 lw=2.5, c=colorcyc[T], alpha=0.65, label="ActMax T%d" % (T + 1))  # linestyle=lstyles[T],
        plt.plot(scores_trace[:, T * runnum + 1: (T + 1) * runnum],
                 lw=2.5, c=colorcyc[T], alpha=0.65, )  # linestyle=lstyles[T],
    plt.xticks(range(total_Tstep), range(1, total_Tstep + 1))
    plt.xlabel("recurrent step", fontsize=14)
    plt.ylabel("activation", fontsize=14)
    plt.title(f"{chanlabel} Activ Dynamics", fontsize=14)
    plt.legend()
    plt.show()
    return figh


def plot_score_heatmap(scores_trace, chanlabel, total_Tstep, runnum=5):
    ncols = total_Tstep * runnum
    figh = plt.figure(figsize=[12 / 20 * ncols, 2.5])
    if np.median(scores_trace) > 10:
        fmt = ".0f"
    else:
        fmt = ".1f"
    sns.heatmap(scores_trace, annot=True, yticklabels=np.arange(1, total_Tstep + 1), fmt=fmt)
    plt.axis("equal")
    plt.vlines(np.arange(runnum, ncols, runnum), 0, total_Tstep, color='green', linewidth=2.0)
    plt.xticks(np.arange(2, ncols, runnum), ["ActMax T%d" % T for T in range(total_Tstep)])
    plt.ylabel("recurrent step")
    plt.xlabel("Actmax Images")
    plt.title(f"{chanlabel} Activ Dynamics", fontsize=12)
    plt.tight_layout()
    plt.show()
    return figh


def sweep_act_dynamics(model, area, sublayer, chanrng, outdir):
    datadir = join(dataroot, f"{area}-{sublayer}")
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    # prep for fetching
    with torch.no_grad():
        model(preprocess_fun(torch.randn(1, 3, 224, 224).cuda()))
    tsr = fetcher["target"][0]
    total_Tstep = len(fetcher["target"])
    _, C, H, W = tsr.shape
    pos = (H // 2, W // 2)
    scores_trace_col = []
    for channum in tqdm(range(chanrng[0], chanrng[1])):
        chanlabel = f"{area}-{sublayer}-Ch{channum:03d}"
        imgcol = []
        for time_step in range(total_Tstep):
            for runi in range(runnum):
                explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runi:02d}"
                img = plt.imread(join(datadir, "bestimg_%s.jpg" % (explabel)))
                imgcol.append(img)

        imgarr = np.array(imgcol)
        fetcher.activations["target"] = []
        with torch.no_grad():
            imgbatch = preprocess_fun_np(imgarr)
            model(imgbatch)
        scores_trace = np.array([
            fetcher["target"][time_step][:, channum, pos[0], pos[1]].numpy()
            for time_step in range(total_Tstep)])

        # plotting the traces
        figh = plot_score_traces(scores_trace, chanlabel, total_Tstep, runnum=5)
        figh.savefig(join(outdir, f"{chanlabel}_act_traces.png"))
        # plotting the heatmap
        figh2 = plot_score_heatmap(scores_trace, chanlabel, total_Tstep, runnum=5)
        figh2.savefig(join(outdir, f"{chanlabel}_act_heatmap.png"))
        # save the scores data
        np.savez(join(outdir, f"{chanlabel}_actdata.npz"),
                 scores_trace=scores_trace,
                 runnum=runnum, Tstep=total_Tstep,
                 area=area, sublayer=sublayer, channum=channum, pos=pos, )
        scores_trace_col.append(scores_trace)

    fetcher.remove_hook()
    del fetcher
    return scores_trace_col


# %%

model = get_model()
# %%
outdir = r"F:\insilico_exps\CorNet-recurrent-evol\actdyn_summary"
scores_trace_col = sweep_act_dynamics(model, "V2", "output", (0, 50), outdir)
scores_trace_col = sweep_act_dynamics(model, "V4", "output", (0, 50), outdir)
scores_trace_col = sweep_act_dynamics(model, "IT", "output", (0, 100), outdir)

# %%
outdir = r"F:\insilico_exps\CorNet-recurrent-evol\actdyn_summary"
scores_trace_col = sweep_act_dynamics(model, "V4", "output", (50, 100), outdir)
# %% Dev zone
figh = plot_score_traces(scores_trace, chanlabel, total_Tstep, runnum=5)
figh.savefig(join(outdir, f"{chanlabel}_act_traces.png"))
figh2 = plot_score_heatmap(scores_trace, chanlabel, total_Tstep, runnum=5)
figh2.savefig(join(outdir, f"{chanlabel}_act_heatmap.png"))
np.savez(join(outdir, f"{chanlabel}_actdata.npz"),
         scores_trace=scores_trace,
         runnum=runnum, Tstep=total_Tstep,
         area=area, sublayer=sublayer, channum=channum, pos=pos, )

# %% Dev zone
chanrng = [0, 2]
area = "V4"
timestepN = 4
runnum = 5
datadir = join(dataroot, "%s-%s" % (area, sublayer))
for channum in tqdm(range(chanrng[0], chanrng[1])):
    outlabel = f"{area}-{sublayer}-Ch{channum:03d}_allproto_mtg"
    imgcol = []
    for time_step in range(timestepN):
        for runi in range(runnum):
            explabel = f"{area}-{sublayer}-Ch{channum:03d}-T{time_step:d}-run{runi:02d}"
            img = plt.imread(join(datadir, "bestimg_%s.jpg" % (explabel)))
            imgcol.append(img)
    break
# %%

imgarr = np.array(imgcol)