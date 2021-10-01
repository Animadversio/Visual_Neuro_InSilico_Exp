"""BigGAN Evol Figure
Partially inherit from BigGAN_Evol_summary.py but more focused. 
"""
import os
import re
from time import time
from glob import glob
from os.path import join
from easydict import EasyDict
from imageio import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy.stats import ttest_rel, ttest_ind
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#%%
dataroot = r"E:\Cluster_Backup\BigGAN_Optim_Tune_new"
figdir = r"O:\BigGAN_FC6_insilico"
outdir = r"O:\ThesisProposal\BigGAN"
summarydir = join(figdir, "summary")
os.makedirs(summarydir, exist_ok=True)

# load the table that indexing all experiments
exptab = pd.read_csv(join(summarydir, "optim_raw_score_tab.csv"))
exptab.suffix.fillna("", inplace=True)  # substitute nan as ""
exptab.suffix = exptab.suffix.astype(str)
rfmsk = exptab.layer.str.contains("fc") | exptab.suffix.str.contains("RF")  # Experiments that do RFfitting
fullmsk = exptab.layer.str.contains("fc") | ~exptab.suffix.str.contains("RF")  # Experiments that use full size images
# two masks are overlapping.
#%%
def crop_from_montage(img, imgid:int=-1, imgsize=256, pad=2):
    nrow, ncol = (img.shape[0] - pad) // (imgsize + pad), (img.shape[1] - pad) // (imgsize + pad)
    if imgid < 0: imgid = nrow * ncol + imgid
    ri, ci = np.unravel_index(imgid, (nrow, ncol))
    img_crop = img[pad + (pad+imgsize)*ri:pad + imgsize + (pad+imgsize)*ri, \
                   pad + (pad+imgsize)*ci:pad + imgsize + (pad+imgsize)*ci, :]
    return img_crop

row = exptab.loc[25]
npzpath = join(dataroot, row.unitstr, 'scores%s_%05d.npz'%(row.optimizer, row.RND))
imgtrajpath = glob(join(dataroot, row.unitstr, "besteachgen%s_%05d.jpg"%(row.optimizer, row.RND)))[0]
data = np.load(npzpath)
evolmtg = imread(imgtrajpath)
proto = crop_from_montage(evolmtg)

#%%
for layer in exptab.layer.unique():
    unit_cols = exptab.unitstr[(exptab.layer==layer) & rfmsk].unique()
    # for unitstr in unit_cols:
    unitstr = np.random.choice(unit_cols, 1)[0]
unitstr = "alexnet_conv5_33_RFrsz"
for optim in ['CholCMA', 'HessCMA', 'CholCMA_fc6']:
    msk = (exptab.unitstr==unitstr) & (exptab.optimizer==optim) & rfmsk
    print(layer, optim, unitstr, exptab[msk].shape)
    # row = exptab[msk].sample(1).iloc[0]

#%% Plot the optim trajectory comparison! 
from PIL import Image
from build_montages import make_grid_np
from stats_utils import summary_by_block, saveallforms
def load_data_from_row(row, imgid=-1):
    npzpath = join(dataroot, row.unitstr, 'scores%s_%05d.npz' % (row.optimizer, row.RND))
    imgtrajpath = glob(join(dataroot, row.unitstr, "besteachgen%s_%05d.jpg" % (row.optimizer, row.RND)))[0]
    data = np.load(npzpath)
    evolmtg = imread(imgtrajpath)
    proto = crop_from_montage(evolmtg, imgid=imgid)
    scorevec = data['scores_all']
    genvec = data["generations"]
    score_m, score_s, blockarr = summary_by_block(scorevec, genvec, sem=False)
    return scorevec, genvec, score_m, score_s, blockarr, proto

def shadedErrorbar(blockarr, score_m, score_s, alpha=0.2, linealpha=1.0, label=None, color=None, linecolor=None):
    L = plt.plot(blockarr, score_m, label=label, c=linecolor, alpha=linealpha)
    if color is None: color = L[0]._color
    plt.fill_between(blockarr, score_m-score_s, score_m+score_s, alpha=alpha, color=color)

# unitstr = "alexnet_conv5_31_RFrsz"
# unitstr = "alexnet_conv1_33_RFrsz"

unitstrs = ["alexnet_conv2_30_RFrsz",
            "alexnet_conv3_32_RFrsz",
            "alexnet_conv4_31_RFrsz",
            #"alexnet_conv5_30_RFrsz",
            "alexnet_conv5_33_RFrsz",
            #"alexnet_fc6_30",
            "alexnet_fc6_32",
            "alexnet_fc7_33",
            "alexnet_fc8_31",]
mtg_col = []
ncol = len(unitstrs)
figh, axs = plt.subplots(1, ncol, figsize=[ncol*2.75, 2.75])
for ci, unitstr in enumerate(unitstrs):
    proto_col = []
    for optim in ['CholCMA', 'HessCMA', 'CholCMA_fc6']:
        msk = (exptab.unitstr == unitstr) & (exptab.optimizer == optim)# & rfmsk
        if "fc6" not in optim: optim+="_BigGAN"
        print(layer, optim, unitstr, exptab[msk].shape)
        # row = exptab[msk].sample(1).iloc[0]
        maxid = exptab[msk].score.argmax()
        row = exptab[msk].loc[maxid]
        _, _, score_m, score_s, blockarr, proto = load_data_from_row(row)
        proto_col.append(proto)
        plt.sca(axs[ci])
        shadedErrorbar(blockarr, score_m, score_s, label=optim)

    plt.title(unitstr)
    if ci == 0: plt.legend()
    mtg = make_grid_np(proto_col, nrow=1)  # np.stack(tuple(proto_col), axis=3)
    mtg_col.append(mtg)
saveallforms([figdir, outdir], "best_trajs_exemplar_alllayer")
plt.show()
mtg_full = make_grid_np(mtg_col, nrow=ncol)
mtg_PIL = Image.fromarray(mtg_full)
mtg_PIL.show()
mtg_PIL.save(join(figdir, "proto_cmp_alllayers.jpg"))
mtg_PIL.save(join(outdir, "proto_cmp_alllayers.jpg"))

#%%
#%%
# compare performance across optimizers.
# BigGAN FC6 pair alignment
optimnames = ["CholCMA", "HessCMA", "CholCMA_fc6"]
def BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=True):
    Scol = []
    overallmsk = rfmsk if rffit else fullmsk 
    unitstr_uniq = exptab.loc[overallmsk].unitstr.unique()
    for unitstr in unitstr_uniq:
        unitmsk = (exptab.unitstr == unitstr)
        unit = exptab.unit[unitmsk].iloc[0]
        layer = exptab.layer[unitmsk].iloc[0]
        unitfc6msk = unitmsk & overallmsk & (exptab.optimizer=="CholCMA_fc6")
        unitBGmsk = unitmsk & overallmsk & (exptab.optimizer=="CholCMA")
        unitBGHmsk = unitmsk & overallmsk & (exptab.optimizer=="HessCMA")
        unitoptimmsk = unitmsk & overallmsk & ((exptab.optimizer=="CholCMA")\
                                            |  (exptab.optimizer=="HessCMA")\
                                            |  (exptab.optimizer=="CholCMA_fc6"))
        if any([sum(unitfc6msk)==0, sum(unitBGmsk)==0, sum(unitBGHmsk)==0]): continue
        if norm_scheme is "allmax":
            normalizer = exptab.score[unitoptimmsk].max()
        elif norm_scheme is "fc6max":
            normalizer = exptab.score[unitfc6msk].max()
        elif norm_scheme is "fc6mean":
            normalizer = exptab.score[unitfc6msk].mean()
        else: 
            raise NotImplementedError
         # no fc6 to normalize to
        for optim in optimnames:
            msk = unitmsk & overallmsk & (exptab.optimizer==optim)
            scorevec = exptab.score[msk]
            scorevec_norm = scorevec / normalizer
            newdicts = [{"layer":layer,"unit":unit,"optimizer":optim,"score":score,"score_norm":score_norm}
                        for score, score_norm in zip(scorevec, scorevec_norm)]
            Scol.extend(newdicts)

    BigGANFC6cmptab = pd.DataFrame(Scol)
    deadunitmsk = (BigGANFC6cmptab.score_norm.isna())
    fc6failedmsk = np.isinf(BigGANFC6cmptab.score_norm)
    print("Dead channel trial number:%d"%sum(deadunitmsk))
    print("Failed trial number:%d"%sum(fc6failedmsk))
    figh = plt.figure(figsize=[7, 7])
    ax = sns.violinplot(x='layer', y='score_norm', hue="optimizer", jitter=0.25,
                       hue_order=['CholCMA', 'HessCMA', 'CholCMA_fc6'], cut=0.1,
                       data=BigGANFC6cmptab[~deadunitmsk&~fc6failedmsk], alpha=0.4)
    ax.set_title("Comparison of Optimizer and GAN space over Units of AlexNet %s"%("RFfit" if rffit else "FullImage"))
    ax.set_ylabel("activ normalized by %s"%norm_scheme)
    ax.figure.show()
    ax.figure.savefig(join(summarydir, "BigGANFC6_cmp_%snorm_layer_all%s.jpg"%(norm_scheme, "RFfit" if rffit else
                    "_Full")))
    ax.figure.savefig(join(summarydir, "BigGANFC6_cmp_%snorm_layer_all%s.pdf"%(norm_scheme, "RFfit" if rffit else 
                    "_Full")))
    return BigGANFC6cmptab, figh

# BigGANFC6_comparison_plot(norm_scheme="fc6max", rffit=True)
# BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=True)
BigGANFC6_comparison_plot(norm_scheme="allmax", rffit=False)