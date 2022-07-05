import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from glob import glob
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.sparse_invariance_lib import *
from stats_utils import saveallforms
from NN_sparseness.visualize_sparse_inv_example import *
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"
#%%
netname = "resnet50_linf8"
exampledir = join(rootdir, f"tuning_map_examples_{netname}")

Invfeatdata = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
INdataset = create_imagenet_valid_dataset(normalize=False)
df_merge_all = pd.read_csv(join(sumdir, f"{netname}_sparse_invar_prctl_merge.csv"), index_col=0)

#%%
netname = "resnet50_linf8"
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
#%%
from torchvision.transforms.functional import rotate, affine, resize, center_crop
from NN_sparseness.sparse_invariance_lib import *
def calc_invariance(invrespvec):
    tuningmap = invrespvec.reshape(10, 7, -1).permute(2, 1, 0)  # 6 by 10 by Channel N
    unit_cctsr = corrcoef_batch(tuningmap)  # Chan, 6, 6
    unit_cctsr_msk = mask_diagonal(unit_cctsr)  # Chan, 6, 6
    invar_cc = unit_cctsr_msk.nanmean(dim=(1, 2))  # Chan,
    return invar_cc


def measure_invariance(model, layer_long, unit_id, imgtsrs, netname=netname, plot=True):
    if isinstance(imgtsrs, list):
        imgtsr = torch.stack(imgtsrs)
    else:
        imgtsr = imgtsrs
    imgstr_sml = affine(imgtsr, 0, [0, 0], scale=0.75, shear=[0, 0], fill=0.5)
    imgstr_lrg = affine(imgtsr, 0, [0, 0], scale=1.5, shear=[0, 0], fill=0.5)
    imgstr_lft = affine(imgtsr, 0, [45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rgt = affine(imgtsr, 0, [-45, 0], scale=1.0, shear=[0, 0], fill=0.5)
    imgstr_rot = rotate(imgtsr, 30, expand=False, center=None, fill=0.5)
    imgstr_rot_rev = rotate(imgtsr, -30, expand=False, center=None, fill=0.5)
    augimgtsr = torch.stack([imgtsr,
                             imgstr_lrg,
                             imgstr_sml,
                             imgstr_rgt,
                             imgstr_lft,
                             imgstr_rot,
                             imgstr_rot_rev,
                             ])
    augimgtsr = augimgtsr.permute([1, 0, 2, 3, 4])
    augimgtsr = augimgtsr.reshape(-1, *augimgtsr.shape[2:], )
    augimgtsr_pp = normalize(resize(augimgtsr, [256, 256]))
    invfeattsrs_new = record_imgtsrs_dataset(model, [layer_long], augimgtsr_pp)[layer_long]
    # feattsrs = torch.load(join(outdir, "resnet50_linf8_INvalid_feattsrs.pt"))
    invrespvec = invfeattsrs_new[:, unit_id]
    tuningmap = invrespvec.reshape(-1, 7)
    invar_cc = calc_invariance(invrespvec)
    if plot:
        mtgtsr = make_grid(augimgtsr,
                       nrow=7, padding=2, pad_value=0)
        figh = plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(mtgtsr.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"{netname} {layer_long} unit {unit_id}")
        plt.subplot(1, 2, 2)
        plt.imshow(tuningmap.numpy(), cmap="inferno")
        plt.title(f"Invariance {invar_cc.item():.2f}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    else:
        figh = None
    return augimgtsr, tuningmap, invar_cc, figh

#%%
layer_long, unit_id = '.layer4.Bottleneck0', 30 #
INresps = feattsrs[layer_long][:, unit_id]
#%%
layer_long, unit_id = '.Linearfc', 20#'.layer4.Bottleneck0'
INresps = feattsrs[layer_long][:, unit_id]
#%%
resp_topk, idx_topk = torch.topk(INresps, 10)
#%%
idx_topk = torch.randint(0, INresps.shape[0], (10,))
imgtsrs = [INdataset[idx][0] for idx in idx_topk]
augimgtsr, tuningmap, invar_cc, figh = \
    measure_invariance(model, layer_long, unit_id, imgtsrs, )
#%%
msk = (INresps < .5) #& (INresps > 0.0)
idx_k = torch.multinomial(msk.float(), 10)
imgtsrs = [INdataset[idx][0] for idx in idx_k]
augimgtsr, tuningmap, invar_cc, figh = \
    measure_invariance(model, layer_long, unit_id, imgtsrs, )
#%%
msk = (INresps > 0.5)
idx_k = torch.multinomial(msk.float(), 10)
imgtsrs = [INdataset[idx][0] for idx in idx_k]
augimgtsr, tuningmap, invar_cc, figh = \
    measure_invariance(model, layer_long, unit_id, imgtsrs, )
#%%
#%%
layer_long, unit_id = ('.layer4.Bottleneck2',5)
unitrow = df_merge_all[(df_merge_all.layer_x == layer_long)  \
                     & (df_merge_all.unitid == unit_id)].iloc[0]
visualize_unit_data_montage(unitrow, netname, Invfeatdata,
                                feattsrs, INdataset, topk=4)
#%%
topN = 2
INresps = feattsrs[layer_long][:, unit_id]
normthresh = INresps.quantile(0.998)
msk = INresps > 0.75 * normthresh
if topN > 0:
    idx_topk = torch.multinomial(msk.float(), topN)
    idx_rand = torch.randint(0, INresps.shape[0], (10 - topN,))
    idx_k = torch.cat([idx_topk, idx_rand])
else:
    idx_k = torch.randint(0, INresps.shape[0], (10 - topN,))
imgtsrs = [INdataset[idx][0] for idx in idx_k]
augimgtsr, tuningmap, invar_cc, figh = \
    measure_invariance(model, layer_long, unit_id, imgtsrs, )

#%%
import seaborn as sns
layer_long, unit_id = '.Linearfc', 30  #
for layer_long, unit_id in [('.Linearfc', 20),
                            ('.layer4.Bottleneck2', 20),
                            ('.layer4.Bottleneck0', 20),
                            ('.layer3.Bottleneck4', 20),
                            ('.layer3.Bottleneck2', 20),
                            ('.layer2.Bottleneck3', 20),
                            ('.layer2.Bottleneck0', 20),
                            ('.layer1.Bottleneck1', 20),
                            ]: #
    INresps = feattsrs[layer_long][:, unit_id]
    normthresh = INresps.quantile(0.998)
    msk = INresps > 0.75 * normthresh
    #%%
    inv_df = []
    for topN in range(11):
        for i in range(10):
            if topN > 0:
                idx_topk = torch.multinomial(msk.float(), topN)
                idx_rand = torch.randint(0, INresps.shape[0], (10 - topN,))
                idx_k = torch.cat([idx_topk, idx_rand])
            else:
                idx_k = torch.randint(0, INresps.shape[0], (10,))
            imgtsrs = [INdataset[idx][0] for idx in idx_k]
            _, tuningmap, invar_cc, _ = \
                measure_invariance(model, layer_long, unit_id, imgtsrs, plot=False)
            entry = {"topN":topN, "unit_inv":invar_cc.item(),
                                    "obj_resp_norm_std": (tuningmap.mean(axis=1).std()/normthresh).item(),
                                    "inv_resp_norm_mean": (tuningmap.mean() /normthresh).item(),
                                    "inv_resp_norm_max": (tuningmap.max() / normthresh).item(),
                                    "unit_id":unit_id, "layer_long":layer_long}
            inv_df.append(pd.DataFrame(entry, index=[i]))
            print(pd.Series(entry))
    #%
    df_inv = pd.concat(inv_df)
    #%%
    savedir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\actlevel_invariance"
    df_inv.to_csv(join(savedir, f"{netname}_{layer_long}_unit{unit_id}_inv_tuning_df.csv"))
    #%
    plt.figure(figsize=(5, 5))
    sns.lineplot(x="topN", y="unit_inv", data=df_inv)
    sns.stripplot(x="topN", y="unit_inv", data=df_inv)
    plt.title(f"{netname} {layer_long} unit {unit_id}")
    saveallforms(savedir, f"unit_inv_vs_top_resp_N_{netname}_{layer_long}_unit{unit_id}")
    plt.show()
    #%
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x="inv_resp_norm_mean", y="unit_inv", data=df_inv)
    plt.title(f"{netname} {layer_long} unit {unit_id}")
    saveallforms(savedir, f"unit_inv_vs_norm_resp_mean_{netname}_{layer_long}_unit{unit_id}")
    plt.show()

    #%
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x="obj_resp_norm_std", y="unit_inv", hue="topN",
                    data=df_inv, palette="rainbow",)
    plt.title(f"{netname} {layer_long} unit {unit_id}")
    saveallforms(savedir, f"unit_inv_vs_obj_resp_norm_std_{netname}_{layer_long}_unit{unit_id}")
    plt.show()
    # break
    #%%
    # sns.scatterplot(x="inv_resp_norm_max", y="unit_inv", data=df_inv)
    # plt.show()
#%%
df_col = []
for layer_long, unit_id in [('.layer1.Bottleneck1', 5),
                            ('.layer2.Bottleneck0', 5),
                            ('.layer2.Bottleneck3', 5),
                            ('.layer3.Bottleneck2', 5),
                            ('.layer3.Bottleneck4', 5),
                            ('.layer4.Bottleneck0', 5),
                            ('.layer4.Bottleneck2', 5),
                            ('.Linearfc', 5),
                            ]:
    df = pd.read_csv(join(savedir, f"{netname}_{layer_long}_unit{unit_id}_inv_tuning_df.csv"),
                     index_col=0)
    df_col.append(df)
df_layers = pd.concat(df_col, axis=0)
#%%
figh = plt.figure(figsize=(6, 6))
sns.lineplot(x="topN", y="unit_inv", hue="layer_long",
             data=df_layers, palette="rainbow",)
plt.show()


#%%
