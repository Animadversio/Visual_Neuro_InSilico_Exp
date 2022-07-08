import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from glob import glob
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *
from scipy.stats import pearsonr, spearmanr
from stats_utils import saveallforms
from NN_sparseness.visualize_sparse_inv_example import *
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.sparse_plot_utils import scatter_density_grid
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
sumdir  = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
figdir  = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"
#%%
netname = "resnet50_linf8"
exampledir = join(rootdir, f"tuning_map_examples_{netname}")

Invfeatdata = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
INdataset = create_imagenet_valid_dataset(normalize=False)
#%%
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
#%%
from torchvision.transforms.functional import rotate, affine, resize, center_crop
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
    # invar_cc = calc_invariance(invrespvec)
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
        plt.title(f"Invariance ") #{invar_cc.item():.2f}
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    else:
        figh = None
    return augimgtsr, tuningmap, figh
#%%
def sample_idxs_by_bins(INresps, normthresh=1.0, bins=None, idx_per_bin=5):
    if bins is None:
        bins = [(0.0, 0.2),
                (0.2, 0.4),
                (0.4, 0.6),
                (0.6, 0.8),
                (0.8, 1.0),
                (1.0, 1.3),
                (1.3, 100),
                ]
    idx_all = []
    for LB, UB in bins:
        msk = (INresps > LB * normthresh) * (INresps < UB * normthresh)
        if msk.sum() == 0.0:
            continue
        elif msk.sum() < idx_per_bin:
            idx_bin = msk.nonzero()[:, 0]
        else:
            idx_bin = torch.multinomial(msk.float(), idx_per_bin)
        idx_all.extend(idx_bin)
    idx_all.append(INresps.argmax())
    return idx_all
#%%
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\actlevel_tolerence"
# layer_long = '.layer3.Bottleneck4'
layer_long = '.layer4.Bottleneck0'
# layer_long = '.layer3.Bottleneck2'
unit_id = 500
for layer_long in feattsrs.keys():
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    for unit_id in range(100, 250, 10,): # feattsrs[layer_long].shape[1]
        if unit_id >= feattsrs[layer_long].shape[1]: continue
        INresps = feattsrs[layer_long][:, unit_id]
        normthresh = INresps.quantile(0.998)
        idx_all = sample_idxs_by_bins(INresps, normthresh, idx_per_bin=10)
        imgtsrs = [INdataset[idx][0] for idx in idx_all]
        augimgtsr, tuningmap, figh = measure_invariance(model, layer_long, unit_id,
                    imgtsrs, netname=netname, plot=False)
        #%%
        Nobj, NTfm = tuningmap.shape
        centact = tuningmap[:, 0]
        meanact = tuningmap.mean(dim=1)
        maxact  = tuningmap.max(dim=1).values
        obj_toler = (tuningmap / maxact[:, None]).sum(dim=1) / (NTfm - 1)
        rho_m, pval_m = pearsonr(meanact, obj_toler)
        rho_x, pval_x = pearsonr(maxact, obj_toler)
        rho_c, pval_c = pearsonr(centact, obj_toler)
        print(f"{netname} {layer_short} unit {unit_id}")
        print(f"object tolerence ~ cent resp {rho_c:.2f} (P={pval_c:.1e})\t mean resp {rho_m:.2f} (P={pval_m:.1e})\t max resp {rho_x:.2f} (P={pval_x:.1e})")
        #%%
        df = pd.DataFrame({"imgidx": idx_all, "centact": centact, "meanact": meanact, "maxact": maxact, "obj_toler": obj_toler})
        df.to_csv(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}.csv"), index=False)
        # .to_csv(join(outdir, "resnet50_linf8_INvalid_tuningmap.csv"))
        g = scatter_density_grid(df, ["obj_toler", "centact", "meanact", "maxact"], )
        g.fig.suptitle(f"{netname} {layer_long} unit {unit_id}")
        g.tight_layout()
        g.fig.show()
        saveallforms(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}", g.fig)
        # g.fig.savefig(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}.png"))

#%% summary
for layer_long in feattsrs.keys():
    if "fc" in layer_long: continue
    layer_short = shorten_layername(layer_long)
    for unit_id in range(0, 250, 10,): # feattsrs[layer_long].shape[1]
        df = pd.read_csv(join(figdir, f"{netname}_{layer_short}_unit{unit_id:04d}.csv"), )

