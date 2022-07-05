"""
Recreate Zocollan 2007 in silico.

"""
import matplotlib.pyplot as plt
from stats_utils import saveallforms
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.visualize_sparse_inv_example import *
from NN_sparseness.sparse_invariance_lib import shorten_layername
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
sumdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary"
figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\summary_figs"
#%%
netname = "resnet50_linf8"
exampledir = join(rootdir, f"tuning_map_examples_{netname}")

Invfeatdata = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
# INdataset = create_imagenet_valid_dataset(normalize=False)

#%%
Nobj, Ntfm = 10, 6
layer_long = '.layer2.Bottleneck3'
for layer_long in feattsrs.keys():
    layer_short = shorten_layername(layer_long)
    invrespmat = Invfeatdata[layer_long]  # = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
    INrespmat = feattsrs[layer_long]  # = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
    tuning_tsr = invrespmat.reshape([Nobj, Ntfm, -1])
    #%%
    maxresp_per_obj = tuning_tsr.max(dim=1, keepdim=True).values
    # Tfmtoler = (tuning_tsr / maxresp_per_obj).nanmean(dim=[0,1])
    if "fc" not in layer_long:
        # Tfmtoler = ((tuning_tsr / maxresp_per_obj).nansum(dim=[0, 1]) - Nobj) / Nobj / (Ntfm - 1)
        Tfmtoler = (tuning_tsr / maxresp_per_obj).nanmean(dim=[0]).sum(dim=0) / (Ntfm - 1)
        sparseness = 1 - (INrespmat.mean(dim=0) ** 2) / (INrespmat ** 2).mean(dim=0)
    else:
        minresp_thr = INrespmat.quantile(0.5, dim=0, keepdim=True)  # .values
        Tfmtoler = (torch.clamp(tuning_tsr - minresp_thr.unsqueeze(1), 0) /
                    torch.clamp(maxresp_per_obj - minresp_thr.unsqueeze(1), 0)).nanmean(dim=[0]).sum(dim=0) / (Ntfm - 1)
        clamp_resp = torch.clamp(INrespmat - minresp_thr, 0)
        sparseness = 1 - (clamp_resp.mean(dim=0) ** 2) / (clamp_resp ** 2).mean(dim=0)
    #%%
    valmsk = (~sparseness.isnan()) & (~Tfmtoler.isnan())
    rho, pval = pearsonr(sparseness[valmsk], Tfmtoler[valmsk])
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=sparseness, y=Tfmtoler, )
    plt.ylabel("Absolute Tolerence (in invariance img)") #to Transforms
    plt.xlabel("Sparseness (in ImageNet 50k)")
    plt.title(f"{netname} {layer_short}\nrho={rho:.2f} (p={pval:.1e})")
    plt.tight_layout()
    # saveallforms(figdir, f"sparseness_vs_toler_{netname}_{layer_short}")
    plt.show()
    #%%
    print(f"{netname} {layer_short} Sparseness{sparseness[valmsk].mean():.2f}+-{sparseness[valmsk].std():.2f}"
          f"\nTolerence {Tfmtoler[valmsk].mean():.2f}+-{Tfmtoler[valmsk].std():.2f}")

#%%
Nobj, Ntfm = 10, 6
figh, axs = plt.subplots(1, len(feattsrs.keys())-1, figsize=(15, 3.5))
for li, layer_long in enumerate(feattsrs.keys()):
    if "fc" in layer_long:
        continue
    layer_short = shorten_layername(layer_long)
    invrespmat = Invfeatdata[layer_long]  # = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
    INrespmat = feattsrs[layer_long]  # = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
    tuning_tsr = invrespmat.reshape([Nobj, Ntfm, -1])
    #%%
    maxresp_per_obj = tuning_tsr.max(dim=1, keepdim=True).values
    # Tfmtoler = (tuning_tsr / maxresp_per_obj).nanmean(dim=[0,1])
    # Tfmtoler = ((tuning_tsr / maxresp_per_obj).nansum(dim=[0, 1]) - Nobj) / Nobj / (Ntfm - 1)
    Tfmtoler = (tuning_tsr / maxresp_per_obj).nanmean(dim=[0]).sum(dim=0) / (Ntfm - 1)
    sparseness = 1 - (INrespmat.mean(dim=0) ** 2) / (INrespmat ** 2).mean(dim=0)

    #%%
    valmsk = (~sparseness.isnan()) & (~Tfmtoler.isnan())
    rho, pval = pearsonr(sparseness[valmsk], Tfmtoler[valmsk])
    sns.scatterplot(x=sparseness, y=Tfmtoler, ax=axs[li], alpha=0.6)
    axs[li].set_title(f"{layer_short}\nrho={rho:.2f}\np={pval:.0e}")
axs[0].set(ylabel="Absolute Tolerence (in invariance img)") #to Transforms
axs[3].set(xlabel="Sparseness (in ImageNet 50k)")
plt.suptitle(f"{netname} Sparseness vs Tolerance")
figh.tight_layout()
saveallforms(figdir, f"sparseness_vs_toler_{netname}_alllayers")
plt.show()

#%%
unit = 180
tuning_map = tuning_tsr[:,:,unit]
maxact = tuning_map.max(dim=1, keepdim=False).values
tuning_normed = tuning_map / maxact[:, None]
obj_tolerence = tuning_normed.mean(dim=1)
meanact = tuning_map.mean(dim=1, keepdim=False)
msk = ~obj_tolerence.isnan()
rho_max, pval_max = pearsonr(obj_tolerence[msk], maxact[msk])
rho_mean, pval_mean = pearsonr(obj_tolerence[msk], meanact[msk])
print("corr with max %.3f (%.1e) with mean %.3f (%.1e)"%(rho_max, pval_max, rho_mean, pval_mean))

