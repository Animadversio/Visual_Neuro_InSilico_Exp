"""Relating tuning width with sparseness"""
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from NN_sparseness.insilico_manif_configs import RN50_config
from NN_sparseness.sparse_invariance_lib import shorten_layername
from NN_sparseness.sparse_plot_utils import annotate_corrfunc
from stats_utils import saveallforms
dataroot = r"E:\Cluster_Backup\manif_allchan"
rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
sumdir  = join(rootdir, "summary")
figdir  = join(rootdir, "summary_figs")
#%%
mergedf = pd.read_csv(join(sumdir, r"resnet50_linf8_sparse_invar_prctl_merge.csv"), index_col=0)
kappadf = pd.read_csv(join(dataroot, "summary", r"resnet50_linf_8_ManifExpFitSum_RFfit.csv"), index_col=0)
#%%
nettab_d = pd.read_csv(join(dataroot, "summary", r"resnet50_linf_8_ManifExpFitSum_RFfit.csv"), index_col=0)
nettab_nonparam = pd.read_csv(join(dataroot, "summary", r"resnet50_linf_8_ManifExpNonParamSum_RFfit.csv"), index_col=0)
# merge and construct the sparseness vs kappa table.
kappa_nonparam_df = pd.merge(kappadf, nettab_nonparam[["layer", "iCh", "space", "manifsparseness","normVUS"]], on=["layer", "iCh", "space"])
kappa_sprs_df = pd.merge(mergedf, kappa_nonparam_df, left_on=["layer", "unitid"], right_on=["layer", "iCh"])
#%%
# EMdir = r"E:\Cluster_Backup\CNN_manifold\summary"
# kappadf = pd.read_csv(join(EMdir,r"resnet50_linf_8_ManifExpFitSum_RFfit.csv"))
#%%
netname = "resnet50_linf_8"
unit_list = [("resnet50_linf_8", ".ReLUrelu", 5, 57, 57, True),
             ("resnet50_linf_8", ".layer1.Bottleneck1", 5, 28, 28, True),
             ("resnet50_linf_8", ".layer2.Bottleneck0", 5, 14, 14, True),
             ("resnet50_linf_8", ".layer2.Bottleneck2", 5, 14, 14, True),
             ("resnet50_linf_8", ".layer3.Bottleneck0", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer3.Bottleneck2", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer3.Bottleneck4", 5, 7, 7, True),
             ("resnet50_linf_8", ".layer4.Bottleneck0", 5, 4, 4, False),
             ("resnet50_linf_8", ".layer4.Bottleneck2", 5, 4, 4, False),
             ("resnet50_linf_8", ".Linearfc", 5, False)]
layerlist = [unit[1] for unit in unit_list]
layerlabel = [name[1:] for name in layerlist]
layerlabel[0] = "Relu"
#%%
from Manifold.Manifold_Tuning_lib import load_fit_manif2table, load_nonparam_manif2table

nettab_d = load_fit_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")
nettab_nonparam = load_nonparam_manif2table(unit_list, netname, dataroot, save=True, savestr="_RFfit")


#%%
nettab_d = pd.read_csv(join(dataroot, "summary", r"resnet50_linf_8_ManifExpFitSum_RFfit.csv"), index_col=0)
nettab_nonparam = pd.read_csv(join(dataroot, "summary", r"resnet50_linf_8_ManifExpNonParamSum_RFfit.csv"), index_col=0)
#%%
# kappa_sprs_df = pd.merge(mergedf, kappadf, left_on=["layer", "unitid"], right_on=["layer", "iCh"])
#%%
kappa_nonparam_df = pd.merge(kappadf, nettab_nonparam[["layer", "iCh", "space", "manifsparseness","normVUS"]], on=["layer", "iCh", "space"])
kappa_sprs_df = pd.merge(mergedf, kappa_nonparam_df, left_on=["layer", "unitid"], right_on=["layer", "iCh"])

#%%
"""
Printing statistics across layers 
"""
nettab_nonparam.groupby(["layer","space"],sort=False)["manifsparseness","normVUS"].agg(["mean","sem"])
#%%
kappa_sprs_df[kappa_sprs_df.space==0].groupby(["layer", ], sort=False)["kappa"].mean()
#%%
kappa_sprs_df[kappa_sprs_df.space==0].groupby(["layer", ], sort=False)["kappa", "sparseness"].corr(method="spearman")
#%%
kappa_sprs_df[kappa_sprs_df.space==0][["kappa", "sparseness"]].corr(method="spearman")
#%%
""" Summary plots
 * kappa vs sparseness
 * normVUS vs sparseness
 * manifold sparseness vs sparseness 
"""
#%%
msk = (kappa_sprs_df.space == 0) & (kappa_sprs_df.kappa<10) \
      & (kappa_sprs_df.layer_s != "fc")

g = sns.FacetGrid(kappa_sprs_df[msk], col="layer_s",
                  height=4, aspect=1.0, sharex=False)
g.map(sns.scatterplot, "kappa", "sparseness", alpha=0.25)#.add_legend()
g.map(annotate_corrfunc, "kappa", "sparseness", xy=(.05, .1))
saveallforms(figdir, f"{netname}_kappa-sparseness_per_layer_mod", g.fig)
plt.show()
#%%
msk = (kappa_sprs_df.space == 0) \
      & (kappa_sprs_df.layer_s != "fc")
g = sns.FacetGrid(kappa_sprs_df[msk], col="layer_s",
                  height=4, aspect=1.0, sharex=True)
g.map(sns.scatterplot, "normVUS", "sparseness", alpha=0.25)#.add_legend()
g.map(annotate_corrfunc, "normVUS", "sparseness", xy=(.05, .1))
saveallforms(figdir, f"{netname}_normVUS-sparseness_per_layer_mod", g.fig)
plt.show()
#%%
msk = (kappa_sprs_df.space == 0) \
      & (kappa_sprs_df.layer_s != "fc")
g = sns.FacetGrid(kappa_sprs_df[msk], col="layer_s",
                  height=4, aspect=1.0, sharex=True)
g.map(sns.scatterplot, "manifsparseness", "sparseness", alpha=0.25)#.add_legend()
g.map(annotate_corrfunc, "manifsparseness", "sparseness", xy=(.05, .1))
saveallforms(figdir, f"{netname}_manif-INetsparseness_per_layer_mod", g.fig)
plt.show()
