import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from glob import glob
from scipy.stats import kurtosis
from dataset_utils import ImagePathDataset, ImageFolder
from NN_PC_visualize.NN_PC_lib import *


def Invariance_dataset():
    img_src = r"N:\Stimuli\Invariance\Project_Manifold\ready"
    imglist = sorted(glob(join(img_src, "*.jpg")))
    return ImagePathDataset(imglist, None)


def shorten_layername(s):
    return  s.replace(".layer", "layer").replace("Bottleneck", "B").replace(".Linear", "")


def corrcoef_batch(feattsr):
    """
    :param feattsr: B,T,C
    :return: batched correlation tensor for each sample, B,T,T
    """
    feattsr_cnt = feattsr - feattsr.mean(dim=2, keepdim=True)
    feattsr_norm = feattsr_cnt.norm(dim=2, keepdim=False)
    inprod = torch.einsum("BTI,BtI->BTt", feattsr_cnt, feattsr_cnt)
    corrcoef_all = inprod / feattsr_norm.unsqueeze(2) / feattsr_norm.unsqueeze(1)
    return corrcoef_all


def mask_diagonal(batch_cctsr):
    B, C, C2 = batch_cctsr.shape
    assert C==C2
    mask = torch.zeros((C, C))
    mask = mask + torch.diag(torch.nan * torch.zeros(C))
    batch_cctsr_msk = batch_cctsr + mask.unsqueeze(0)
    return batch_cctsr_msk


#%% Calculate statistics
def calculate_sparseness(feattsrs, subsample=False, sample_size=10000, layeralias=None):
    if isinstance(layeralias, dict):
        layeralias = layeralias.__getitem__
        #lambda x: layermap_inv[x]
    df_all = pd.DataFrame()
    sparseness_coef_D = {}
    kurtosis_coef_D = {}
    for layer in feattsrs:
        if subsample:
            mask = np.random.choice(feattsrs[layer].shape[0], sample_size, replace=False)
            featmat = feattsrs[layer][mask, :]
        else:
            featmat = feattsrs[layer]
        s_coef = (1 - featmat.mean(dim=0)**2 / featmat.pow(2).mean(dim=0)) / (1 - 1 / featmat.shape[0])
        zero_ratio = (featmat == 0.0).sum(dim=0) / featmat.shape[0]
        sparseness_coef_D[layer] = s_coef
        kurtosis_coef_D[layer] = kurtosis(featmat.numpy(), axis=0)
        df = pd.DataFrame({"sparseness":sparseness_coef_D[layer],
                           "zero_ratio":zero_ratio,
                           "kurtosis":kurtosis_coef_D[layer]})
        print(f"{layer} Sparseness {torch.nanmean(s_coef):.3f}+-{np.nanstd(s_coef):.3f}"
              f"  0 ratio {torch.mean(zero_ratio):.3f}+-{torch.std(zero_ratio):.3f}"
              f"  Kurtosis {np.mean(kurtosis_coef_D[layer]):.3f}+-{np.std(kurtosis_coef_D[layer]):.3f}")
        df["layer"] = layer
        df["unitid"] = np.arange(len(df))
        df_all = pd.concat((df_all, df), axis=0)
    if layeralias is not None:
        df_all["layer_s"] = df_all.layer.apply(layeralias)
    return df_all, sparseness_coef_D


def calculate_invariance(Invfeatdata, popsize=64, subsample=False, reps=1, layeralias=None):
    if isinstance(layeralias, dict):
        layeralias = layeralias.__getitem__
        #lambda x: layermap_inv[x]
    unit_inv_cc_dict = {}
    pop_inv_cc_dict = {}
    df_inv_all = pd.DataFrame()
    df_inv_all_pop = pd.DataFrame()
    for layer in Invfeatdata:
        featmat = Invfeatdata[layer]  # 60 by Channel N
        feattsr = featmat.reshape(10, 6, -1).permute(2,1,0)  # 6 by 10 by Channel N
        # torch.corrcoef(feattsr[0, :, :]).mean()

        unit_cctsr = corrcoef_batch(feattsr)  # Chan, 6, 6
        unit_cctsr_msk = mask_diagonal(unit_cctsr)  # Chan, 6, 6
        invar_cc = unit_cctsr_msk.nanmean(dim=(1,2))  # Chan,
        unit_inv_cc_dict[layer] = invar_cc
        df = pd.DataFrame({"unit_inv": unit_inv_cc_dict[layer]})
        df["layer"] = layer
        df["unitid"] = np.arange(len(df))
        df_inv_all = pd.concat((df_inv_all, df), axis=0)

        if subsample:
            pop_invar_col = []
            for i in range(reps):
                mask = np.random.choice(featmat.shape[1], popsize, replace=False)
                popfeattsr = feattsr[mask, :, :]
                pop_cctsr = corrcoef_batch(popfeattsr.permute([2, 1, 0]))  # 10, 6, 6
                pop_cctsr_msk = mask_diagonal(pop_cctsr)  # 10, 6, 6
                pop_invar_trial = pop_cctsr_msk.nanmean(dim=(1, 2))  # 10,
                pop_invar_col.append(pop_invar_trial)
            pop_invar_cc = torch.stack(pop_invar_col).mean(dim=0)
        else:
            popfeattsr = feattsr
            # correlation of  responses to 10 objects across 6 transformations
            # correlation of population representations across 6 transformations
            pop_cctsr = corrcoef_batch(popfeattsr.permute([2, 1, 0]))  # 10, 6, 6
            pop_cctsr_msk = mask_diagonal(pop_cctsr)  # 10, 6, 6
            pop_invar_cc = pop_cctsr_msk.nanmean(dim=(1, 2))  # 10,

        pop_inv_cc_dict[layer] = pop_invar_cc
        df_obj = pd.DataFrame({"pop_inv": pop_inv_cc_dict[layer]})
        df_obj["layer"] = layer
        df_obj["objid"] = np.arange(len(pop_invar_cc))
        df_inv_all_pop = pd.concat((df_inv_all_pop, df_obj), axis=0)
        print(f"{layer} unit invariance {torch.nanmean(invar_cc):.3f}+-{np.nanstd(invar_cc):.3f}\t "
            f"object invariance {torch.nanmean(pop_invar_cc):.3f}+-{np.nanstd(pop_invar_cc):.3f}")
    if layeralias is not None:
        df_inv_all["layer_s"] = df_inv_all.layer.apply(layeralias)
        df_inv_all_pop["layer_s"] = df_inv_all_pop.layer.apply(layeralias)
    return df_inv_all, df_inv_all_pop
#%%
def calculate_percentile(INet_feattsrs, inv_feattsrs, layeralias=None):
    """Caluclate the percentile of invariance responses in the distribution of all responses"""
    if isinstance(layeralias, dict):
        layeralias = layeralias.__getitem__
    df_prct_all = None
    # for layer_s, layer_x in layermap.items():
    for layer_x in INet_feattsrs:
        df_prct = []
        for unitid in tqdm(range(inv_feattsrs[layer_x].shape[1])):
            inv_resps = inv_feattsrs[layer_x][:, unitid]
            INet_resps = INet_feattsrs[layer_x][:, unitid]
            inv_resp_np = inv_resps.numpy()
            INet_resp_np = INet_resps.numpy()
            INet_idx = np.argsort(INet_resp_np)
            rank_prct = np.searchsorted(INet_resp_np, inv_resp_np, sorter=INet_idx) / len(INet_resp_np)
            top100_resp = INet_resp_np[INet_idx[-100]]
            top_norm_inv_resp = inv_resp_np / top100_resp
            df_prct.append({"layer_x": layer_x,
                             "layer_s": layeralias(layer_x),
                             "unitid": unitid,
                             "prct_mean": rank_prct.mean(),
                             "prct_std": rank_prct.std(),
                             "prct_max": rank_prct.max(),
                             "prct_min": rank_prct.min(),
                             "top100_resp": top100_resp,
                             "inv_resp_norm_mean": top_norm_inv_resp.mean(),
                             "inv_resp_norm_std": top_norm_inv_resp.std(),
                             "inv_resp_norm_max": top_norm_inv_resp.max(),})
        df_prct = pd.DataFrame(df_prct)
        inv_zero_ratio = inv_feattsrs[layer_x].count_nonzero(dim=0) / inv_feattsrs[layer_x].shape[0]
        df_prct["inv_zero_ratio"] = 1 - inv_zero_ratio
        df_prct_all = pd.concat([df_prct_all, df_prct]) if df_prct_all is not None else df_prct
    return df_prct_all
    # df_prct_all.to_csv(join(sumdir, f"{netname}_inv_res p_prctl.csv"))

#%%
