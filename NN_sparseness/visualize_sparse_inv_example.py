"""Create montage plot showing response distribution of ImageNet Validation
and to Invariance set images
"""
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from NN_sparseness.sparse_plot_utils import plot_invariance_tuning,\
    plot_resp_histogram, plot_prototype, plot_Manifold_maps, inv_imglist
from NN_sparseness.insilico_EM_data_utils import _load_proto_montage, \
    _load_proto_info, _load_proto_info_rf
from dataset_utils import create_imagenet_valid_dataset
from stats_utils import saveallforms
from build_montages import make_grid_np, make_grid
def summarize_statstr(unitrow, netname):
    layer = unitrow.layer_s
    layer_x = unitrow.layer_x
    unitid = unitrow.unitid
    sprs = unitrow.sparseness
    zeroratio = unitrow.zero_ratio
    unitinv = unitrow.unit_inv
    statstr = f"{netname} {layer} Unit {unitid} Invariance {unitinv:.2f}\nSparseness {sprs:.2f} Zero ratio {zeroratio:.2f}"
    if "kappa" in unitrow:
        kappa = unitrow.kappa
        beta = unitrow.beta
        statstr += f"\nKappa {kappa:.2f} Beta {beta:.2f}"
    if "inv_resp_norm_mean" in unitrow:
        statstr += f"\nNorm Invar Resp Mean {unitrow.inv_resp_norm_mean:.2f} Max {unitrow.inv_resp_norm_max:.2f}"
    if "prct_mean" in unitrow:
        statstr += f"  Prctl Mean {unitrow.prct_mean:.2f} Max {unitrow.prct_max:.2f}"
    return statstr


def visualize_unit_data(unitrow, netname, Invfeatdata, feattsrs, INdataset):
    layer = unitrow.layer_s
    layer_x = unitrow.layer_x
    unitid = unitrow.unitid
    statstr = summarize_statstr(unitrow, netname)
    inv_resps = Invfeatdata[layer_x][:, unitid]
    INet_resps = feattsrs[layer_x][:, unitid]
    if netname == "resnet50_linf8":
        netname = "resnet50_linf_8"

    layerdir = join(proto_root, "prototypes", f"{netname}_{layer_x}_manifold-")
    layerfulldir = join(proto_root, f"{netname}_{layer_x}_manifold-")
    # protoimg, Edata, Mdata = _load_proto_info(unitrow, layerdir, layerfulldir)
    protoimg, protoimg_rffit, protoimg_padded, Edata, Mdata, layercfg = \
             _load_proto_info_rf(unitrow, layerdir, layerfulldir, )
    evollastgen = Edata.evol_score[Edata.evol_gen == 99].mean()
    evolmax = Edata.evol_score.max()
    natimgtsr, _ = INdataset[INet_resps.argmax()]
    natimg = natimgtsr.permute(1, 2, 0).numpy()
    # %
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    plt.suptitle(statstr, fontsize=20)
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[1, 3])
    plot_resp_histogram(INet_resps, inv_resps, "", ax=ax1)
    plot_invariance_tuning(inv_resps, "", ax=ax2)
    # plot_prototype(protoimg, f"Last mean {evollastgen:.1f} Max {evolmax:.1f}", ax=ax3)
    plot_prototype(protoimg_padded, f"Last mean {evollastgen:.1f} Max {evolmax:.1f}", ax=ax3)
    plot_prototype(natimg, f"Score {INet_resps.max():.1f}", ax=ax4)
    plt.tight_layout()
    plt.show()
    return fig, protoimg, Edata, Mdata, natimg


def visualize_unit_data_montage(unitrow, netname, Invfeatdata,
                                feattsrs, INdataset, topk=4):
    layer = unitrow.layer_s
    layer_x = unitrow.layer_x
    unitid = unitrow.unitid
    statstr = summarize_statstr(unitrow, netname)
    inv_resps = Invfeatdata[layer_x][:, unitid]
    INet_resps = feattsrs[layer_x][:, unitid]
    if netname == "resnet50_linf8":
        netname = "resnet50_linf_8"

    layerdir = join(proto_root, "prototypes", f"{netname}_{layer_x}_manifold-")
    layerfulldir = join(proto_root, f"{netname}_{layer_x}_manifold-")
    protoimg, protoimg_rffit, protoimg_padded, Edata, Mdata, layercfg = \
        _load_proto_info_rf(unitrow, layerdir, layerfulldir, )
    evollastgen = Edata.evol_score[Edata.evol_gen == 99].mean()
    evolmax = Edata.evol_score.max()
    natimgtsr, _ = INdataset[INet_resps.argmax()]
    natimg = natimgtsr.permute(1, 2, 0).numpy()
    nrow = np.sqrt(4).round().astype(int)
    invresp_topk, invidx_topk = torch.topk(inv_resps, k=topk, largest=True)
    INresp_topk, INidx_topk = torch.topk(INet_resps, k=10, largest=True)
    invimgs = [plt.imread(inv_imglist[idx]) for idx in invidx_topk]
    invimgmtg = make_grid_np(invimgs, nrow=nrow)
    natimgtsrs = [INdataset[idx][0] for idx in INidx_topk]
    natimgmtgtsr = make_grid(natimgtsrs, nrow=nrow)
    natimgmtg = natimgmtgtsr.permute(1, 2, 0).numpy()
    # %
    fig = plt.figure(figsize=(15, 8), constrained_layout=False)
    plt.suptitle(statstr, fontsize=20)
    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[1, 3])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax8 = fig.add_subplot(gs[0:2, 4])
    plot_resp_histogram(INet_resps, inv_resps, "", ax=ax1)
    plot_invariance_tuning(inv_resps, "", ax=ax2)
    # plot_prototype(protoimg, f"Last mean {evollastgen:.1f} Max {evolmax:.1f}", ax=ax3)
    plot_prototype(protoimg_padded, f"Last mean {evollastgen:.1f} Max {evolmax:.1f}", ax=ax3)
    plot_prototype(natimg, f"Nat Score {INresp_topk.max():.1f}", ax=ax4)
    plot_prototype(invimgs[0], f"Inv {invresp_topk.max():.1f}", ax=ax6)
    plot_prototype(invimgmtg, f"inv {invresp_topk.max():.1f} - {invresp_topk.min():.1f} (mean {invresp_topk.mean():.1f})", ax=ax5)
    plot_prototype(natimgmtg, f"nat {INresp_topk.max():.1f} - {INresp_topk.min():.1f} (mean {INresp_topk.mean():.1f})", ax=ax8)
    plt.tight_layout()
    plt.show()
    return fig, protoimg, Edata, Mdata, natimg

proto_root = r"E:\Cluster_Backup\manif_allchan"
if __name__ == "__main__":
    #%%
    rootdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness"
    sumdir = join(rootdir, "summary")
    figdir = join(rootdir, "summary_figs")
    #%%
    netname = "resnet50_linf8"
    exampledir = join(rootdir, f"tuning_map_examples_{netname}")

    Invfeatdata = torch.load(join(rootdir, f"{netname}_invariance_feattsrs.pt"))
    feattsrs = torch.load(join(rootdir, f"{netname}_INvalid_feattsrs.pt"))
    INdataset = create_imagenet_valid_dataset(normalize=False)
    df_merge_all = pd.read_csv(join(sumdir, f"{netname}_sparse_invar_prctl_merge.csv"), index_col=0)
    #%%
    unitrow = df_merge_all[df_merge_all.layer_s != 'layer2.B3'].sample().iloc[0]
    # fig, protoimg, Edata, Mdata, natimg = visualize_unit_data(unitrow, netname, Invfeatdata, feattsrs, INdataset)
    fig, protoimg, Edata, Mdata, natimg = visualize_unit_data_montage(unitrow, netname, Invfeatdata, feattsrs, INdataset)

    #%% Select a unit to visualize
    unitrow = df_merge_all[df_merge_all.layer_s != 'layer2.B3'].sample().iloc[0]
    # fig, protoimg, Edata, Mdata, natimg = visualize_unit_data(unitrow, netname, Invfeatdata, feattsrs, INdataset)
    fig, protoimg, Edata, Mdata, natimg = visualize_unit_data_montage(unitrow, netname, Invfeatdata, feattsrs, INdataset)
    # saveallforms(exampledir, f"{netname}_{unitrow.layer_s}_Ch{unitrow.unitid}_summary", fig)
    # 'E:\\Cluster_Backup\\manif_allchan\\prototypes\\resnet50_linf_8_.Linearfc_manifold-\\proto_.Linearfc_888_original.png'
    #%%
    for layer in df_merge_all.layer_s.unique():
        if layer == 'layer2.B3': continue
        df_layer = df_merge_all[df_merge_all.layer_s == layer]
        for i in range(20):
            unitrow = df_layer.sample().iloc[0]
            # fig, protoimg, Edata, Mdata, natimg = visualize_unit_data(unitrow, netname, Invfeatdata, feattsrs, INdataset)
            fig, protoimg, Edata, Mdata, natimg = visualize_unit_data_montage(unitrow,
                      netname, Invfeatdata, feattsrs, INdataset)
            saveallforms(exampledir, f"{netname}_{layer}_Ch{unitrow.unitid}_RF_summary", fig)
    #%%
    layer = 'layer3.B4'
    df_layer = df_merge_all[df_merge_all.layer_s == layer]
    #%
    unitrow = df_layer[df_layer.inv_resp_norm_max > 1.2].sample().iloc[0]
    fig, protoimg, Edata, Mdata, natimg = visualize_unit_data_montage(unitrow, netname, Invfeatdata, feattsrs, INdataset)


    #%%
    df_merge_all.plot(x="layer_s", y="unit_inv", kind='bar', legend=True, )
    plt.show()
    #%%

    # sns.violinplot(x="layer_s", y="unit_inv", data=df_merge_all, legend=True)
    sns.stripplot(x="layer_s", y="unit_inv", data=df_merge_all, alpha=0.1, jitter=True,)
    sns.pointplot(x="layer_s", y="unit_inv", data=df_merge_all, legend=True, color="black")
    plt.xticks(rotation=30)
    plt.show()

    #%%
    df_merge_all.groupby("layer_s", sort=False)["unit_inv"].plot(kind='density', legend=True, )
    plt.show()
    #%%

    sns.stripplot(x="layer_s", y="pop_inv", data=df_inv_all_pop, alpha=0.7, jitter=True,)
    sns.pointplot(x="layer_s", y="pop_inv", data=df_inv_all_pop, legend=True, color="black")
    plt.xticks(rotation=30)
    plt.show()