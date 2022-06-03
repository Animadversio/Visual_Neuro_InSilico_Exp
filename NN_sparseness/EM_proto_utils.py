"""
Utils for finding corresponding Evol, Manif, Prototype in the
previous data stores.
Sort out filenames and paths
"""

import pandas as pd
from build_montages import make_grid_np, build_montages
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict


def _load_proto_montage(tab, layerdir, ):
    """
    Load set of prototype images in a layer according `tab`
    Example:
        tab = df_kappa_merge[msk].nsmallest(5, "unit_inv")
        mtg_inv_min, protos_inv_min = _load_proto_montage(tab, layerdir)
        plt.imsave(join(outdir, f"{netname}_{layer}_montage_inv_min.png"), mtg_inv_min)

    :param tab:
    :param layerdir: str
    :return:
        prototype image Montage (np.array)
        list of prototype images (list)
    """
    if isinstance(tab, pd.DataFrame):
        layer, unitid = tab.layer_s.iloc[0], tab.unitid
    elif isinstance(tab, pd.Series):
        layer, unitid = tab.layer_s, [tab.unitid]
    else:
        raise ValueError("tab must be a pandas.DataFrame or pandas.Series")
    if "fc" in layer:
        suffix = "original"
    else:
        suffix = "rf_fit_full"
    imgcol = []
    filenametemplate = glob(join(layerdir, f"*_{suffix}.png"))[0]
    unitpos = filenametemplate.split("\\")[-1].split("_")[3:5]
    for unit in unitid:
        if "fc" in layer:
            img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{suffix}.png"))
        else:
            img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.png"))
        # img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_rf_fit.png"))
        imgcol.append(img)
    return make_grid_np(imgcol, nrow=5), imgcol


def _load_proto_info(tabrow, layerdir, layerfulldir):
    """
    Example:
        proto_dir = r"E:\Cluster_Backup\manif_allchan\prototypes"
        layerdir = join(proto_dir, f"vgg16_{layer}_manifold-")
        layerfulldir = join(r"E:\Cluster_Backup\manif_allchan", f"vgg16_{layer}_manifold-")
        protoimg, Edata, Mdata = _load_proto_info(unitrow, layerdir, layerfulldir)

    :param tabrow:
    :param layerdir:
    :param layerfulldir:
    :return:
        prototype image (np),
        Evol data (easydict),
        Manif data (numpy array 4x21x21)
    """
    if isinstance(tabrow, pd.Series):
        layer, unitid = tabrow.layer_s, tabrow.unitid
    elif isinstance(tabrow, pd.DataFrame):
        layer, unitid = tabrow.layer_s.iloc[0], tabrow.unitid[0]
    else:
        raise ValueError("tab must be a pandas.DataFrame or pandas.Series")
    if "fc" in layer:
        suffix = "original"
    else:
        suffix = "rf_fit"
    filenametemplate = glob(join(layerdir, f"*_{suffix}.png"))[0]
    unitpos = filenametemplate.split("\\")[-1].split("_")[3:5]
    unit = unitid
    if "fc" in layer:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{suffix}.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{suffix}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{suffix}.npy"))
    else:
        img = plt.imread(join(layerdir, f"proto_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}_full.png"))
        Edata = np.load(join(layerfulldir, f"Manifold_set_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.npz"))
        Mdata = np.load(join(layerfulldir, f"Manifold_score_{layer}_{unit:d}_{unitpos[0]}_{unitpos[1]}_{suffix}.npy"))
    return img, edict(Edata), Mdata
