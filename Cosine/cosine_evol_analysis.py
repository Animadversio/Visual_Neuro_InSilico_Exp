import os
import re
import numpy as np
import pandas as pd
from os.path import join
from glob import glob
from imageio import imread, imsave
from build_montages import crop_from_montage
from PIL import Image
from tqdm import tqdm
from build_montages import make_grid_np
# from core.montage_utils import ToPILImage, make_grid, make_grid_np, show_tsrbatch, PIL_tsrbatch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sumdir = r"E:\insilico_exps\Cosine_insilico\summary"
expdir = r"E:\insilico_exps\Cosine_insilico\rec_objects-familiar-07_rsz"
expdir = r"E:\insilico_exps\Cosine_insilico\rec_objects-familiar-13_rsz"
cos_root = r"E:\insilico_exps\Cosine_insilico"
def get_recimgnames():
    subpaths = glob(join(cos_root, "rec*"))
    subdirs = [os.path.split(path)[1] for path in subpaths]
    subdir_nosfx = [dirnm[4:-4] if dirnm.endswith("_rsz") else dirnm[4:] for dirnm in subdirs]
    imgnames = set(subdir_nosfx)
    if 2 * len(imgnames) != len(subdirs):
        print("warning some images without suffix")
    return imgnames


def shorten_layer(layer):
    return layer.replace(".layer", "L").replace("Bottleneck", "Btn")


def extract_bestimgs(expdir):
    target_label = expdir.split("\\")[-1]
    lastgen_imgfns = glob(join(expdir, "lastgen_resnet50*.jpg"))
    imgfp_patt = re.compile("lastgen_resnet50-([^-]*)-(\d*)-(\d*)_(.*)_fc6_(\d*)_score([-\.\d]*).jpg")
    # lastgen_resnet50-.layer4.Bottleneck2-2048-0000_corr_fc6_61988_score0.7
    imgfp = lastgen_imgfns[0]
    exptab = []
    proto_list = []
    for imgfp in lastgen_imgfns:
        imgfn = os.path.basename(imgfp)
        elems = imgfp_patt.findall(imgfn)
        assert len(elems) == 1, imgfn
        layer, popsize, popseed, score_method, RND, score = elems[0]
        popsize, popseed, RND, score = int(popsize), int(popseed), int(RND), float(score)
        img = imread(imgfp)
        crops = crop_from_montage(img, imgid=(1, 0), imgsize=227, pad=2)
        # Image.fromarray(crops).show()
        exptab.append((layer, popsize, popseed, score_method, RND, score))
        proto_list.append(crops)

    exptab = pd.DataFrame(exptab, columns=["layer", "popsize", "popseed", "score_method", "RND", "score", ])
    return exptab, proto_list, target_label
#%%
targetnames = get_recimgnames()
#%%
# expdir = join(cos_root, "rec_objects-familiar-11_rsz")
# exptab, proto_list, target_label = extract_bestimgs(expdir)
# layer_uniq = exptab.layer.unique()
# popsize_uniq = exptab.popsize.unique()
# popsize_uniq.sort()
# obj_uniq = exptab.score_method.unique()
#%%
"""
For analysis there are 3 experimental variables
* Layer '.layer2.Bottleneck2', '.layer3.Bottleneck0',
       '.layer4.Bottleneck2'
* Size of population in a layer  [ 200,  500, 1024]
* Objective 'corr', 'cosine', 'dot', 'MSE'
* random runs 

 `exptab.groupby(['layer','popsize']).size()` 
"""
#%%
outdir = r"E:\insilico_exps\Cosine_insilico\summary\proto_summary"


#%%
REPS = 3
def merge_proto_print(exptab, proto_list, targetnm, show=False, show_target=True,
                      expdir="", objorder=None, suffix=""):
    popdict = exptab.groupby(['layer', 'popsize']).indices
    for poplayer, popsize in popdict:
        subdf = exptab.iloc[popdict[(poplayer, popsize)]]
        row = subdf.iloc[0]
        if show_target:
            targimg = imread(join(expdir,
                f"targetimg_resnet50-{poplayer}-{popsize}-{row.popseed:04d}_"
                f"{row.score_method}_fc6_{row.RND:05d}.png"))
        mtg_img_col = []
        for mi, method in enumerate(objorder):
            msk = (exptab.layer == poplayer) & \
                  (exptab.popsize == popsize) & \
                  (exptab.score_method == method)
            mtg_img_col.extend([proto_list[i] for i in exptab[msk].index])
            for i in range(len(exptab[msk].index), REPS):
                mtg_img_col.append(np.zeros_like(proto_list[0]))

            if show_target:
                mtg_img_col.append(targimg if mi == 0
                                   else np.zeros_like(proto_list[0]))

        mtg = make_grid_np(mtg_img_col, rowfirst=True,
                           nrow=REPS + 1 if show_target else REPS)
        if show: Image.fromarray(mtg).show()
        outlabel = f"{targetnm}-{shorten_layer(poplayer)}-pop{popsize}_optimobj_cmp{suffix}"
        plt.imsave(join(outdir, outlabel + ".jpg"), mtg, )

#%%
objorder = ['corr', 'cosine', 'dot', 'MSE']
exptab_col = []
# targetnm = "objects-familiar-11"
for targetnm in tqdm(list(targetnames)[108:]): #107 bug '06-06-08_sh'
    expdir = join(cos_root, f"rec_{targetnm}_rsz")
    exptab, proto_list, target_label = extract_bestimgs(expdir)
    exptab["target"] = targetnm
    merge_proto_print(exptab, proto_list, targetnm, expdir=expdir,
                      show=False, show_target=True, objorder=objorder)
    exptab_col.append(exptab)
#%%
for targetnm in tqdm(targetnames):
    expdir = join(cos_root, f"rec_{targetnm}")
    exptab, proto_list, target_label = extract_bestimgs(expdir)
    exptab["target"] = targetnm
    merge_proto_print(exptab, proto_list, targetnm, expdir=expdir,
                      show=False, show_target=True, objorder=objorder, suffix="_fullimg")
    exptab_col.append(exptab)

#%%
exptab_all = pd.concat(exptab_col, axis=0)
exptab_all.to_csv(join(sumdir, "exptab_merge.csv"))






#%%  Dev zone ... ?
# = '.layer4.Bottleneck2'
for poplayer in layer_uniq[:]:
    mtg_img_col = []
    for popsize in popsize_uniq:
        for method in obj_uniq:
            msk = (exptab.layer == poplayer) & (exptab.score_method == method) & (exptab.popsize == popsize)
            mtg_img_col.extend([proto_list[i] for i in exptab[msk].index])

    mtg = make_grid_np(mtg_img_col, nrow=3, rowfirst=True)
    Image.fromarray(mtg).show()
    Image.fromarray(mtg).save(join(sumdir, "resnet_%s_%s_recon_cmp_all.jpg"%(poplayer, target_label)))

    mtg_img_col_sel = []
    for popsize in popsize_uniq:
        for method in obj_uniq:
            msk = (exptab.layer == poplayer) & (exptab.score_method == method) & (exptab.popsize == popsize)
            if msk.sum() > 0:
                mtg_img_col_sel.append(proto_list[exptab[msk].index[0]])
            else:
                print(poplayer, popsize, method, "param combination doesn't exist! ")
    mtg2 = make_grid_np(mtg_img_col_sel, nrow=4, rowfirst=True)
    Image.fromarray(mtg2).save(join(sumdir, "resnet_%s_%s_recon_cmp_samp.jpg"%(poplayer, target_label)))
    Image.fromarray(mtg2).show()
