import os
import re
import pandas as pd
from os.path import join
from glob import glob
from imageio import imread, imsave
from os.path import join
from build_montages import crop_from_montage
from PIL import Image
from build_montages import make_grid_np
sumdir = r"E:\insilico_exps\Cosine_insilico\summary"
expdir = r"E:\insilico_exps\Cosine_insilico\rec_objects-familiar-07_rsz"
expdir = r"E:\insilico_exps\Cosine_insilico\rec_objects-familiar-13_rsz"
cos_root = r"E:\insilico_exps\Cosine_insilico"
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
exptab, proto_list, target_label = extract_bestimgs(join(cos_root,"rec_objects-familiar-11_rsz"))
layer_uniq = exptab.layer.unique()
popsize_uniq = exptab.popsize.unique()
popsize_uniq.sort()
obj_uniq = exptab.score_method.unique()
#%%
for poplayer in layer_uniq[:]:#= '.layer4.Bottleneck2'
    mtg_img_col = []
    for popsize in popsize_uniq:
        for method in obj_uniq:
            msk = (exptab.layer == poplayer) & (exptab.score_method == method) & (exptab.popsize == popsize)
            mtg_img_col.extend([proto_list[i] for i in exptab[msk].index])

    mtg = make_grid_np(mtg_img_col, nrow=12, rowfirst=True)
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
