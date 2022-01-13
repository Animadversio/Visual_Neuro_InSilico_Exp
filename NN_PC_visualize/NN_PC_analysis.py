#%%
"""
UMAP reduction of the feature vectors
"""
import os, umap
import numpy as np
import pandas as pd
# import datatable as dt
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from os.path import join
outdir = "H:\CNN-PCs"
netname = "resnet50_linf8"
#%%
feattsrs = torch.load(join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))
#%%
matrix = feattsrs['.layer3.Bottleneck2'].numpy()
#
feat_umap = umap.UMAP(random_state=999, n_neighbors=30, min_dist=.25)
# Fit UMAP and extract latent vars 1-2
embedding = pd.DataFrame(feat_umap.fit_transform(matrix), columns=['UMAP1','UMAP2'])
#%% Produce sns.scatterplot and pass metadata.subclasses as color
plt.figure(figsize=(7, 7))
sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
                # hue=metadata.subclass_label.to_list(),
                alpha=.4, linewidth=0, s=1)
# Adjust legend
# sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
# Save PNG
sns_plot.figure.savefig(join(outdir, "summary", 'umap_scatter_layer3-Btn2.png'), bbox_inches='tight', dpi=500)
plt.show()


#%%
""" 
 Most aligned Evolved patch  and 
 (Favorite) natural patches from the validation set
"""
from NN_PC_visualize.NN_PC_lib import load_featnet, create_imagenet_valid_dataset, \
    shorten_layername, unnormalize, normalize, get_cent_pos, get_RF_location
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from build_montages import crop_from_montage, make_grid_np
from tqdm import tqdm


def visualize_topK(dataset, score, K=16, corner=None, imgpix=None, descending=True, show=False):
    """Visualize top (or bottom) K patches from image dataset

        score: a vector / arr tensor the same length as dataset.
        descending: True for top K, False for bottom K.
        corner: corner coordinate, (left, top) (x, y)
        imgpix: scaler image pixel height / width.
        show: show the PIL image in window.
    """
    sortidx = torch.argsort(score, descending=descending)
    idxmax = sortidx[:K]
    img_col = []
    for idx in idxmax:
        img, _ = dataset[idx]
        img_col.append(unnormalize(img))

    imgtsr = torch.stack(img_col)
    mtgfull = ToPILImage()(make_grid(imgtsr, nrow=4))
    if show: mtgfull.show()
    if corner is not None and imgpix is not None:
        rfpatchtsr = imgtsr[:, :, corner[1]:corner[1] + imgpix, corner[0]:corner[0] + imgpix]
        mtgpatch = ToPILImage()(make_grid(rfpatchtsr, nrow=4))
        if show: mtgpatch.show()
        return mtgfull, mtgpatch
    else:
        return mtgfull


def find_evol_natpatch(dataset, feattsrs, tsr_svds, rfdict, reclayers, figdir, savedir):
    """Higher order wrapper to find out the image that has most similar feature vector to a
    direction and print out the image montage"""
    os.makedirs(savedir, exist_ok=True)
    for layeri in range(len(reclayers)):  # [2]: #
        layer = reclayers[layeri]
        print("Processing layer %s (%d/%d)"%(layer, layeri, len(reclayers)))
        feat_mean, U, S, V = tsr_svds[layer]
        feattsr = feattsrs[layer]
        # align_score = U
        align_score = (feattsr - feat_mean) @ V / (feattsr - feat_mean).norm(dim=1, keepdim=True)
        corner, imgpix = rfdict[layer]
        for iPC in tqdm(range(100)):
            for dir in ["pos", "neg"]:
                vecnm = "%s_%s_PC%03d_%s_cosine" % (netname, shorten_layername(layer), iPC, dir)
                filenm = "%s_%s_PC%03d_%s_cosine_dir_RFfit_norm_G_%s.npz" \
                         % (netname, shorten_layername(layer), iPC, dir, layer)
                evolnm = "%s_%s_PC%03d_%s_cosine_dir_RFfit_norm_G_%s%s.png" \
                         % (netname, shorten_layername(layer), iPC, dir, layer, "" if imgpix == 256 else "_RFpad")
                data = np.load(join(figdir, filenm))
                score_traj = data["score_traj"]
                finalscores = score_traj[-1, :]
                # ToPILImage()(torch.tensor(imgmtg).permute([2, 0, 1])).show()
                figh = plt.figure(figsize=(5, 3.5))
                plt.plot(score_traj)
                if dir == "pos":
                    topval, _ = torch.topk(align_score[:, iPC], 16)
                elif dir == "neg":
                    topval, _ = torch.topk(-align_score[:, iPC], 16)
                plt.scatter(100 * np.ones(16), topval.numpy(), s=25, alpha=0.4)
                plt.title(vecnm)
                plt.ylabel("cosine")
                plt.savefig(join(savedir, "%s_scoretraj.png" % vecnm))
                plt.close(figh)

                imgmtg = plt.imread(join(figdir, evolnm))
                evolpatchcol = [crop_from_montage(imgmtg, (0, i)) \
                                    [corner[1]:corner[1] + imgpix, corner[0]:corner[0] + imgpix, :] for i in range(8)]
                patchmtg = make_grid_np(evolpatchcol, nrow=4)
                plt.imsave(join(savedir, "%s_evolpatch.png" % vecnm), patchmtg)

                if dir == "pos":
                    mtgfull, mtgpatch = visualize_topK(dataset, align_score[:, iPC], K=16,
                                                       corner=corner, imgpix=imgpix, descending=True)
                elif dir == "neg":
                    mtgfull, mtgpatch = visualize_topK(dataset, -align_score[:, iPC], K=16,
                                                       corner=corner, imgpix=imgpix, descending=True)

                mtgpatch.save(join(savedir, "%s_natrpatch.png" % vecnm))


#%%
figdir = r"H:\CNN-PCs\RFfit_norm_lr_reldir_vis" # RFfit_norm_lr_vis
outdir = r"H:\CNN-PCs"
savedir = r"H:\CNN-PCs\natpatch_norm_lr_reldir_vis"

#%%
dataset = create_imagenet_valid_dataset()
netname = "resnet50_linf8"
feattsrs = torch.load(join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*feattsrs.keys()]
#%% Calculate RF
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
model.requires_grad_(False)
rfdict = {}
for layeri in range(3):
    layer = reclayers[layeri]
    cent_pos = get_cent_pos(model, reclayers[layeri], imgfullpix=256)
    corner, imgpix = get_RF_location(model, reclayers[layeri], cent_pos, imgfullpix=256)
    rfdict[layer] = (corner, imgpix)

#%%
for layeri in range(3):  # [2]: #
    layer = reclayers[layeri]
    feat_mean, U, S, V = tsr_svds[layer]
    feattsr = feattsrs[layer]
    # align_score = U
    align_score = (feattsr - feat_mean) @ V / (feattsr - feat_mean).norm(dim=1, keepdim=True)
    corner, imgpix = rfdict[layer]
    for iPC in range(100):
        for dir in ["pos", "neg"]:
            vecnm = "%s_%s_PC%03d_%s_cosine"% (netname, shorten_layername(layer), iPC, dir)
            filenm = "%s_%s_PC%03d_%s_cosine_dir_RFfit_norm_G_%s.npz" % (netname, shorten_layername(layer), iPC, dir, layer)
            evolnm = "%s_%s_PC%03d_%s_cosine_dir_RFfit_norm_G_%s%s.png" \
                 % (netname, shorten_layername(layer), iPC, dir, layer, "" if imgpix == 256 else "_RFpad")
            data = np.load(join(figdir, filenm))
            score_traj = data["score_traj"]
            finalscores = score_traj[-1, :]
            # ToPILImage()(torch.tensor(imgmtg).permute([2, 0, 1])).show()
            figh = plt.figure(figsize=(5, 3.5))
            plt.plot(score_traj)
            if dir == "pos":
                topval, _ = torch.topk(align_score[:, iPC], 16)
            elif dir == "neg":
                topval, _ = torch.topk(-align_score[:, iPC], 16)
            plt.scatter(100*np.ones(16), topval.numpy(), s=25, alpha=0.4)
            plt.title(vecnm)
            plt.ylabel("cosine")
            plt.savefig(join(savedir, "%s_scoretraj.png"%vecnm))
            plt.close(figh)

            imgmtg = plt.imread(join(figdir, evolnm))
            evolpatchcol = [crop_from_montage(imgmtg, (0, i))\
                                [corner[1]:corner[1] + imgpix, corner[0]:corner[0] + imgpix, :] for i in range(8)]
            patchmtg = make_grid_np(evolpatchcol, nrow=4)
            plt.imsave(join(savedir, "%s_evolpatch.png"%vecnm), patchmtg)

            if dir == "pos":
                mtgfull, mtgpatch = visualize_topK(dataset, align_score[:, iPC], K=16,
                                corner=corner, imgpix=imgpix, descending=True)
            elif dir == "neg":
                mtgfull, mtgpatch = visualize_topK(dataset, -align_score[:, iPC], K=16,
                                corner=corner, imgpix=imgpix, descending=True)

            mtgpatch.save(join(savedir, "%s_natrpatch.png"%vecnm))

#%% VGG16 test case.
outdir = r"H:\CNN-PCs"
figdir = r"H:\CNN-PCs\VGG16_RFfit_norm_lr_reldir_vis" # RFfit_norm_lr_vis
savedir = r"H:\CNN-PCs\VGG16_natpatch_norm_lr_reldir_vis"
#%%
dataset = create_imagenet_valid_dataset()
netname = "vgg16"
feattsrs = torch.load(join(outdir, "%s_INvalid_feattsrs.pt"%(netname)))
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*feattsrs.keys()]
#%% Calculate RF
_, model = load_featnet("vgg16")
model.eval().cuda()
model.requires_grad_(False)
rfdict = {}
for layeri in range(len(reclayers)):
    layer = reclayers[layeri]
    try:
        cent_pos = get_cent_pos(model, reclayers[layeri], imgfullpix=256)
        corner, imgpix = get_RF_location(model, reclayers[layeri], cent_pos, imgfullpix=256)
    except NotImplementedError:
        corner = (0, 0)
        imgpix = 256
    rfdict[layer] = (corner, imgpix)
#%%
find_evol_natpatch(dataset, feattsrs, tsr_svds, rfdict, reclayers[1:], figdir, savedir)
#%%
sortidx = torch.argsort(U[:, iPC], descending=True)
idxmax = sortidx[:16]
img_col = []
for idx in idxmax:
    img, _ = dataset[idx]
    img_col.append(unnormalize(img))

imgtsr = torch.stack(img_col)
rfpatchtsr = imgtsr[:, :, corner[1]:corner[1]+imgpix, corner[0]:corner[0]+imgpix]
ToPILImage()(make_grid(imgtsr, nrow=4)).show()
ToPILImage()(make_grid(rfpatchtsr, nrow=4)).show()
#%%  Average aligned evolved patch



