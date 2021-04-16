# Inner Feature Comparison 
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from time import time
import os, re
from os.path import join, exists
from easydict import EasyDict
from imageio import imsave
from skimage.transform import resize
import matplotlib.pylab as plt
dataroot = r"E:\Cluster_Backup\manif_allchan"
sumdir = r"E:\Cluster_Backup\manif_allchan\summary"
#%%
from torch_utils import save_imgrid, show_imgrid
from GAN_utils import upconvGAN
G = upconvGAN().cuda()
#%%
def to_uint8(img):
    return (255 * img).astype("uint8")
#%% Load and Export the prototypes
dataroot = r"E:\Cluster_Backup\manif_allchan"
sumdir = r"E:\Cluster_Backup\manif_allchan\summary"
os.makedirs(sumdir, exist_ok=True)
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57, True), # last entry signify if we do RF resizing or not.
            ("resnet50", ".layer1.Bottleneck1", 5, 28, 28, True),
            ("resnet50", ".layer2.Bottleneck0", 5, 14, 14, True),
            ("resnet50", ".layer2.Bottleneck2", 5, 14, 14, True),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck4", 5, 7, 7, True),
            ("resnet50", ".layer4.Bottleneck0", 5, 4, 4, False),
            ("resnet50", ".layer4.Bottleneck2", 5, 4, 4, False),
            ("resnet50", ".Linearfc", 5, False), ]

netname = "resnet50"  # "resnet50"
GANname = ""
for unit in tqdm(unit_list[:]):
    layer = unit[1]
    layerdir = "%s_%s_manifold-%s" % (netname, layer, GANname)
    figdir = join(dataroot, "prototypes", layerdir)
    os.makedirs(figdir, exist_ok=True)
    RFfit = unit[-1]
    code_col = []
    norm_col = []
    evoact_col = []
    suffix = "rf_fit" if RFfit else "original"
    npzfns = glob(join(dataroot, layerdir, "*.npz"))
    pngfns = glob(join(dataroot, layerdir, "Manifold_summary*.png"))
    if len(unit) == 6:
        pattern = re.compile("Evolution_codes_%s_(\d*)_%d_%d_%s.npz"%(layer, unit[3], unit[4], suffix))
        Alterpattern = re.compile("Manifold_set_%s_(\d*)_%d_%d_%s.npz"%(layer, unit[3], unit[4], suffix))
        PNGpattern = re.compile("Manifold_summary_%s_(\d*)_%d_%d_%s_norm(\d*)"%(layer, unit[3], unit[4], suffix))
    else:  # Evolution_codes_.Linearfc_792_original.npz
        pattern = re.compile("Evolution_codes_%s_(\d*)_%s.npz"%(layer, suffix))
        Alterpattern = re.compile("Manifold_set_%s_(\d*)_%s.npz"%(layer, suffix))
        PNGpattern = re.compile("Manifold_summary_%s_(\d*)_%s_norm(\d*)"%(layer, suffix))
    matchpatt = [pattern.findall(fn) for fn in npzfns]
    iChlist = [int(mat[0]) for mat in matchpatt if len(mat)==1]
    if len(iChlist)==0:
        oldform = True
        matchpatt = [Alterpattern.findall(fn) for fn in npzfns]
        iChlist = [int(mat[0]) for mat in matchpatt if len(mat) == 1]
        print("Found %d units in %s - %s layer with old patterns!" % (len(iChlist), netname, layer))
        norm_matchs = [PNGpattern.findall(fn) for fn in pngfns]
        normdict = {int(match[0][0]): int(match[0][1]) for match in norm_matchs if len(match) == 1}
    else:
        oldform = False
        print("Found %d units in %s - %s layer with new patterns!" % (len(iChlist), netname, layer))
    iChlist = sorted(iChlist)
    for iCh in tqdm(iChlist):
        if len(unit) == 6:
            unit_lab = "%s_%d_%d_%d" % (layer, iCh, unit[3], unit[4])
        elif len(unit) == 4:
            unit_lab = "%s_%d" % (layer, iCh,)
        else:
            raise ValueError("Unit tuple length not correct.")
        if oldform:
            fn = "Manifold_set_%s_%s.npz" % (unit_lab, suffix)
            Mdata = np.load(join(dataroot, layerdir, fn))
            spherenorm = normdict[iCh]
            corner = Mdata['corner']
            imgsize = Mdata['imgsize']
            PC1vec = Mdata['Perturb_vec'][0:1, :]
            latentcode = PC1vec * spherenorm
            img = G.render(latentcode)[0]
            lastgenact = Mdata['evol_score'][Mdata['evol_gen'] == Mdata['evol_gen'].max()]
            evoact = lastgenact.mean()
        else:
            fn = "Manifold_set_%s_%s.npz" % (unit_lab, suffix)
            Mdata = np.load(join(dataroot, layerdir, fn))
            Efn = "Evolution_codes_%s_%s.npz" % (unit_lab, suffix)
            Edata = np.load(join(dataroot, layerdir, Efn))
            spherenorm = Mdata['sphere_norm']
            corner = Mdata['corner']
            imgsize = Mdata['imgsize']
            PC1vec = Mdata['Perturb_vec'][0:1, :]
            latentcode = PC1vec * spherenorm
            # latentcode = Edata['best_code']
            evoact = Edata['lastgen_score']
            img = G.render(latentcode)[0]
            # raise NotImplementedError
        code_col.append(latentcode)
        norm_col.append(spherenorm)
        evoact_col.append(evoact)
        if RFfit:
            imsave(join(figdir, "proto_%s_%s_full.png" % (unit_lab, suffix)), to_uint8(img))
            imrsz = resize(img, imgsize)
            imsave(join(figdir, "proto_%s_%s.png" % (unit_lab, suffix)), to_uint8(imrsz))
        else:
            imsave(join(figdir, "proto_%s_%s.png"%(unit_lab, suffix)), to_uint8(img))
    code_arr = np.array(code_col)
    norm_arr = np.array(norm_col, dtype=np.float64)
    chan_arr = np.array(iChlist)
    evoact_arr = np.array(evoact_col)
    np.savez(join(sumdir, "%s_%s_%s_protostat.npz"%(netname, layer, suffix)), code_arr=code_arr,
             norm_arr=norm_arr,  chan_arr=chan_arr, evoact_arr=evoact_arr,
             corner=corner, imgsize=imgsize)
#%%
dataroot = r"E:\Cluster_Backup\manif_allchan"
sumdir = r"E:\Cluster_Backup\manif_allchan\summary"
def extract_prototype(unit_list, netname, G=G, GANname="", doMontage=True, imgN=49, RND=None):
    for unit in tqdm(unit_list[:]):
        layer = unit[1]
        layerdir = "%s_%s_manifold-%s" % (netname, layer, GANname)
        figdir = join(dataroot, "prototypes", layerdir)
        os.makedirs(figdir, exist_ok=True)
        RFfit = unit[-1]
        code_col = []
        norm_col = []
        evoact_col = []
        suffix = "rf_fit" if RFfit else "original"
        npzfns = glob(join(dataroot, layerdir, "*.npz"))
        pngfns = glob(join(dataroot, layerdir, "Manifold_summary*.png"))
        if len(unit) == 6:
            pattern = re.compile("Evolution_codes_%s_(\d*)_%d_%d_%s.npz" % (layer, unit[3], unit[4], suffix))
            Alterpattern = re.compile("Manifold_set_%s_(\d*)_%d_%d_%s.npz" % (layer, unit[3], unit[4], suffix))
            PNGpattern = re.compile("Manifold_summary_%s_(\d*)_%d_%d_%s_norm(\d*)" % (layer, unit[3], unit[4], suffix))
        else:  # Evolution_codes_.Linearfc_792_original.npz
            pattern = re.compile("Evolution_codes_%s_(\d*)_%s.npz" % (layer, suffix))
            Alterpattern = re.compile("Manifold_set_%s_(\d*)_%s.npz" % (layer, suffix))
            PNGpattern = re.compile("Manifold_summary_%s_(\d*)_%s_norm(\d*)" % (layer, suffix))
        matchpatt = [pattern.findall(fn) for fn in npzfns]
        iChlist_E = [int(mat[0]) for mat in matchpatt if len(mat) == 1]
        if len(iChlist_E) == 0:
            oldform = True
            matchpatt = [Alterpattern.findall(fn) for fn in npzfns]
            iChlist = [int(mat[0]) for mat in matchpatt if len(mat) == 1]
            print("Found %d units in %s - %s layer with old patterns!" % (len(iChlist), netname, layer))
            norm_matchs = [PNGpattern.findall(fn) for fn in pngfns]
            normdict = {int(match[0][0]): int(match[0][1]) for match in norm_matchs if len(match) == 1}
        else:
            oldform = False
            matchpatt = [Alterpattern.findall(fn) for fn in npzfns]
            iChlist_M = [int(mat[0]) for mat in matchpatt if len(mat) == 1]
            iChlist = set(iChlist_E).intersection(iChlist_M) # both E and M data is present
            print("Found %d units in %s - %s layer with new patterns!" % (len(iChlist), netname, layer))
        iChlist = sorted(iChlist)
        for iCh in tqdm(iChlist):
            if len(unit) == 6:
                unit_lab = "%s_%d_%d_%d" % (layer, iCh, unit[3], unit[4])
                unit_pypatt = "%s_%%d_%d_%d" % (layer, unit[3], unit[4])
            elif len(unit) == 4:
                unit_lab = "%s_%d" % (layer, iCh,)
                unit_pypatt = "%s_%%d" % (layer, )
            else:
                raise ValueError("Unit tuple length not correct.")
            if oldform:
                fn = "Manifold_set_%s_%s.npz" % (unit_lab, suffix)
                Mdata = np.load(join(dataroot, layerdir, fn))
                spherenorm = normdict[iCh]
                corner = Mdata['corner']
                imgsize = Mdata['imgsize']
                PC1vec = Mdata['Perturb_vec'][0:1, :]
                latentcode = PC1vec * spherenorm
                img = G.render(latentcode)[0]
                lastgenact = Mdata['evol_score'][Mdata['evol_gen'] == Mdata['evol_gen'].max()]
                evoact = lastgenact.mean()
            else:
                fn = "Manifold_set_%s_%s.npz" % (unit_lab, suffix)
                Mdata = np.load(join(dataroot, layerdir, fn))
                Efn = "Evolution_codes_%s_%s.npz" % (unit_lab, suffix)
                Edata = np.load(join(dataroot, layerdir, Efn))
                spherenorm = Mdata['sphere_norm']
                corner = Mdata['corner']
                imgsize = Mdata['imgsize']
                PC1vec = Mdata['Perturb_vec'][0:1, :]
                latentcode = PC1vec * spherenorm
                # latentcode = Edata['best_code']
                evoact = Edata['lastgen_score']
                img = G.render(latentcode)[0]
                # raise NotImplementedError
            code_col.append(latentcode)
            norm_col.append(spherenorm)
            evoact_col.append(evoact)
            if RFfit:
                imsave(join(figdir, "proto_%s_%s_full.png" % (unit_lab, suffix)), to_uint8(img))
                imrsz = resize(img, imgsize)
                imsave(join(figdir, "proto_%s_%s.png" % (unit_lab, suffix)), to_uint8(imrsz))
            else:
                imsave(join(figdir, "proto_%s_%s.png" % (unit_lab, suffix)), to_uint8(img))
        code_arr = np.array(code_col)
        norm_arr = np.array(norm_col, dtype=np.float64)
        chan_arr = np.array(iChlist)
        evoact_arr = np.array(evoact_col)
        np.savez(join(sumdir, "%s_%s_%s_protostat.npz" % (netname, layer, suffix)), code_arr=code_arr,
                 norm_arr=norm_arr, chan_arr=chan_arr, evoact_arr=evoact_arr,
                 corner=corner, imgsize=imgsize)
        if doMontage:
            RND = np.random.randint(1E3) if RND is None else RND
            thresh = max(np.percentile(evoact_arr, [0.5]), 0.01)
            evomask = evoact_arr > thresh
            iChlist = chan_arr[evomask]
            idx_list = np.random.choice(iChlist, imgN, replace=False)
            idx_list = sorted(idx_list)
            img_list = []
            for iCh in idx_list:
                img = plt.imread(join(figdir, "proto_%s_%s.png" % (unit_pypatt % iCh, suffix)))
                img_list.append(img)
            img_arr = np.transpose(np.array(img_list), [1, 2, 3, 0])
            imgmtg = make_grid_np(img_arr, nrow=nrow, padding=3)
            imsave(join(sumdir, "proto_montage_%s_%s_%03d.png" % (netname, layer, RND)), imgmtg)
            print("Image saved to %s"%join(sumdir, "proto_montage_%s_%s_%03d.png" % (netname, layer, RND)))
#%%
unit_list = [("Cornet_s", ".V1.ReLUnonlin1", 5, 57, 57, True),
            ("Cornet_s", ".V1.ReLUnonlin2", 5, 28, 28, True),
            ("Cornet_s", ".V2.Conv2dconv_input", 5, 28, 28, True),
            ("Cornet_s", ".CORblock_SV2", 5, 14, 14, True),
            ("Cornet_s", ".V4.Conv2dconv_input", 5, 14, 14, True),
            ("Cornet_s", ".CORblock_SV4", 5, 7, 7, True),
            ("Cornet_s", ".IT.Conv2dconv_input", 5, 7, 7, True),
            ("Cornet_s", ".CORblock_SIT", 5, 3, 3, True),
            ("Cornet_s", ".decoder.Linearlinear", 5, False), ]

extract_prototype(unit_list, "cornet_s", G=G, imgN=49)
#%%
unit_list = [#("vgg16", "conv2", 5, 112, 112, True),
            ("vgg16", "conv3", 5, 56, 56, True),
            ("vgg16", "conv4", 5, 56, 56, True),
            ("vgg16", "conv5", 5, 28, 28, True),
            ("vgg16", "conv6", 5, 28, 28, True),
            ("vgg16", "conv7", 5, 28, 28, True),
            ("vgg16", "conv9", 5, 14, 14, True),
            ("vgg16", "conv10", 5, 14, 14, True),
            ("vgg16", "conv12", 5, 7, 7, True),
            ("vgg16", "conv13", 5, 7, 7, True),]
            # ("vgg16", "fc1", 1, False),
            # ("vgg16", "fc2", 1, False),
            # ("vgg16", "fc3", 1, False), ]

# extract_prototype(unit_list, "vgg16", G=G, imgN=49)
extract_prototype(unit_list, "vgg16-face", G=G, imgN=49)
#%%
unit_list = [("densenet169", ".features.ReLUrelu0", 5, 57, 57, True),
             ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28, True),
             ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28, True),
             ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14, True),
             ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14, True),
             ("densenet169", ".features._DenseBlockdenseblock3", 5, 7, 7, False),
             ("densenet169", ".features.transition3.Conv2dconv", 5, 7, 7, False),
             ("densenet169", ".features._DenseBlockdenseblock4", 5, 3, 3, False),
             ("densenet169", ".Linearclassifier", 5, False), ]
extract_prototype(unit_list, "densenet169", G=G, imgN=49)
#%%
unit_list = [("alexnet", "conv1_relu", 5, 28, 28, True),
            ("alexnet", "conv2_relu", 5, 13, 13, True),
            ("alexnet", "conv3_relu", 5, 6, 6, True),
            ("alexnet", "conv4_relu", 5, 6, 6, True),
            ("alexnet", "conv5_relu", 5, 6, 6, True),
            ("alexnet", "fc6", 5, False),
            ("alexnet", "fc7", 5, False),
            ("alexnet", "fc8", 5, False), ]

extract_prototype(unit_list, "alexnet", G=G, imgN=49)
#%% Comparing the prototpye distribution of 2 architecturally matched network
from build_montages import build_montages, make_grid_np
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57, True), # last entry signify if we do RF resizing or not.
            ("resnet50", ".layer1.Bottleneck1", 5, 28, 28, True),
            ("resnet50", ".layer2.Bottleneck0", 5, 14, 14, True),
            ("resnet50", ".layer2.Bottleneck2", 5, 14, 14, True),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7, True),
            ("resnet50", ".layer3.Bottleneck4", 5, 7, 7, True),
            ("resnet50", ".layer4.Bottleneck0", 5, 4, 4, False),
            ("resnet50", ".layer4.Bottleneck2", 5, 4, 4, False),
            ("resnet50", ".Linearfc", 5, False), ]

paired = True
GANname = ""
nets = ["resnet50_linf_8", "resnet50", ]
for unit in unit_list:
    layer = unit[1] # layer = ".layer3.Bottleneck2"
    RFfit = unit[-1]
    suffix = "rf_fit" if RFfit else "original"
    idx_list = None
    RND = np.random.randint(1E3)
    imgN = 49
    nrow = int(np.ceil(np.sqrt(imgN)))
    plt.figure(figsize=[15, 7.2])
    for neti, netname in enumerate(nets):
        layerdir = "%s_%s_manifold-%s" % (netname, layer, GANname)
        figdir = join(dataroot, "prototypes", layerdir)
        npzpath = join(dataroot, "summary", "%s_%s_%s_protostat.npz"%(netname, layer, suffix))
        sumdata = np.load(npzpath)
        chan_arr = sumdata['chan_arr']
        thresh = max(np.percentile(sumdata['evoact_arr'], [0.5]), 0.01)
        evomask = sumdata['evoact_arr'] > thresh
        iChlist = chan_arr[evomask]
        if len(unit) == 6:
            # unit_lab = "%s_*_%d_%d" % (layer, unit[3], unit[4])
            # unit_patt = "%s_(\d*)_%d_%d" % (layer, unit[3], unit[4])
            unit_pypatt = "%s_%%d_%d_%d" % (layer, unit[3], unit[4])
        elif len(unit) == 4:
            # unit_lab = "%s_*" % (layer, )
            # unit_patt = "%s_(\d*)" % (layer, )
            unit_pypatt = "%s_%%d" % (layer, )
        # imglist = glob(join(figdir, "proto_%s_%s.png"%(unit_lab, suffix)))
        # protopatt = re.compile(unit_patt)
        # matches = [protopatt.findall(fn) for fn in imglist]
        # iChlist = sorted([int(mat[0]) for mat in matches if len(mat) == 1])
        if paired:
            idx_list = np.random.choice(iChlist, imgN, replace=False) \
                if idx_list is None else idx_list
        else:
            idx_list = np.random.choice(iChlist, imgN, replace=False)
        idx_list = sorted(idx_list)
        img_list = []
        for iCh in idx_list:
            img = plt.imread(join(figdir, "proto_%s_%s.png"%(unit_pypatt%iCh, suffix)))
            img_list.append(img)
        img_arr = np.transpose(np.array(img_list), [1, 2, 3, 0])
        imgmtg = make_grid_np(img_arr, nrow=nrow, padding=3)
        imsave(join(sumdir, "proto_montage_%s_%s_%03d.png"%(netname, layer, RND)), imgmtg)
        ax = plt.subplot(1, len(nets), neti+1)
        plt.imshow(imgmtg)
        plt.axis("off")
        plt.title(netname+"-"+layer)
    plt.savefig(join(sumdir, "proto_montage_%s_cmp_%s_%03d.png"%("-".join(nets), layer, RND)))
    plt.show()


