import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from load_neural_data import ExpData
from scipy.io import loadmat
from scipy.stats import sem
rootdir = r"E:\OneDrive - Harvard University\CMA_localize"
matpath = r"E:\OneDrive - Harvard University\CMA_localize\preMeta.mat"
metatab = pd.read_csv(join(rootdir, "metatab.csv"))
#%%
Python_dir = r"D:\Github"
sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"Visual_Neuron_Modelling"))
#%%
# from torchvision import fit_models
from CorrFeatTsr_lib import Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, visualize_cctsr
from CorrFeatTsr_predict_lib import score_images, softplus, fitnl_predscore
from featvis_lib import rectify_tsr, tsr_factorize, vis_featmap_corr, vis_feattsr, \
    vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, vis_featvec_point, load_featnet, \
    tsr_posneg_factorize, posneg_sep, visualize_cctsr_simple


def visualize_evolution(expdata, scorevec_thread, threadid=1):
    blockvec_thread = expdata.generations[expdata.gen_rows]# np.array([int(imgpatt.findall(imgfn)[0]) for imgfn in imgfp_thread])
    blockarr = range(min(blockvec_thread), max(blockvec_thread)+1)
    meanarr = np.array([np.mean(scorevec_thread[blockvec_thread==blocki]) for blocki in blockarr])
    semarr = np.array([sem(scorevec_thread[blockvec_thread==blocki]) for blocki in blockarr])

    figh = plt.figure(figsize=[6,5]);
    plt.scatter(blockvec_thread, scorevec_thread,alpha=0.4)
    plt.plot(blockarr, meanarr, 'k-')
    plt.fill_between(blockarr, meanarr-semarr, meanarr+semarr,alpha=0.4)
    plt.ylabel("Spike rate");
    plt.xlabel("Generations");
    plt.title("Evolution Trajectory %s\nprefchan %02d (unit %d), %.1f deg pos [%.1f %.1f], thread %d"%\
              (expdata.ephysFN, expdata.pref_chan, expdata.pref_unit,
               expdata.imgsize_deg,
               expdata.imgpos[0], expdata.imgpos[1], threadid))
    plt.show()
    return figh


def corr_feat_factorize(expdata, scorevec_thread, imgfp_thread,
                        backup_dir, savedir, threadid=1):
    if isinstance(imgfp_thread, list):
        imgfp_thread = np.array(imgfp_thread)

    exptime = backup_dir.split("\\")[-1]
    netname = "resnet50_linf8"
    Animal = expdata.Animal
    layers2plot = ["layer2", "layer3", "layer4", ]

    layer = "layer3";
    bdr = 1;
    vis_score_mode = "cosine"  # "corr"
    NF = 3;
    rect_mode = "Tthresh";
    thresh = (None, 3)  # "pos"

    # savedir = savedir # join(backup_dir, "CCFactor_%s%s" % (netname, threadlabel))
    os.makedirs(join(savedir, "img"), exist_ok=True)
    featnet, net = load_featnet(netname)

    score_idx = np.argsort(-scorevec_thread)
    score_examp = scorevec_thread[score_idx[:5]]
    imgfp_examp = imgfp_thread[score_idx[:5]]
    imgcol_examp = [imread(fp) for fp in imgfp_examp]
    for i, img in enumerate(imgcol_examp):
        imgid = imgfp_examp[i].split("\\")[-1].split(".")[0]
        imsave(join(savedir, "img", "evol_best_%02d_%s.png" % (i, imgid)), img)

    imgpix = int(expdata.imgsize_deg * 40)  # %224  #
    # titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(net, layers2plot, netname=netname, )
    featFetcher.init_corr()
    Corr_Feat_pipeline(featnet, featFetcher, scorevec_thread, imgfp_thread,
                       lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
                       batchsize=100, savedir=savedir, savenm="Evol")  # % (Animal, Expi, expsuffix),
    featFetcher.clear_hook()
    corrdict = np.load(join(savedir, "%s_corrTsr.npz" % ("Evol")), allow_pickle=True)

    figh = visualize_cctsr_simple(featFetcher, layers2plot, imgcol_examp[:4],
                 savestr="%s_Evol%s_%s" % (Animal, exptime, netname),
                 titstr="%s_Evol%s_%s" % (Animal, exptime, netname),
                 figdir=savedir)

    cctsr_dict = corrdict.get("cctsr").item()
    Ttsr_dict = corrdict.get("Ttsr").item()
    stdtsr_dict = corrdict.get("featStd").item()

    Ttsr = Ttsr_dict[layer]
    cctsr = cctsr_dict[layer]
    stdtsr = stdtsr_dict[layer]
    covtsr = cctsr * stdtsr
    covtsr = np.nan_to_num(covtsr)
    cctsr = np.nan_to_num(cctsr)
    # Ttsr_pp = rectify_tsr(Ttsr, rect_mode)  #
    covtsr_pp = rectify_tsr(covtsr, mode=rect_mode, thr=thresh, Ttsr=Ttsr)  # add thresholding to T tsr
    # Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=ccdir, savestr="%s-%s"%(netname, layer))
    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(covtsr_pp, bdr=bdr, Nfactor=NF,
                                               figdir=savedir, savestr="%s-%s" % (netname, layer))
    Tcomponents = None

    print("Saving record for the factorization method")
    np.savez(join(savedir, "factor_record.npz"), Hmat=Hmat, Hmaps=Hmaps, Tcomponents=Tcomponents, ccfactor=ccfactor,
             netname=netname, layer=layer, bdr=bdr, NF=NF, rect_mode=rect_mode,
             vis_score_mode=vis_score_mode)
    factdict = np.load(join(savedir, "factor_record.npz"))
    # G = upconvGAN("fc6")
    # G.requires_grad_(False).cuda().eval();
    return corrdict, factdict

#%% Load in evolution data

# Expi == 1 beto-190611b image incomplete.  block001_gen_gen000_000002
# Expi == 8 Solved
# Expi == 9 beto-190611b Unit number in correct
#           'monkey64chan-21062019-007' need to re-sort.
# Expi == 57 2019-06-Evolutions\\beto-190802c\\backup_08_02_2019_14_29_14
#            block049_thread000_gen_gen048_001936 is missing... seems to be moved
# Expi == 93  image block042_thread000_gen_gen041_001656  is missing

for Expi in range(len(metatab)): #range(118, len(metatab)):
    if Expi == 9:
        continue
    ephysFN = metatab.ephysFN[Expi]
    stimpath = metatab.stimuli[Expi].strip()

    expdata = ExpData(ephysFN, stimpath)
    expdata.load_mat()
    expdata.find_generated()
    # preferred unit
    rasters = expdata.rasters[:]
    pref_ch_idx = (expdata.spikeID.squeeze() == expdata.pref_chan).nonzero()[0]
    pref_ch_idx = pref_ch_idx[expdata.pref_unit - 1]
    meanfr = rasters[:, 50:200, pref_ch_idx].mean(axis=(0, 1))
    if meanfr < 3:
        pref_ch_idx = (expdata.spikeID.squeeze() == expdata.pref_chan).nonzero()[0]
        pref_ch_idx = pref_ch_idx[expdata.pref_unit - 1 + 1]
        print("excluding the inactive zero unit")
    #%%
    # imgfp_glob = glob(stimpath + "\\*")
    # imgnm_thread = expdata.imgnms[expdata.gen_rows]
    # imgfp_thread = [join(stimpath, nm+".jpg") for nm in imgnm_thread]
    # try:
    #     for fp in imgfp_thread:
    #         assert os.path.exists(fp)
    # except AssertionError:
    #     imgfp_thread = []
    #     for nm in imgnm_thread:
    #         fullname = [fp for fp in imgfp_glob if nm in fp]
    #         assert len(fullname) == 1, "0 or more image matching the name %s"%nm
    #         imgfp_thread.append(fullname[0])
    #     # imgfp_thread = [[fp for fp in imgfp_glob if nm in fp][0] for nm in imgnm_thread]

    #%%
    evokevect = rasters[:, 50:200, pref_ch_idx].mean(axis=1)
    bslvect = rasters[:, 0:40, pref_ch_idx].mean(axis=1)
    bslmean = bslvect.mean()
    bslstd = bslvect.std()
    scorevec_thread = evokevect[expdata.gen_rows] - bslmean

    savedir = join(rootdir, expdata.ephysFN)
    os.makedirs(savedir, exist_ok=True)
    figh = visualize_evolution(expdata, scorevec_thread)
    figh.savefig(join(savedir, "evolution_traj.png"))
    #%%
    # CorrDict, FactDict = corr_feat_factorize(expdata, scorevec_thread, imgfp_thread, stimpath, savedir, threadid=1)
#%%
#%% Create masks for selected exps
import PIL
targetdir = r"N:\Users\Katie\binary_classifier\images\testing\V1\example_prototypes"
# targetdir = r"N:\Users\Katie\binary_classifier\images\testing\IT\example_prototypes"
import re
from glob import glob
from easydict import EasyDict
from skimage.transform import resize
from PIL import Image
from featvis_lib import pad_factor_prod
from shutil import copy

imglist = glob(targetdir + r"\*")

outdir = r"N:\Data-Computational\Project_CMA_Masks"
# Expilist = [  16,  50,  52, 104, 105, 106, 108, 119, 123, 129]
Expilist = [i for i in range(1, 120) if i!=10]#,   9,  35,  49,  61,  79,  92, 107, 127, 134, 135]
for Expi in Expilist:
    ephysFN = metatab.ephysFN[Expi - 1]
    stimpath = metatab.stimuli[Expi - 1].strip()
    fdrnm = stimpath.split("\\")[-1]
    savedir = join(rootdir, ephysFN)
    protodir = join(rootdir, ephysFN, "img")
    try:
        # img = imread(join(savedir, "Beto_Evol%s_resnet50_linf8_corrTsr_vis.png"%fdrnm))
        img = PIL.Image.open(join(savedir, "Beto_Evol%s_resnet50_linf8_corrTsr_vis.png"%fdrnm))
        # img.show()
        ccdata = EasyDict(np.load(join(savedir, "Evol_corrTsr.npz"), allow_pickle=True))
        data = EasyDict(np.load(join(savedir, "factor_record.npz"), allow_pickle=True))
        Hmaps = data["Hmaps"]
        bdr = data.bdr
        padded_Hmaps = np.pad(Hmaps[:, :, :],
              ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        facttsr = pad_factor_prod(Hmaps, data.ccfactor, bdr=data.bdr)
        factnorm = np.linalg.norm(facttsr, axis=0)
        Hmap_merge = factnorm
        # Hmap_merge = padded_Hmaps.sum(axis=2)
        Hmap_norm = Hmap_merge / Hmap_merge.max()
        alphamsk_rsz = resize(Hmap_norm, [256, 256])
        plt.imshow(alphamsk_rsz)
        vmin, vmax = 0.2, 0.8
        alphamsk_clip = np.clip(alphamsk_rsz, vmin, vmax)#min(vmax, max(vmin, alphamsk_rsz));
        alphamsk_clip = (alphamsk_clip - vmin) / (vmax - vmin)
        # maskedimg = (double(bestimg). * alphamsk_clip);
        # Image(np.ones((256, 256, 3), ))
        maskmat = np.concatenate([np.zeros((256, 256, 3)), \
                                  1 - alphamsk_clip[:, :, np.newaxis]], axis=2)
        Image.fromarray((maskmat*255).astype("uint8"), mode='RGBA').\
            save(join(outdir, "Beto_exp%03d_mask.png"%Expi))
        plt.figure(figsize=[5, 5])
        plt.imshow(padded_Hmaps**2 / (padded_Hmaps**2).max())
        plt.colorbar()
        plt.savefig(join(outdir, "Beto_exp%03d_mask_nonmerge.png"%Expi))
        imglist = os.listdir(protodir)
        for nm in imglist:
            copy(join(protodir, nm), join(outdir, "Beto_exp%03d_"%(Expi) + nm))
    except FileNotFoundError as e:
        print(e.args)

#%% Regenrate masks for selected exps
import PIL
import re
from glob import glob
from easydict import EasyDict
from skimage.transform import resize
from PIL import Image
from featvis_lib import pad_factor_prod
from shutil import copy
rootdir = r"E:\OneDrive - Harvard University\CMA_localize"
outdir = r"N:\Data-Computational\Project_CMA_Masks_tune"
metatab = pd.read_csv(join(rootdir, "metatab.csv"))
Expilist = [16, 50, 52, 104, 105, 106, 108, 119, 123, 129]
# Expilist = [8, 9, 35, 49, 61, 79, 92, 107, 127, 134, 135]
for Expi in Expilist:
    ephysFN = metatab.ephysFN[Expi - 1]
    stimpath = metatab.stimuli[Expi - 1].strip()
    fdrnm = stimpath.split("\\")[-1]
    savedir = join(rootdir, ephysFN)
    protodir = join(rootdir, ephysFN, "img")
    try:
        # img = imread(join(savedir, "Beto_Evol%s_resnet50_linf8_corrTsr_vis.png"%fdrnm))
        img = PIL.Image.open(join(savedir, "Beto_Evol%s_resnet50_linf8_corrTsr_vis.png"%fdrnm))
        # img.show()
        ccdata = EasyDict(np.load(join(savedir, "Evol_corrTsr.npz"), allow_pickle=True))

        data = EasyDict(np.load(join(savedir, "factor_record.npz"), allow_pickle=True))
        Hmaps = data["Hmaps"]
        bdr = data.bdr
        padded_Hmaps = np.pad(Hmaps[:, :, :],
              ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        facttsr = pad_factor_prod(Hmaps, data.ccfactor, bdr=data.bdr)
        factnorm = np.linalg.norm(facttsr, axis=0)
        Hmap_merge = factnorm
        # Hmap_merge = padded_Hmaps.sum(axis=2)
        Hmap_norm = Hmap_merge / Hmap_merge.max()
        alphamsk_rsz = resize(Hmap_norm, [256, 256])
        plt.imshow(alphamsk_rsz)

        vmin, vmax = 0.08, 0.4
        alphamsk_clip = np.clip(alphamsk_rsz, vmin, vmax)#min(vmax, max(vmin, alphamsk_rsz));
        alphamsk_clip = (alphamsk_clip - vmin) / (vmax - vmin)

        maskmat = np.concatenate([np.zeros((256, 256, 3)), \
                                  1 - alphamsk_clip[:, :, np.newaxis]], axis=2)
        Image.fromarray((maskmat*255).astype("uint8"), mode='RGBA').\
            save(join(outdir, "Beto_exp%03d_mask.png"%Expi))

        plt.figure(figsize=[5, 5])
        plt.imshow(padded_Hmaps**2 / (padded_Hmaps**2).max())
        plt.colorbar()
        plt.savefig(join(outdir, "Beto_exp%03d_mask_nonmerge.png"%Expi))
        imglist = os.listdir(protodir)
        for nm in imglist:
            copy(join(protodir, nm), join(outdir, "Beto_exp%03d_"%(Expi) + nm))
            protoimg = imread(join(protodir, nm))
            imsave(join(outdir, "Beto_exp%03d_masked_"%(Expi) + nm), \
                   (protoimg * alphamsk_clip[:, :, np.newaxis]).astype("uint8"))
    except FileNotFoundError as e:
        print(e.args)
