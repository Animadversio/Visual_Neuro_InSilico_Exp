# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:36:45 2021

@author: PonceLab-Office
"""
from time import time
import os
import re
from os.path import join
import sys
if os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
    Python_dir = r"C:\Users\Ponce lab\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
    Python_dir = r"C:\Users\Poncelab-ML2a\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':
    Python_dir = r"E:\Github_Projects"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-OFFICE':
    Python_dir = r"G:\My Drive\Python"

sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"Visual_Neuron_Modelling"))
# sys.path.append(join(Python_dir,"PerceptualSimilarity"))
from glob import glob
from skimage.io import imread
import numpy as np
from scipy.stats import sem
import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from GAN_utils import upconvGAN
from skimage.io import imsave, imread
from torchvision.utils import make_grid
from build_montages import build_montages
from  scipy.io import loadmat
import matplotlib.pylab as plt
from torchvision import models
from CorrFeatTsr_lib import Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, visualize_cctsr
from CorrFeatTsr_predict_lib import score_images, softplus, fitnl_predscore
from featvis_lib import rectify_tsr, tsr_factorize, vis_featmap_corr, vis_feattsr, \
    vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, vis_featvec_point, load_featnet, \
    tsr_posneg_factorize, posneg_sep
    
def visualize_cctsr_simple(featFetcher, layers2plot, imgcol, savestr="Evol", titstr="Alfa_Evol", figdir=""):
    """
    Demo
    ExpType = "EM_cmb"
    layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
    figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    """
    nlayer = max(4, len(layers2plot))
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer,8])
    if imgcol is not None:
        for imgi in range(len(imgcol)):
            axs[0,imgi].imshow(imgcol[imgi])
            axs[0,imgi].set_title("Highest Score Evol Img")
            axs[0,imgi].axis("off")
    for li, layer in enumerate(layers2plot):
        chanN = featFetcher.cctsr[layer].shape[0]
        tmp=axs[1,li].matshow(np.nansum(featFetcher.cctsr[layer].abs().numpy(),axis=0) / chanN)
        plt.colorbar(tmp, ax=axs[1,li])
        axs[1,li].set_title(layer+" mean abs cc")
        tmp=axs[2,li].matshow(np.nanmax(featFetcher.cctsr[layer].abs().numpy(),axis=0))
        plt.colorbar(tmp, ax=axs[2,li])
        axs[2,li].set_title(layer+" max abs cc")
    figh.suptitle("%s Exp Corr Feat Tensor"%(titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_corrTsr_vis.png" % (savestr)))
    figh.savefig(join(figdir, "%s_corrTsr_vis.pdf" % (savestr)))
    return figh
#%% What carlos need to fill in
Animal = ... 
exptime = ""
scorevec_thread = ...
imgfp_thread = ...
outputdir = ...  #join(backup_dir, "CCFactor_%s"%netname)
#%%
dicarlo_imlist = sorted(glob(r"N:\Stimuli\Kar_DiCarlo\*.tif"))
Animal = "Alfa"
exptime = "20211111"
scorevec_thread = np.random.randn(len(dicarlo_imlist))
imgfp_thread = np.array(dicarlo_imlist)
outputdir = r"C:\Users\PonceLab-Office\Documents\ccfactor_tmp"
#%%
score_idx = np.argsort(-scorevec_thread)
score_examp = scorevec_thread[score_idx[:4]]
imgfp_examp = imgfp_thread[score_idx[:4]]
imgcol_examp = [imread(fp) for fp in imgfp_examp]
#%%
# netname = "alexnet";layers2plot = ["conv2", "conv3", "conv4", "conv5",]
# netname = "vgg16";layers2plot = ["conv2_2", "conv3_3", "conv4_3",  "conv5_3", ]
netname = "resnet50"; layers2plot = ["layer2", "layer3", "layer4", ]
# netname = "resnet50_linf8";layers2plot = ["layer2", "layer3", "layer4", ]
# ccdir = "debug_tmp_%s"%netname

ccdir = join(outputdir, "CCFactor_%s-%s"%(netname,layer))
os.makedirs(join(ccdir, "img"), exist_ok=True)
# figh.savefig(join(outputdir,"ExpEvolTraj.png"))
featnet, net = load_featnet(netname)
imgpix = 224
# G = upconvGAN("fc6")
# G.requires_grad_(False).cuda().eval();
#%% Correlation fitting
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(net, layers2plot, netname=netname,)
featFetcher.init_corr()
Corr_Feat_pipeline(featnet, featFetcher, scorevec_thread, imgfp_thread,
        lambda x:loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
        batchsize=100, savedir=outputdir, savenm="Evol" ) #  % (Animal, Expi, expsuffix),
corrDict = np.load(join(outputdir, "%s_corrTsr.npz" % ("Evol")), allow_pickle=True)
figh = visualize_cctsr_simple(featFetcher, layers2plot, imgcol_examp, savestr="%s_Evol%s_%s"%(Animal,exptime,netname), 
                              titstr="%s_Evol%s_%s"%(Animal,exptime,netname), figdir=ccdir)
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()
stdtsr_dict = corrDict.get("featStd").item()
featFetcher.clear_hook()

layer = "layer3"; bdr = 1;
vis_score_mode = "cosine" # "corr"


NF = 3; rect_mode = "Tthresh"; thresh = (None, 3)#"pos"
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
                                     figdir=ccdir, savestr="%s-%s"%(netname, layer))
Tcomponents = None
#%% make the scorer
from featvis_lib import show_img
from CorrFeatTsr_predict_lib import fitnl_predscore, loadimg_preprocess, score_images, CorrFeatScore
scorer = CorrFeatScore()
padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
# padded_mask = np.pad(Hmaps[:, :, ci:ci + 1], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
# fact_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci + 1], padded_mask))
fact_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
# if show_featmap or imshow: show_img(padded_mask[:, :, 0])
scorer.register_weights({layer: fact_Wtsr})
scorer.register_hooks(net, layer, netname=netname)

# imgfps = imgfp_thread 
# pred_scores = score_images(featnet, scorer, layer, imgfps, imgloader=loadimg_preprocess, batchsize=70,)
#%% Predict new images
dicarlo_imlist = sorted(glob(r"N:\Stimuli\Kar_DiCarlo\*.tif"))
pred_scores_dicarlo = score_images(featnet, scorer, layer, dicarlo_imlist, imgloader=loadimg_preprocess, batchsize=70,)
best_idx = np.argsort(-pred_scores_dicarlo)
#%% See the best and worst images.
mtg = build_montages([imread(dicarlo_imlist[i]) for i in best_idx[:25]],[256,256],[5,5])
plt.imshow(mtg[0])
plt.show()