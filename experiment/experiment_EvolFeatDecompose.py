# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:58:22 2021

@author: Binxu Wang
"""
%load_ext autoreload
%autoreload 2
#%%
backup_dir = r'C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2021-04-16-12-05-37'
threadid = 1

exptime = backup_dir.split("\\")[-1]
#%%
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

sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"Visual_Neuron_Modelling"))
sys.path.append(join(Python_dir,"PerceptualSimilarity"))
import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from GAN_utils import upconvGAN
from skimage.io import imsave, imread
from torchvision.utils import make_grid
from build_montages import build_montages
from  scipy.io import loadmat
import matplotlib.pylab as plt
def visualize_cctsr_simple(featFetcher, layers2plot, imgcol, savestr="Evol", titstr="Alfa_Evol", figdir=""):
    """
    Demo
    ExpType = "EM_cmb"
    layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
    figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    """
    nlayer = len(layers2plot)
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer,8])
    if imgcol is not None:
        for imgi in range(4):
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
#%% Load basic information
data = loadmat(join(backup_dir, "Evol_ScoreImgTraj.mat"))
imgfp_col = data.get("imgfp_col")
score_col = data.get("score_col")
imgsize = data.get('imgsize').astype('float')
imgpos = data.get('imgpos').astype('float')
pref_chan = data.get('prefchan').astype('int')
imgsize = imgsize[threadid-1, 0]
imgpos = imgpos[threadid-1, :]
pref_chan = pref_chan[threadid-1, 0]
scorevec_thread = score_col[0, threadid-1][:,0]
imgfp_thread = imgfp_col[0, threadid-1]
#%% Collect some best images
score_idx = np.argsort(-scorevec_thread)
score_examp = scorevec_thread[score_idx[:4]]
imgfp_examp = imgfp_thread[score_idx[:4]]
imgcol_examp = [imread(fp) for fp in imgfp_examp]
#%%
from torchvision import models
from CorrFeatTsr_lib import Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, visualize_cctsr
from featvis_lib import rectify_tsr, tsr_factorize, vis_featmap_corr, vis_feattsr, \
    vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, vis_featvec_point, load_featnet

netname = "vgg16"
ccdir = join(backup_dir, "CCFactor_%s"%netname)
os.makedirs(join(ccdir, "img"), exist_ok=True)
featnet, net = load_featnet(netname)
G = upconvGAN("fc6")
G.requires_grad_(False).cuda().eval();
#%%
layers2plot = ["conv2_2", "conv3_3", "conv4_3",  "conv5_3", ]
imgpix = int(imgsize * 40) #%224  # 
#    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(net, ["conv2_2", "conv3_3","conv4_3", "conv5_3"])
featFetcher.init_corr()
#    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="S")
Corr_Feat_pipeline(featnet, featFetcher, scorevec_thread, imgfp_thread,
        lambda x:loadimg_preprocess(x, borderblur=True, imgpix=120), online_compute=True,
        batchsize=100, savedir=ccdir, savenm="Evol" ) #  % (Animal, Expi, expsuffix),
corrDict = np.load(join(ccdir, "%s_corrTsr.npz" % ("Evol")), allow_pickle=True)
figh = visualize_cctsr_simple(featFetcher, layers2plot, imgcol_examp, savestr="Alfa_Evol%s_%s"%(exptime,netname), 
                              titstr="Alfa_Evol%s_%s"%(exptime,netname), figdir=ccdir)

# corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)), allow_pickle=True)#
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()
#%% OK starts decompostion.
layer = "conv4_3"
bdr = 1; NF = 3; rect_mode = "abs"


Ttsr = Ttsr_dict[layer]
cctsr = cctsr_dict[layer]
Ttsr_pp = rectify_tsr(Ttsr, rect_mode)  # "mode="thresh", thr=(-5,5))
Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=ccdir, savestr="%s-%s"%(netname, layer))
#%%
torchseed = int(time())
torch.manual_seed(torchseed)
finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, 
                                Bsize=5, figdir=ccdir, savestr="", saveimg=True)
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, 
                                Bsize=5, bdr=bdr, figdir=ccdir, savestr="", saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, 
                     featnet=featnet, Bsize=5, figdir=ccdir, savestr="", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname, \
                     featnet=featnet, bdr=bdr, Bsize=5, figdir=ccdir, savestr="", imshow=False, saveimg=True)
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,\
                     featnet=featnet, bdr=bdr, Bsize=5, figdir=ccdir, savestr="", imshow=False, saveimg=True)
#%%
score_examp = scorevec_thread[score_idx[:5]]
imgfp_examp = imgfp_thread[score_idx[:5]]
imgcol_examp = [imread(fp) for fp in imgfp_examp]
for i, img in enumerate(imgcol_examp):
    imgid = imgfp_examp[i].split("\\")[-1].split(".")[0]
    imsave(join(ccdir, "img", "evol_best_%02d_%s.png"%(i, imgid)), img)
#%%
np.savez(join(ccdir, "factor_record.npz"), Hmat=Hmat, Hmaps=Hmaps, Tcomponents=Tcomponents, ccfactor=ccfactor, 
    netname=netname, layer=layer, bdr=bdr, NF=NF, rect_mode=rect_mode, torchseed=torchseed)