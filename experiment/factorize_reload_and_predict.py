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

sys.path.append(join(Python_dir, "Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir, "Visual_Neuron_Modelling"))
# sys.path.append(join(Python_dir,"PerceptualSimilarity"))
from glob import glob
from skimage.io import imread
import numpy as np
from scipy.stats import sem
import torch
# from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from GAN_utils import upconvGAN
from skimage.io import imsave, imread
from torchvision.utils import make_grid
from build_montages import build_montages
from scipy.io import loadmat
import matplotlib.pylab as plt
from torchvision import models
from CorrFeatTsr_lib import Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, visualize_cctsr
from CorrFeatTsr_predict_lib import score_images, softplus, fitnl_predscore
from featvis_lib import rectify_tsr, tsr_factorize, vis_featmap_corr, vis_feattsr, \
    vis_feattsr_factor, vis_featvec, vis_featvec_wmaps, vis_featvec_point, load_featnet, \
    tsr_posneg_factorize, posneg_sep, CorrFeatScore
#%%
"""
Extra dependency 
    easydict, seaborn, kornia
"""
#%%
layer = "layer3"; netname = "resnet50_linf8"
featnet, net = load_featnet("resnet50_linf8")
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
#%%
# factorNPZ = join(r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8-layer3_thread2BigGAN","factor_record.npz")
# corrtsrNPZ = join(r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8_thread2BigGAN","Evol_corrTsr.npz")
factorNPZ = join(r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp\CCFactor_resnet50_linf8-layer3", "factor_record.npz")
corrtsrNPZ = join(r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp\CCFactor_resnet50_linf8", "Evol_corrTsr.npz")
print(list(np.load(factorNPZ)))
factdata = np.load(factorNPZ)
print(list(np.load(corrtsrNPZ)))
corrtsrdata = np.load(corrtsrNPZ)
#%%
def get_imgfullpath_list(stimpath):
    valid_imlist_raw = sorted(glob(stimpath+"\\*"))
    valid_imlist = [imgfp for imgfp in valid_imlist_raw if os.path.splitext(imgfp)[1].lower() in
                    [".bmp",".jpg",".png",".jpeg",".tif"]]
    return valid_imlist

def resample_correlation(scorecol, trial=100):
    """ Compute noise ceiling for correlating with a collection of noisy data"""
    resamp_scores_col = []
    for tri in range(trial):
        resamp_scores = np.array([np.random.choice(A, len(A), replace=True).mean() for A in scorecol])
        resamp_scores_col.append(resamp_scores)
    resamp_scores_arr = np.array(resamp_scores_col)
    resamp_ccmat = np.corrcoef(resamp_scores_arr)
    resamp_ccmat += np.diag(np.nan*np.ones(trial))
    split_cc_mean = np.nanmean(resamp_ccmat)
    split_cc_std = np.nanstd(resamp_ccmat)
    return split_cc_mean, split_cc_std
#%%
bdr = 1
ccfactor = factdata['ccfactor']
Hmaps = factdata['Hmaps']
padded_Hmaps = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
DR_Wtsr = np.einsum("cf,hwf->chw", ccfactor, padded_Hmaps)

# nlfunc, popt, pcov, scaling, nlpred_score = fitnl_predscore(pred_score.numpy(), score_vect)
#%%
Snat = loadmat("O:\\CorrFactor_natvalid\\Beto-20211220-3-5_selectivity.mat", squeeze_me=True)["Snat"]
#%%
outdir = "O:\\CorrFactor_natvalid\\2021-12-20-Beto-01-Chan02"
Snat = loadmat("O:\\CorrFactor_natvalid\\Beto-20211220-3-5_selectivity.mat", squeeze_me=True)["Snat"]
factorNPZs = [r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8-layer3_thread2BigGAN",
              r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp\CCFactor_resnet50_linf8-layer3"]
corrtsrNPZs = [r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8_thread2BigGAN",
               r"N:\Stimuli\2021-EvolDecomp\2021-12-20-Beto-01-decomp\CCFactor_resnet50_linf8"]

#%%
outdir = "O:\\CorrFactor_natvalid\\2021-12-22-Beto-01-Chan12"
Snat = loadmat(join(outdir, "Beto-22122021-003-005_selStats.mat"), squeeze_me=True)["Snat"]
factorNPZs = [r"N:\Stimuli\2021-EvolDecomp\2021-12-22-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8-layer3_threadBigGAN",
              r"N:\Stimuli\2021-EvolDecomp\2021-12-22-Beto-01-decomp\CCFactor_resnet50_linf8-layer3"]
corrtsrNPZs = [r"N:\Stimuli\2021-EvolDecomp\2021-12-22-Beto-01-decomp-BigGAN\CCFactor_resnet50_linf8_threadBigGAN",
               r"N:\Stimuli\2021-EvolDecomp\2021-12-22-Beto-01-decomp\CCFactor_resnet50_linf8"]

#%%
modelstrs = ["BigGANEvol 3 Factor Model","FC6Evol 3 Factor Model"]
expdatastrs = ["Natural Validation", "BigGAN Factorization", "FC6 Factorization"]
for mi, modelstr in enumerate(modelstrs):
    factdata = np.load(join(factorNPZs[mi], "factor_record.npz"))
    corrtsrdata = np.load(join(corrtsrNPZs[mi], "Evol_corrTsr.npz"))
    ccfactor = factdata['ccfactor']
    Hmaps = factdata['Hmaps']
    padded_Hmaps = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = np.einsum("cf,hwf->chw", ccfactor, padded_Hmaps)
    for j, expdatastr in enumerate(expdatastrs):
        stimpath = Snat[j]["meta"]["stimuli"].item()
        meanMat = Snat[j]["resp"]['meanMat'].item()
        meanMat_pref = Snat[j]["resp"]['meanvec_pref'].item()
        valid_imlist = Snat[j]["stim"]['imgfps'].item()

        prefchan_id = Snat[j]["units"]["pref_chan_id"].item()
        trial_col = list(Snat[j]["resp"]['trial_col'].item())
        trial_col_pref = [mat[prefchan_id - 1, :] for mat in trial_col]
        resamp_cc, resamp_ccstd = resample_correlation(trial_col_pref)
        print("Dataset bootstrapped self correlation %.3f+-%.3f" % (resamp_cc, resamp_ccstd))
        # valid_imlist = get_imgfullpath_list(stimpath)
        scorer = CorrFeatScore()
        scorer.register_hooks(net, layer, netname=netname)
        scorer.register_weights({layer: DR_Wtsr})
        pred_score = score_images(featnet, scorer, layer, valid_imlist, imgloader=loadimg_preprocess, batchsize=40,)
        scorer.clear_hook()
        nlfunc, popt, pcov, scaling, nlpred_score, fitstats = fitnl_predscore(pred_score.numpy(), meanMat_pref,
                          suptit="Model %s Pred dataset %s"%(modelstr, expdatastr), savedir=outdir,
                          savenm="Model%s-Pred%s"%(modelstr, expdatastr))
#%
from scipy.stats import spearmanr, pearsonr
# pearsonr(meanMat_pref, pred_score)
# spearmanr(meanMat_pref, pred_score)
#%%
# the fc6 evolution factorized model failed totally.
# BigGAN evolution factorized model has some success in predicting natural images.
#%%
#%%

#%%


