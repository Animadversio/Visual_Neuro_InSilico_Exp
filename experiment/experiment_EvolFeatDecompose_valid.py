# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:07:21 2021

@author: Ponce lab
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

sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"Visual_Neuron_Modelling"))
sys.path.append(join(Python_dir,"PerceptualSimilarity"))
from featvis_lib import CorrFeatScore, score_images, fitnl_predscore, pad_factor_prod
#%%
DR_Wtsr = pad_factor_prod(Hmaps[:,:,:], ccfactor[:,:], bdr)
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
scorer.register_weights({layer: DR_Wtsr})
pred_score = score_images(featnet, scorer, layer, imgfp_thread, batchsize=60,
        imgloader=lambda x: loadimg_preprocess(x, imgpix=120, borderblur=True, ), )
scorer.clear_hook()
nlfunc, popt, pcov, scaling, nlpred_score = fitnl_predscore(pred_score.numpy(), scorevec_thread, savedir=ccdir, savenm="Evol_fit")
#%% Loading up the record for factors. 
data = np.load(join(ccdir, "factor_record.npz"))
#%%
[Hmat=Hmat, Hmaps=Hmaps, Tcomponents=Tcomponents, ccfactor=ccfactor, 
 netname=netname, layer=layer, bdr=bdr, NF=NF, rect_mode=rect_mode, torchseed=torchseed]
#%%
bhv2nm = "210416_Alfa_selectivity_basic_prefchan_rsp.mat"
seldir = r"C:\Users\Ponce lab\Documents\ml2a-monk\selectivityBasic"
#%%
newdata = loadmat(join(seldir, bhv2nm))
imgfps_facvis = [fp[0][0] for fp in newdata['imgfps']]
rspvec_pfch_facvis = newdata["rspvec_pfch"]
semvec_pfch_facvis = newdata["semvec_pfch"]
#%%
DR_Wtsr = cctsr##pad_factor_prod(Hmaps[:,:,1:], ccfactor[:,1:], bdr)
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
scorer.register_weights({layer: DR_Wtsr})
# scorer.register_weights({layer: np.reshape(ccfactor[:,1:2].mean(axis=1),[1,-1,1,1])})
pred_score = score_images(featnet, scorer, layer, imgfps_facvis, batchsize=60,
        imgloader=lambda x: loadimg_preprocess(x, imgpix=120, borderblur=True, ), )
scorer.clear_hook()
#%
nlfunc, popt, pcov, scaling, nlpred_score = fitnl_predscore(pred_score.numpy(), rspvec_pfch_facvis[:,0], savedir=ccdir, savenm="FactVis_Pred")


