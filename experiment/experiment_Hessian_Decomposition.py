# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:08:54 2020

@author: Binxu Wang
"""
#%% Prepare the generator model and perceptual loss networks
from time import time
import os
from os.path import join
import sys
if os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
    Python_dir = r"C:\Users\Ponce lab\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
    Python_dir = r"C:\Users\Ponce lab\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':
    Python_dir = r"E:\Github_Projects"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"

sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"PerceptualSimilarity"))
import torch
from GAN_utils import upconvGAN
from GAN_hvp_operator import GANHVPOperator, compute_hessian_eigenthings
import models  # from PerceptualSimilarity folder
from build_montages import build_montages
# model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
G = upconvGAN(name="fc6")
G.cuda()
model_squ.cuda()
for param in model_squ.parameters():
    param.requires_grad_(False)
for param in G.parameters():
    param.requires_grad_(False)
#%% Test code for hessian eigendecomposition
#t0 = time()
#feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
#eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_squ,
#    num_eigenthings=300, mode="lanczos", use_gpu=True,)
#print(time() - t0,"\n")  # 81.02 s 
#%%
import os
from  scipy.io import loadmat
import re
def load_codes_mat(backup_dir, savefile=False):
    """ load all the code mat file in the experiment folder and summarize it into nparrays"""
    # make sure enough codes for requested size
    if "codes_all.npz" in os.listdir(backup_dir):
        # if the summary table exist, just read from it!
        with np.load(join(backup_dir, "codes_all.npz")) as data:
            codes_all = data["codes_all"]
            generations = data["generations"]
        return codes_all, generations
    codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "_code.mat" in fn])
    codes_all = []
    img_ids = []
    for i, fn in enumerate(codes_fns[:]):
        matdata = loadmat(join(backup_dir, fn))
        codes_all.append(matdata["codes"])
        img_ids.extend(list(matdata["ids"]))

    codes_all = np.concatenate(tuple(codes_all), axis=0)
    img_ids = np.concatenate(tuple(img_ids), axis=0)
    img_ids = [img_ids[i][0] for i in range(len(img_ids))]
    generations = [int(re.findall("gen(\d+)", img_id)[0]) if 'gen' in img_id else -1 for img_id in img_ids]
    if savefile:
        np.savez(join(backup_dir, "codes_all.npz"), codes_all=codes_all, generations=generations)
    return codes_all, generations
#%% 
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pylab as plt
from imageio import imwrite
backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_integrated\2020-06-01-09-46-37"
newimg_dir = join(backup_dir,"Hess_imgs")
summary_dir = join(backup_dir,"Hess_imgs","summary")

os.makedirs(newimg_dir,exist_ok=True)
os.makedirs(summary_dir,exist_ok=True)
print("Loading the codes from experiment folder %s", backup_dir)
codes_all, generations = load_codes_mat(backup_dir)
generations = np.array(generations)
print("Shape of code", codes_all.shape)
#%%
final_gen_norms = np.linalg.norm(codes_all[generations==max(generations), :], axis=1)
final_gen_norm = final_gen_norms.mean()
print("Average norm of the last generation samples %.2f" % final_gen_norm)
sphere_norm = final_gen_norm
print("Set sphere norm to the last generations norm!")
#%% Do PCA and find the major trend of evolution
print("Computing PCs")
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1

PC1_vect = PC1_sign * PC_vectors[0,:] 
#%% Compute Hessian decomposition and get the vectors
print("Computing Hessian Decomposition Through Lanczos decomposition")
t0 = time()
feat = torch.from_numpy(sphere_norm * PC1_vect).float().requires_grad_(False).cuda()
eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_squ,
    num_eigenthings=800, mode="lanczos", use_gpu=True)
print("%.2f sec"% (time() - t0))  # 31.75 secs for 300 eig, 87.52 secs for 800 eigs. 
eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[::-1,:]
#% Angle with PC1 vector
innerprod2PC1 = PC1_vect @ eigenvecs.T
print("EigenDecomposition of Hessian of Image Similarity Metric\nEigen value: max %.3E min %.3E std %.3E \nEigen vector: Innerproduct max %.3E min %.3E std %.3E"%
      (eigenvals.max(), eigenvals.min(), eigenvals.std(), innerprod2PC1.max(), innerprod2PC1.min(), innerprod2PC1.std())) 
#%% Create images along the spectrum
save_indiv = False
save_row = False
vec_norm = sphere_norm
ang_step = 180 / 10
theta_arr_deg = ang_step * np.linspace(-5,5,21)# np.arange(-5, 6)
theta_arr = theta_arr_deg / 180 * np.pi
img_list_all = []
for eig_id in [0,1,5,10,15,20,40,60,80,99,150,200,250,299,450,600,799]:
    # eig_id = 0
    perturb_vect = eigenvecs[eig_id,:] # PC_vectors[1,:]    
    codes_arc = np.array([np.cos(theta_arr), 
                          np.sin(theta_arr) ]).T @ np.array([PC1_vect, perturb_vect])
    norms = np.linalg.norm(codes_arc, axis=1)
    codes_arc = codes_arc / norms[:,np.newaxis] * vec_norm
    imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
    npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()
    if save_indiv:
        for i in range(npimgs.shape[3]):
            angle = theta_arr_deg[i]
            imwrite(join(newimg_dir, "norm%d_eig%d_ang%d.jpg"%(vec_norm, eig_id, angle)), npimgs[:,:,:,i])
    
    img_list = [npimgs[:,:,:,i] for i in range(npimgs.shape[3])]
    img_list_all.extend(img_list)
    if save_row:
        mtg1 = build_montages(img_list, [256, 256], [len(theta_arr), 1])[0]
        imwrite(join(summary_dir, "norm%d_eig_%d.jpg"%(vec_norm, eig_id)),mtg1)
mtg_all = build_montages(img_list_all, [256, 256], [len(theta_arr), int(len(img_list_all)//len(theta_arr))])[0]
imwrite(join(summary_dir, "norm%d_eig_all.jpg"%(vec_norm)),mtg_all)
#%% 
#vec_norm = 220# sphere_norm
#eig_id = 0
#perturb_vect = eigenvecs[eig_id,:] # PC_vectors[1,:]    
#ang_step = 180 / 10
#theta_arr_deg = ang_step * np.arange(-5, 6)
#theta_arr = ang_step * np.arange(-5, 6) / 180 * np.pi
#codes_arc = np.array([np.cos(theta_arr), 
#                      np.sin(theta_arr) ]).T @ np.array([PC1_vect, perturb_vect])
#norms = np.linalg.norm(codes_arc, axis=1)
#codes_arc = codes_arc / norms[:,np.newaxis] * vec_norm
#imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
#
#npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()
#for i in range(npimgs.shape[3]):
#    angle = theta_arr_deg[i]
#    imwrite(join(newimg_dir, "norm%d_eig%d_ang%d.jpg"%(vec_norm, eig_id, angle)), npimgs[:,:,:,i])
#
#img_list = [npimgs[:,:,:,i] for i in range(npimgs.shape[3])]
#mtg1 = build_montages(img_list, [256, 256], [11, 1])[0]
##imwrite(join(backup_dir, "norm%d_eig%d.jpg"%(vec_norm, eig_id)),mtg1)
#imwrite(join(newimg_dir, "norm%d_eig%d.jpg"%(vec_norm, eig_id)),mtg1)