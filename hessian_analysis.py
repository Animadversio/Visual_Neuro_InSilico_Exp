
import sys
import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from cv2 import imread, imwrite
import matplotlib
matplotlib.use('Agg') # if you dont want image show up
import matplotlib.pylab as plt
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from caffenet import *
from hessian import hessian
import utils
from utils import generator
from insilico_Exp import CNNmodel
sys.path.append(r"D:\Github\PerceptualSimilarity")
import torch
import models
model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=1, gpu_ids=[0])
#%%
hess_dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Artiphysiology\Hessian"
output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"

def perturb_images_sphere(cent_vec, perturb_vec, PC2_ang_step = 18, PC3_ang_step = 18):
    sphere_norm = np.linalg.norm(cent_vec)
    vectors = np.zeros((3, cent_vec.size))
    vectors[  0, :] = cent_vec / sphere_norm
    vectors[1:3, :] = perturb_vec
    img_list = []
    for j in range(-5, 6):
        for k in range(-5, 6):
            theta = PC2_ang_step * j / 180 * np.pi
            phi = PC3_ang_step * k / 180 * np.pi
            code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                  np.sin(theta) * np.cos(phi),
                                  np.sin(phi)]]) @ vectors[0:3, :]
            code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
            img = generator.visualize(code_vec)
            img_list.append(img.copy())
    return img_list

def perturb_images_arc(cent_vec, perturb_vec, PC2_ang_step = 18, RNG=range(-5,6)):
    sphere_norm = np.linalg.norm(cent_vec)
    vectors = np.zeros((2, cent_vec.size))
    vectors[  0, :] = cent_vec / sphere_norm
    vectors[1:2, :] = perturb_vec
    img_list = []
    for j in RNG:
        theta = PC2_ang_step * j / 180 * np.pi
        code_vec = np.array([[np.cos(theta),
                              np.sin(theta)]]) @ vectors[0:2, :]
        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
    return img_list

def perturb_images_line(cent_vec, perturb_vec, PC2_step = 18, RNG=range(-5,6)):
    sphere_norm = np.linalg.norm(cent_vec)
    img_list = []
    for j in RNG:
        L = PC2_step * j
        code_vec = cent_vec + L * perturb_vec
        # code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
    return img_list

def visualize_img_and_tuning(img_list, scores, ticks, DS_num=4): # stepsize * RNG
    imgshow_num = len(img_list[::DS_num])
    show_score = scores[::DS_num]
    figh = plt.figure(figsize=[25, 7])
    gs = figh.add_gridspec(2, imgshow_num)
    ax_tune = figh.add_subplot(gs[1, :])
    plt.plot(ticks, scores)
    plt.xlabel("Code L2 Distance")
    ub = scores.max()
    lb = scores.min()
    title_cmap=plt.cm.viridis
    for i, img in enumerate(img_list[::DS_num]):
        plt.subplot(gs[i])
        plt.imshow(img[:])
        plt.axis('off')
        plt.title("{0:.2f}".format(show_score[i]), fontsize=16,
                  color=title_cmap((show_score[i] - lb) / (ub - lb)))  # normalize a value between [0,1]
    #plt.show()
    return figh, ax_tune


def visualize_img_scores_sim(img_list, scores, similarity, ticks, DS_num=4):
    # stepsize * RNG
    imgshow_num = len(img_list[::DS_num])
    show_score = scores[::DS_num]
    figh = plt.figure(figsize=[25, 9])
    gs = figh.add_gridspec(3, imgshow_num)
    ax_tune = figh.add_subplot(gs[2, :])
    plt.plot(ticks, scores)
    plt.xlabel("Code L2 Distance")
    ax_sim = figh.add_subplot(gs[0, :])
    plt.plot(ticks, similarity)
    plt.xlabel("Image dissimilarity with center")
    ub = scores.max()
    lb = scores.min()
    title_cmap=plt.cm.viridis
    for i, img in enumerate(img_list[::DS_num]):
        plt.subplot(gs[imgshow_num + i])
        plt.imshow(img[:])
        plt.axis('off')
        plt.title("{0:.2f}".format(show_score[i]), fontsize=16,
                  color=title_cmap((show_score[i] - lb) / (ub - lb)))  # normalize a value between [0,1]
    #plt.show()
    return figh, ax_tune, ax_sim

def vec_cos(v1, v2):
    return np.vdot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
# %

def similarity_score(img_list, img):
    d_sim_curv = model.forward(torch.from_numpy(np.array(img[np.newaxis, :])).permute([0, 3, 1, 2]),
                               torch.from_numpy(np.array(img_list)).permute([0, 3, 1, 2]))
    scores = d_sim_curv[:, 0, 0, 0].cpu().detach()
    return scores
#%%
if __name__=="__main__":

    #%%
    output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"
    unit_arr = [
                ('caffe-net', 'conv1', 10, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv3', 5, 10, 10),
                ('caffe-net', 'conv4', 5, 10, 10),
                ('caffe-net', 'conv5', 5, 10, 10),
                ('caffe-net', 'fc6', 1),
                ('caffe-net', 'fc6', 2),
                ('caffe-net', 'fc6', 3),
                ('caffe-net', 'fc7', 1),
                ('caffe-net', 'fc7', 2),
                ('caffe-net', 'fc8', 1),
                ('caffe-net', 'fc8', 10),
                ]
    # %% plot the spectrum on same plot and statistics of the matrix
    plt.figure(1).clf()
    plt.figure(2).clf()
    for unit in unit_arr:
        # unit = unit_arr[1]
        data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
        pos1_nums = (data["heig"] > 1).sum()
        pos_nums = (data["heig"] > 0.1).sum()
        num01 = (np.logical_and(data["heig"] < 0.1, data["heig"] > -0.1)).sum()
        num001 = (np.logical_and(data["heig"] < 0.01, data["heig"] > -0.01)).sum()
        num0001 = (np.logical_and(data["heig"] < 0.001, data["heig"] > -0.001)).sum()
        print("%s [1, inf]:%d, [0.1, inf]:%d, [-0.1,0.1]: %d; [-0.01,0.01]: %d; [-0.001,0.001]: %d; " % (unit, pos1_nums, pos_nums, num01, num001, num0001))
        plt.figure(1)
        plt.plot(data["heig"][:100], label="%s-%s" % (unit[0], unit[1]), alpha=0.5, lw=2)
        plt.figure(2)
        plt.plot(data["heig"][-50:], label="%s-%s" % (unit[0], unit[1]), alpha=0.5, lw=2)
    plt.figure(1)
    plt.legend()
    plt.savefig(join(hess_dir, "FirstNegEig_cmp.png"))
    plt.figure(2)
    plt.legend()
    plt.savefig(join(hess_dir, "LastPosEig_cmp.png"))
    #plt.show()
    #%% Angle distribution of hessian
    for unit in unit_arr[-2:-1]:
        data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
        z = data["z"]
        G = data["grad"]
        Heig = data["heig"]
        Heigvec = data["heigvec"]
        vec_cos(z, Heigvec[:, 1]), vec_cos(z, Heigvec[:, 2])
        # plt.plot(data["heig"][:100], label="%s-%s" % (unit[0], unit[1]), alpha=0.5, lw=2)
    # z=feat.detach().numpy(),
    # activation=-neg_activ.detach().numpy(),
    # grad=gradient.numpy(),H=H.detach().numpy(),
    # heig=eigval,heigvec=eigvec
    #%% Visualize the tuning and images along the eigen vectors of the map
    unit = ('caffe-net', 'fc8', 1)
    CNN = CNNmodel(unit[0])  # 'caffe-net'
    CNN.select_unit(unit)
    #%%
    data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
    z = data["z"]
    G = data["grad"]
    Heig = data["heig"]
    Heigvec = data["heigvec"]
    #%%
    stepsize = 5
    RNG = np.arange(-20, 21)
    eigen_idx = -3
    eigen_val = Heig[eigen_idx]
    img_list = perturb_images_line(z, Heigvec[:, eigen_idx], PC2_step=stepsize, RNG=RNG)
    scores = CNNmodel.score(img_list)
    figh = utils.visualize_img_list(img_list[::4], nrow=1, scores=scores[::4]) # visualize sparsely
    # Note the maximum found by exact gradient descent has smaller norm than those found by CMA-ES optimization.
    figh.set_figheight(4)
    figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E"%(unit, eigen_idx, eigen_val))
    #%%
    """Visualize images from a list and maybe label the score on it!"""
    unit = ('caffe-net', 'fc8', 1)
    for unit in unit_arr:
        CNN = CNNmodel(unit[0])  # 'caffe-net'
        CNN.select_unit(unit)
        data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
        z = data["z"]
        G = data["grad"]
        Heig = data["heig"]
        Heigvec = data["heigvec"]
        unit_savedir = join(hess_dir, "%s_%s_%d"%unit[0:3])
        os.makedirs(unit_savedir,exist_ok=True)
        #%
        t1 = time()
        eig_arr = list(range(10))+list(range(50,250,50))+list(range(300,4000,100))+list(range(-200,0,50))+list(range(-10,0))
        eigen_idx = -2
        for eigen_idx in eig_arr:
            eigen_val = Heig[eigen_idx]
            # Linear perturbation
            stepsize = 4.5
            RNG = np.arange(-20, 21)
            img_list = perturb_images_line(z, Heigvec[:, eigen_idx], PC2_step=stepsize, RNG=RNG)
            scores = CNN.score(img_list)
            figh, ax_tune = visualize_img_and_tuning(img_list, scores, stepsize * RNG, DS_num=4)
            figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E"%(unit, eigen_idx, eigen_val), fontsize=16)
            figh.savefig(join(unit_savedir, "Tuning_eigid_%d"%eigen_idx))
            plt.close(figh)
            # figh.show()
            # Angular perturbation
            stepsize = 4.5
            RNG = np.arange(-20, 21)
            img_list = perturb_images_arc(z, Heigvec[:, eigen_idx], PC2_ang_step=stepsize, RNG=RNG)
            scores = CNN.score(img_list)
            figh, ax_tune = visualize_img_and_tuning(img_list, scores, stepsize * RNG, DS_num=4)
            ax_tune.set_xlabel("Code Angular Distance (deg)")
            figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E"%(unit, eigen_idx, eigen_val), fontsize=16)
            figh.savefig(join(unit_savedir, "Ang_Tuning_eigid_%d"%eigen_idx))
            plt.close(figh)
            # figh.show()
            print("Finish Computing Tuning Curve Along EigenVector %d (%.1f sec passed)" % (eigen_idx, time()-t1))


#%%
    # %% Plot the tuning together with similiarity
    #unit = ('caffe-net', 'fc8', 1)
    for unit in unit_arr[-4:]:
        CNN = CNNmodel(unit[0])  # 'caffe-net'
        CNN.select_unit(unit)
        data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
        z = data["z"]
        G = data["grad"]
        Heig = data["heig"]
        Heigvec = data["heigvec"]
        unit_savedir = join(hess_dir, "%s_%s_%d" % unit[0:3])
        os.makedirs(unit_savedir, exist_ok=True)
        # %
        t1 = time()
        eig_arr = list(range(10)) + list(range(50, 250, 50)) + list(range(300, 4000, 100)) + list(
            range(-200, 0, 50)) + list(range(-10, 0))
        eigen_idx = -2
        for eigen_idx in eig_arr:
            eigen_val = Heig[eigen_idx]
            stepsize = 4.5
            RNG = np.arange(-20, 21)
            img_list = perturb_images_line(z, Heigvec[:, eigen_idx], PC2_step=stepsize, RNG=RNG)
            scores = CNN.score(img_list)
            similarity = similarity_score(img_list, img_list[20])
            figh, ax_tune, ax_sim = visualize_img_scores_sim(img_list, scores, similarity, stepsize * RNG, DS_num=4)
            figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E" % (unit, eigen_idx, eigen_val), fontsize=16)
            figh.savefig(join(unit_savedir, "Tuning_Sim_eigid_%d" % eigen_idx))
            plt.close(figh)
            # figh.show()
            stepsize = 4.5
            RNG = np.arange(-20, 21)
            img_list = perturb_images_arc(z, Heigvec[:, eigen_idx], PC2_ang_step=stepsize, RNG=RNG)
            scores = CNN.score(img_list)
            similarity = similarity_score(img_list, img_list[20])
            figh, ax_tune, ax_sim = visualize_img_scores_sim(img_list, scores, similarity, stepsize * RNG, DS_num=4)
            ax_tune.set_xlabel("Code Angular Distance (deg)")
            figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E" % (unit, eigen_idx, eigen_val), fontsize=16)
            figh.savefig(join(unit_savedir, "Ang_Tuning_Sim_eigid_%d" % eigen_idx))
            plt.close(figh)
            # figh.show()
            print("Finish Computing Tuning Curve Along EigenVector %d (%.1f sec passed)" % (eigen_idx, time() - t1))
