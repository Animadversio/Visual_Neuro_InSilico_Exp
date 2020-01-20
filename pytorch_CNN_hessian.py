import sys
import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from cv2 import imread, imwrite
import matplotlib.pylab as plt
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit  # Compute receptive field
from caffenet import *  # Pytorch-caffe converter
from hessian import hessian
print(torch.cuda.current_device())
# print(torch.cuda.device(0))
if torch.cuda.is_available():
    print(torch.cuda.device_count(), " GPU is available:", torch.cuda.get_device_name(0))
#%%
output_dir = join(r"D:\Generator_DB_Windows\data\with_CNN", "hessian")
os.makedirs(output_dir,exist_ok=True)
#% Put all the UNITS of INTEREST HERE! Or input from cmdline !
unit_arr = [('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc6', 2),
            ('caffe-net', 'fc6', 3),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc7', 2),
            ('caffe-net', 'fc8', 10),
            ('caffe-net', 'conv1', 10, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv3', 10, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ]
#%% Prepare PyTorch version of the Caffe networks
from torch_net_utils import load_caffenet,load_generator
net = load_caffenet()
Generator = load_generator()
import net_utils
detfmr = net_utils.get_detransformer(net_utils.load('generator'))
tfmr = net_utils.get_transformer(net_utils.load('caffe-net'))
#%% Script for running Evolution and hessian computation!
# def display_image(ax, out_img):
#     deproc_img = detfmr.deprocess('data', out_img.data.numpy())
#     ax.imshow(np.clip(deproc_img, 0, 1))
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1))
for unit in unit_arr:
    print(unit)
    Evol_Success = False
    feat = 0.05 * np.random.rand(1, 4096)
    feat = torch.from_numpy(np.float32(feat))
    feat = Variable(feat, requires_grad=True)
    offset = 16
    pipe_optimizer = optim.SGD([feat], lr=0.05)  # Seems Adam is not so good, Adagrad ... is not so
    score = []
    feat_norm = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    for step in range(200):
        if step % 5 == 0:
            pipe_optimizer.zero_grad()
        blobs = Generator(feat)  # forward the feature vector through the GAN
        out_img = blobs['deconv0']  # get raw output image from GAN
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
        resz_out_img = F.interpolate(clamp_out_img - BGR_mean, (227, 227), mode='bilinear', align_corners=True)
        blobs_CNN = net(resz_out_img)
        if len(unit) == 5:
            neg_activ = - blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
        elif len(unit) == 3:
            neg_activ = - blobs_CNN[unit[1]][0, unit[2]]
        else:
            neg_activ = - blobs_CNN['fc8'][0, 1]
        neg_activ.backward()
        if feat.grad.norm() < 1E-10:
            print("%d steps, No Gradient, Neuron activation %.3f, Re-initialize" % (step, - neg_activ.data.item()))
            feat = Variable(0.75 * torch.randn((1, 4096)), requires_grad=True)
            pipe_optimizer = optim.SGD([feat], lr=0.05)
            continue
        else:
            Evol_Success = True
        pipe_optimizer.step()
        score.append(- neg_activ.data.item())
        feat_norm.append(feat.norm(p=2).data.item())
        if step % 10 == 0:
            # display_image(ax, out_img)
            # display.clear_output(wait=True)
            # display.display(fig)
            print("%d steps, Neuron activation %.3f" % (step, - neg_activ.data.item()))
    if not Evol_Success:
        print(unit, " evolution not successful! Continue! ")
        continue
    deproc_img = detfmr.deprocess('data', out_img.data.numpy())
    plt.figure(figsize=[6, 6])
    plt.imshow(np.clip(deproc_img, 0, 1))# .view([224,224,3])
    plt.axis('off')
    plt.savefig(join(output_dir, "%s_%s_%d_final_image.png"%(unit[0], unit[1], unit[2])))
    plt.show()

    plt.figure(figsize=[12, 4])
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(score)), score)
    plt.title("Score Trajectory")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(score)), feat_norm)
    plt.title("Norm Trajectory")
    plt.savefig(join(output_dir, "%s_%s_%d_trajectory.png"%(unit[0], unit[1], unit[2])))
    plt.show()

    t0 = time()
    blobs = Generator(feat) # forward the feature vector through the GAN
    out_img = blobs['deconv0'] # get raw output image from GAN
    resz_out_img = F.interpolate(out_img, (224, 224), mode='bilinear', align_corners=True) # Differentiable resizing
    blobs_CNN = net(resz_out_img)
    if len(unit) == 5:
        neg_activ = - blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
    elif len(unit) == 3:
        neg_activ = - blobs_CNN[unit[1]][0, unit[2]]
    else:
        neg_activ = - blobs_CNN['fc8'][0, 1]
    gradient = torch.autograd.grad(neg_activ,feat,retain_graph=True)[0] # First order gradient
    H = hessian(neg_activ,feat, create_graph=False) # Second order gradient
    t1 = time()
    print(t1-t0, " sec, computing Hessian") # Each Calculation may take 1050s esp for deep layer in the network!
    eigval, eigvec = np.linalg.eigh(H.detach().numpy()) # eigen decomposition for a symmetric array! ~ 5.7 s
    g = gradient.numpy()
    g = np.sort(g)
    t2 = time()
    print(t2-t1, " sec, eigen factorizing hessian")
    np.savez(join(output_dir, "hessian_result_%s_%d.npz"%(unit[1], unit[2])),
             z=feat.detach().numpy(),
             activation=-neg_activ.detach().numpy(),
             grad=gradient.numpy(),H=H.detach().numpy(),
             heig=eigval,heigvec=eigvec)
    # Generating figure of Hessian Spectrum and Sorted Gradient entry
    plt.figure(figsize=[12,6])
    plt.subplot(211)
    plt.plot(g[0,::-1])
    plt.xticks(np.arange(0,4200,200))
    plt.ylabel("Gradient")
    plt.subplot(212)
    plt.plot(eigval[::-1])
    plt.xticks(np.arange(0,4200,200))
    plt.ylabel("EigenVal of Hessian")
    plt.suptitle("Gradient and Hessian Spectrum of Hotspot")
    plt.savefig(join(output_dir, "%s_%s_%d_hessian_eig.png"%(unit[0], unit[1], unit[2])))
    plt.show()