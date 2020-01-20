import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
# from cv2 import imread, imwrite
import matplotlib
matplotlib.use('Agg') # if you dont want image show up
import matplotlib.pylab as plt
import sys
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from caffenet import *
from hessian import hessian
#% Set up PerceptualLoss judger
sys.path.append(r"D:\Github\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
model = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
model_alex = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=1, gpu_ids=[0])
#%%
hess_dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Artiphysiology\Hessian"
output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"
output_dir = join(output_dir,"rand_spot")
os.makedirs(output_dir)
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
#%%
# Load Generator
basedir = r"D:/Generator_DB_Windows/nets"
save_path = os.path.join(basedir, r"upconv/fc6/generator_state_dict.pt")
protofile = os.path.join(basedir, r"upconv/fc6/generator.prototxt")  # 'resnet50/deploy.prototxt'
weightfile = os.path.join(basedir, r'upconv/fc6/generator.caffemodel')  # 'resnet50/resnet50.caffemodel'
Generator = CaffeNet(protofile)
print(Generator)
if os.path.exists(save_path):
    Generator.load_state_dict(torch.load(save_path))
else:
    Generator.load_weights(weightfile)
    Generator.save(Generator.state_dict(), save_path)
Generator.eval()
Generator.verbose = False
Generator.requires_grad_(requires_grad=False)
for param in Generator.parameters():
    param.requires_grad = False
# % Load transformer and De-transformer
# import net_utils
# detfmr = net_utils.get_detransformer(net_utils.load('generator'))
# tfmr = net_utils.get_transformer(net_utils.load('caffe-net'))
# Constants for Detransforming Caffe images
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean,(1,3,1,1))
RGB_perm = [2, 1, 0]
#%%
def decaf_tfm(blob):
    clamp_img = torch.clamp(blob + BGR_mean, 0, 255)
    resz_img = F.interpolate(clamp_img - BGR_mean, (224, 224), mode='bilinear', align_corners=True) / 255 # resized
    resz_img = resz_img[:, RGB_perm, :, :]
    img_vis = (clamp_img - BGR_mean)[:, RGB_perm, :, :] / 255 # non resized
    return resz_img, img_vis
#%%
def sim_hessian_computation(z, percept_loss, savepath=None):
    """
    Depending on Generator imported from caffe to pytorch.
    Depending on Hessian, and autograd

    :param z: vector to compute hessian at
    :param percept_loss: the model from PerceptualSimilarity package
    :return: H: Hessian Matrix
     eigval: eigen decomposition, eigen values
     eigvec: eigen vectors
     gradient: gradient from the spot
     d_sim: similarity metric
    """
    feat = Variable(torch.from_numpy(z), requires_grad=False)
    # feat.requires_grad = False
    blobs = Generator(feat)  # forward the feature vector through the GAN
    resz_ref_img, ref_img_vis = decaf_tfm(blobs['deconv0'])
    # ref_img = blobs['deconv0']  # get raw output image from GAN
    # clamp_ref_img = torch.clamp(ref_img + BGR_mean, 0, 255)
    # resz_ref_img = F.interpolate(clamp_ref_img - BGR_mean, (224, 224), mode='bilinear', align_corners=True) / 255
    # resz_ref_img = resz_ref_img[:, RGB_perm, :, :]
    # ref_img_vis = (clamp_ref_img - BGR_mean)[:, RGB_perm, :, :] / 255  # no resize!
    # Perturbation vector for gradient
    perturb_vec = np.zeros((1, 4096))  # 0.00001*np.random.rand(1, 4096)
    perturb_vec = torch.from_numpy(np.float32(perturb_vec))
    perturb_vec = Variable(perturb_vec, requires_grad=True)
    blobs = Generator(feat + perturb_vec)
    resz_out_img, _ = decaf_tfm(blobs['deconv0'])
    # out_img = blobs['deconv0']  # get raw output image from GAN
    # clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    # resz_out_img = F.interpolate(clamp_out_img - BGR_mean, (224, 224), mode='bilinear', align_corners=True) / 255
    # resz_out_img = resz_out_img[:, RGB_perm, :, :]

    d_sim = percept_loss.forward(resz_ref_img, resz_out_img)
    gradient = torch.autograd.grad(d_sim, perturb_vec, retain_graph=True)[0]
    H = hessian(d_sim[0, 0, 0, 0], perturb_vec, create_graph=False)  # 10min for computing a Hessian
    eigval, eigvec = np.linalg.eigh(H.cpu().detach().numpy())  # eigen decomposition for a symmetric array! ~ 5.7 s
    # Print statistics
    pos1_nums = (eigval > 1).sum()
    pos_nums = (eigval > 0.1).sum()
    pos01_nums = (eigval > 0.01).sum()
    num01 = (np.logical_and(eigval < 0.1, eigval > -0.1)).sum()
    num001 = (np.logical_and(eigval < 0.01, eigval > -0.01)).sum()
    num0001 = (np.logical_and(eigval < 0.001, eigval > -0.001)).sum()
    neg1_nums = (eigval < - 1).sum()
    neg_nums = (eigval < - 0.1).sum()
    print("[1, inf]:%d; [0.1, inf]:%d; [0.01, inf]:%d, [-0.1,0.1]: %d; [-0.01,0.01]: %d; [-0.001,0.001]: %d; [-inf, -0.1]:%d; [-inf, -1]:%d" % (
            pos1_nums, pos_nums, pos01_nums, num01, num001, num0001, neg_nums, neg1_nums))
    # Save to disk
    if savepath is not None:
        np.savez(savepath,
             z=feat.detach().numpy(),
             activation=d_sim.cpu().detach().numpy(),
             grad=gradient.cpu().numpy(), H=H.cpu().detach().numpy(),
             heig=eigval, heigvec=eigvec)
        # join(output_dir, "hessian_sim_alex_lin_%s_%d.npz" % (unit[1], unit[2]))
    return H, eigval, eigvec, gradient, d_sim
#%%
def visualize_eig_spectra(gradient,eigval,figh):
    g = gradient.numpy()
    g = np.sort(g)
    plt.figure(figh.number)
    #plt.set_current_figure(figh)
    plt.clf()
    plt.subplot(211)
    plt.plot(g[0, ::-1])
    plt.xticks(np.arange(0, 4200, 200))
    plt.ylabel("Gradient")
    plt.subplot(212)
    plt.plot(eigval[::-1])
    plt.xticks(np.arange(0, 4200, 200))
    plt.ylabel("EigenVal of Hessian")
    return figh
#%%
# original latent vector as reference
for unit in unit_arr[:]:
    # CNN = CNNmodel(unit[0])  # 'caffe-net'
    # CNN.select_unit(unit)
    data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2]))) # Load the hessian data of hotspot
    z = data["z"]
    unit_savedir = join(hess_dir, "%s_%s_%d" % unit[0:3])
    os.makedirs(unit_savedir, exist_ok=True)
    savepath = join(output_dir, "hessian_sim_alex_lin_%s_%d.npz" % (unit[1], unit[2]))
    H, eigval, eigvec, gradient, d_sim = sim_hessian_computation(z, model, savepath=savepath)
    figh = plt.figure(2, figsize=[12, 6]);plt.clf()
    visualize_eig_spectra(gradient,eigval,figh)
    plt.suptitle("Gradient and Hessian Spectrum of Hotspot under Alexnet Linear Perceptual Metric")
    plt.savefig(join(output_dir, "%s_%s_%d_sim_alex_lin_hessian_eig.png" % (unit[0], unit[1], unit[2])))

#%%
for i in range(1,20):
    z = 4 * np.random.randn(1, 4096).astype(np.float32)
    _, ref_img_vis = decaf_tfm(Generator(Variable(torch.from_numpy(z)))['deconv0'])
    plt.figure(1, figsize=[3, 3])
    plt.clf()
    plt.imshow((1 + ref_img_vis.permute([2, 3, 1, 0]).squeeze()) / 2)
    plt.axis("off")
    plt.savefig(join(output_dir, "RAND_VEC_%d_img.png" % (i)))

    savepath = join(output_dir, "hessian_sim_squeeze_lin_randspot%d.npz" % (i))
    H, eigval, eigvec, gradient, d_sim = sim_hessian_computation(z, model, savepath=savepath)
    figh = plt.figure(2, figsize=[12, 6]); plt.clf()
    visualize_eig_spectra(gradient,eigval,figh)
    figh.suptitle("Gradient and Hessian Spectrum of Hotspot under Squeezenet Linear Perceptual Metric")
    figh.savefig(join(output_dir, "RAND_VEC_%d_sim_squeeze_lin_hessian_eig.png" % (i)))

    savepath = join(output_dir, "hessian_sim_alex_lin_randspot%d.npz" % (i))
    H, eigval, eigvec, gradient, d_sim = sim_hessian_computation(z, model_alex, savepath=savepath)
    figh = plt.figure(2, figsize=[12, 6]);plt.clf()
    visualize_eig_spectra(gradient, eigval, figh)
    figh.suptitle("Gradient and Hessian Spectrum of Hotspot under Alexnet Linear Perceptual Metric")
    figh.savefig(join(output_dir, "RAND_VEC_%d_sim_alex_lin_hessian_eig.png" % (i)))

    savepath = join(output_dir, "hessian_sim_vgg_lin_randspot%d.npz" % (i))
    H, eigval, eigvec, gradient, d_sim = sim_hessian_computation(z, model_vgg, savepath=savepath)
    figh = plt.figure(2, figsize=[12, 6]);plt.clf()
    visualize_eig_spectra(gradient, eigval, figh)
    figh.suptitle("Gradient and Hessian Spectrum of Hotspot under VGGnet Linear Perceptual Metric")
    figh.savefig(join(output_dir, "RAND_VEC_%d_sim_vgg_lin_hessian_eig.png" % (i)))
    # figh.show()

    #%%
_, ref_img_vis = decaf_tfm(
    Generator(Variable(torch.from_numpy(5*np.random.randn(1, 4096).astype(np.float32))))['deconv0'])
plt.figure(1, figsize=[3,3]); plt.clf()
plt.imshow((1 + ref_img_vis.permute([2,3,1,0]).squeeze())/2)
plt.axis("off")
plt.savefig("tmp.png")