#%% cluster deploy version!
import torch
from torch_net_utils import load_generator
from skimage.transform import resize, rescale
from imageio import imread, imsave
import matplotlib.pylab as plt
import numpy as np
from os.path import join
import os
from time import time
G = load_generator("fc6")
G.requires_grad_(False)
G.cuda()
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1)).cuda()
#%%
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
# model = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
# percept_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
# target_img = imread(r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\Cat.jpg")

#%%
def visualize(G, code, mode="cuda", percept_loss=True):
    """Do the De-caffe transform (Validated)
    works for a single code """
    # if mode == "cpu":
    #     blobs = G(code)
    # else:
    blobs = G(code.cuda())
    out_img = blobs['deconv0']  # get raw output image from GAN
    # if mode == "cpu":
    #     clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    # else:
    clamp_out_img = torch.clamp(out_img + BGR_mean.cuda(), 0, 255)
    if percept_loss:  # tensor used to perform loss
        return clamp_out_img[:, [2, 1, 0], :, :] / (255 / 2) - 1
    else:
        vis_img = clamp_out_img[:, [2, 1, 0], :, :].permute([2, 3, 1, 0]).squeeze() / 255
        return vis_img

def L1loss(target, img):
    return (img - target).abs().sum(axis=2).mean()

def L2loss(target, img):
    return (img - target).pow(2).sum(axis=2).mean()
#%%
def img_backproj(target_img, lossfun=L1loss, nsteps=150, return_stat=True):
    tsr_target = target_img.astype(float)/255
    rsz_target = resize(tsr_target, (256, 256), anti_aliasing=True)
    tsr_target = torch.from_numpy(rsz_target).float().cuda()
    # assert size of this image is 256 256
    code = np.random.randn(4096)
    code = code.reshape(-1, 4096)
    feat = torch.from_numpy(code).float().requires_grad_(True)
    feat.cuda()
    optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss_col = []
    norm_col = []
    for i in range(nsteps):
        optimizer.zero_grad()
        img = visualize(G, feat)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better? 
        loss = lossfun(img, tsr_target)
        loss.backward()
        optimizer.step()
        norm_col.append(feat.norm().detach().item())
        loss_col.append(loss.detach().item())
        # print("step%d" % i, loss)
    print("step%d" % i, loss.item())
    if return_stat:
        return feat.detach(), img.detach(), loss_col, norm_col
    else:
        return feat.detach(), img.detach()

def img_backproj_PL(target_img, lossfun, nsteps=150, return_stat=True):
    #tsr_target = target_img.astype(float)#/255.0
    # assume the target img has been resized prior to this
    # rsz_target = resize(tsr_target, (256, 256), anti_aliasing=True)
    tsr_target = torch.from_numpy(target_img * 2.0 - 1).float().cuda()  # centered to be [-1, 1]
    tsr_target = tsr_target.unsqueeze(0).permute([0, 3, 1, 2])
    # assert size of this image is 256 256
    code = 4*np.random.randn(4096)
    code = code.reshape(-1, 4096)
    feat = torch.from_numpy(code).float().requires_grad_(True)
    feat.cuda()
    optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss_col = []
    norm_col = []
    for i in range(nsteps):
        optimizer.zero_grad()
        img = visualize(G, feat, percept_loss=True)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better?
        loss = lossfun(img, tsr_target)
        loss.backward()
        optimizer.step()
        norm_col.append(feat.norm().detach().item())
        loss_col.append(loss.detach().item())
        # print("step%d" % i, loss)
    print("step%d" % i, loss.item())
    img = visualize(G, feat, percept_loss=False)
    if return_stat:
        return feat.detach(), img.detach(), loss_col, norm_col
    else:
        return feat.detach(), img.detach()

def img_backproj_L2PL(target_img, lossfun, nsteps1=150, nsteps2=150, return_stat=True):
    #tsr_target = target_img.astype(float)#/255.0
    # assume the target img has been resized prior to this
    # rsz_target = resize(tsr_target, (256, 256), anti_aliasing=True)
    tsr_target = torch.from_numpy(target_img * 2.0 - 1).float().cuda()  # centered to be [-1, 1]
    tsr_target = tsr_target.unsqueeze(0).permute([0, 3, 1, 2])
    # assert size of this image is 256 256
    code = 4*np.random.randn(4096)
    code = code.reshape(-1, 4096)
    feat = torch.from_numpy(code).float().requires_grad_(True)
    feat.cuda()
    optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for i in range(nsteps1):
        optimizer.zero_grad()
        img = visualize(G, feat, percept_loss=True)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better?
        loss = L2loss(img, tsr_target)
        loss.backward()
        optimizer.step()
    print("L2 step%d" % i, loss.item())
    loss_col = []
    norm_col = []
    for i in range(nsteps2):
        optimizer.zero_grad()
        img = visualize(G, feat, percept_loss=True)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better?
        loss = lossfun(img, tsr_target)
        loss.backward()
        optimizer.step()
        norm_col.append(feat.norm().detach().item())
        loss_col.append(loss.detach().item())
    print("PerceptLoss step%d" % i, loss.item())
    img = visualize(G, feat, percept_loss=False)
    if return_stat:
        return feat.detach(), img.detach(), loss_col, norm_col
    else:
        return feat.detach(), img.detach()

def resize_center_crop(curimg, final_L=256):
    if len(curimg.shape) == 2:
        curimg = np.repeat(curimg[:, :, np.newaxis], 3, 2)
    H, W, _ = curimg.shape
    if H <= W:
        newW = round(float(W) / H * final_L)
        rsz_img = resize(curimg, (final_L, newW))
        offset = (newW - final_L) // 2
        fin_img = rsz_img[:, offset:offset + final_L, :]
    else:
        newH = round(float(H) / W * final_L)
        rsz_img = resize(curimg, (newH, final_L))
        offset = (newH - final_L) // 2
        fin_img = rsz_img[offset:offset + final_L, :, :]
    return fin_img
#%%
# savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\PhotoRealism"
#%%
# def percept_loss(img, target, net=percept_net):
#     return net.forward(img.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))
imageroot = r"E:\Datasets\ImageNet"
savedir = r"E:\Datasets\ImageTranslation\GAN_real"

imageroot = r"/scratch/binxu/Datasets/imagenet"
savedir = r"/scratch/binxu/Datasets/ImageTranslation/GAN_real"
os.makedirs(join(savedir, "A"), exist_ok=True)
os.makedirs(join(savedir, "B"), exist_ok=True)
#%%
#%%
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
# zcode, fitimg, loss_col, norm_col = img_backproj_PL(target_img, percept_net.forward, nsteps=nsteps, return_stat=True)
#%%
csr_min, csr_max = 1, 1001
if len(sys.argv) > 1:
    csr_min = int(sys.argv[1])
    csr_max = int(sys.argv[2])
#%
print("Embedding ImageNet images from %d - %d" % (csr_min, csr_max))
# import argparse
# argparse.ArgumentParser
Asuffix = "_L2PL"
nsteps = 300
nsteps1 = 300
nsteps2 = 250
Bsize = 100
os.makedirs(join(savedir, "A"+Asuffix), exist_ok=True)
csr = csr_min  #1
# csr = 34
while csr < csr_max:
    loss_arr = []
    norm_arr = []
    code_arr = []
    idx_arr = []
    t0 = time()
    for i in range(csr, csr + Bsize):
        try:
            curimg = imread(join(imageroot, "ILSVRC2012_val_%08d.JPEG" % i))
        except:
            break
        target_img = resize_center_crop(curimg, 256)
        imsave(join(savedir, "B", "val_crop_%08d.JPEG" % i), target_img)
        zcode, fitimg, loss_col, norm_col = img_backproj_L2PL(target_img, percept_net.forward, nsteps1=nsteps1, nsteps2=nsteps2, return_stat=True)
        imsave(join(savedir, "A" + Asuffix, "val_crop_%08d.JPEG" % i), fitimg.cpu().numpy())
        idx_arr.append(i)
        code_arr.append(zcode.cpu().numpy()[0,:])
        norm_arr.append(norm_col)
        loss_arr.append(loss_col)
        print(time() - t0, "sec, %d img" % i)
    csr += Bsize
    idx_arr = np.array(idx_arr)
    code_arr = np.array(code_arr)
    norm_arr = np.array(norm_arr)
    loss_arr = np.array(loss_arr)
    np.savez(join(savedir, "A" + Asuffix, "codes_%05d-%05d.npz" % (idx_arr[0], idx_arr[-1])), idx_arr=idx_arr, code_arr=code_arr, norm_arr=norm_arr, loss_arr=loss_arr)


#%%
# It seems start from center is easier to fit a image even for gradient based algorithm
#
# plt.imshow(resize_center_crop(curimg, final_L=256))
# plt.show()