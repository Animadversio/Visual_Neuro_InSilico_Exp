"""
Visualize how different image metrics affect results of image fitting!
L2 and L1 seems not easy to optimize
"""
import torch
from torch_net_utils import load_generator
from skimage.transform import resize
from imageio import imread
import matplotlib.pylab as plt
import numpy as np

G = load_generator("fc6")
G.requires_grad_(False)
G.cuda()
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1))
#%%
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
# model = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
# percept_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
target_img = imread(r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\Cat.jpg")
#%%
from os.path import join
savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\PhotoRealism"
#%%
def visualize(G, code, mode="cuda", percept_loss=True):
    """Do the De-caffe transform (Validated)
    works for a single code """
    if mode == "cpu":
        blobs = G(code)
    else:
        blobs = G(code.cuda())
    out_img = blobs['deconv0']  # get raw output image from GAN
    if mode == "cpu":
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    else:
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
#%%
def img_backproj_L2PL(target_img, lossfun, lossfun1=L2loss, nsteps1=150, nsteps2=150, return_stat=True, init_norm=256):
    #tsr_target = target_img.astype(float)#/255.0
    # assume the target img has been resized prior to this
    rsz_target = resize(target_img, (256, 256), anti_aliasing=True)
    tsr_target = torch.from_numpy(rsz_target * 2.0 - 1).float().cuda()  # centered to be [-1, 1]
    tsr_target = tsr_target.unsqueeze(0).permute([0, 3, 1, 2])
    # assert size of this image is 256 256
    code = init_norm / 64 *np.random.randn(4096)
    code = code.reshape(-1, 4096)
    feat = torch.from_numpy(code).float().requires_grad_(True)
    feat.cuda()
    optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for i in range(nsteps1):
        optimizer.zero_grad()
        img = visualize(G, feat, percept_loss=True)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better?
        loss = lossfun1(img, tsr_target)
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
#%%
# model = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
percept_net = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=1, gpu_ids=[0])
def percept_loss(img, target, net=percept_net):
    return net.forward(img.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))
#%%
# d_sim = percept_loss.forward(resz_ref_img, resz_out_img)
zcode, fitimg = img_backproj(target_img, percept_loss)
# zcode, fitimg = img_backproj(target_img, L1loss)

#%%
for nsteps in [1000]: #%150, 250, 500,
    zcode, fitimg = img_backproj(target_img, percept_loss, nsteps=nsteps)
    label = "alexPL_step%d" % (nsteps)
    plt.figure(figsize=[4, 4.5])
    plt.imshow(fitimg.cpu().numpy())
    plt.title(label)
    plt.axis("off")
    plt.savefig(join(savedir, "Embed_%s.jpg"%label))
    plt.show()

#%%
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
def percept_loss(img, target, net=percept_net):
    return net.forward(img.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))
#%%
for nsteps in [1000]: #%150, 250, 500,
    zcode, fitimg = img_backproj(target_img, percept_loss, nsteps=nsteps)
    label = "VGGPL_step%d" % (nsteps)
    plt.figure(figsize=[4, 4.5])
    plt.imshow(fitimg.cpu().numpy())
    plt.title(label)
    plt.axis("off")
    plt.savefig(join(savedir, "Embed_%s.jpg"%label))
    plt.show()
#%%
for nsteps in [150, 250, 500]:
    zcode, fitimg = img_backproj(target_img, L2loss, nsteps=nsteps)
    label = "L2_step%d" % (nsteps)
    plt.figure(figsize=[4, 4.5])
    plt.imshow(fitimg.cpu().numpy())
    plt.title(label)
    plt.axis("off")
    plt.savefig(join(savedir, "Embed_%s.jpg"%label))
    plt.show()
#%%
for nsteps in [150, 250, 500]:
    zcode, fitimg = img_backproj(target_img, L1loss, nsteps=nsteps)
    label = "L1_step%d" % (nsteps)
    plt.figure(figsize=[4, 4.5])
    plt.imshow(fitimg.cpu().numpy())
    plt.title(label)
    plt.axis("off")
    plt.savefig(join(savedir, "Embed_%s.jpg"%label))
    plt.show()
#%%
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
def percept_loss(img, target, net=percept_net):
    return net.forward(img.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))
#%%
zcode, fitimg, loss_col, norm_col = img_backproj(target_img, percept_loss, nsteps=1000, return_stat=True)
#%%
label = "VGGPLtr_step%d" % (nsteps)
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
#(label)
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
#(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%
percept_net = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=1, gpu_ids=[0])
def percept_loss(img, target, net=percept_net):
    return net.forward(img.unsqueeze(0).permute([0, 3, 1, 2]), target.unsqueeze(0).permute([0, 3, 1, 2]))
nsteps = 1000
zcode, fitimg, loss_col, norm_col = img_backproj(target_img, percept_loss, nsteps=nsteps, return_stat=True)
#%
label = "alexPL_step%d" % (nsteps)
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
#(label)
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
#(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%nsteps\
nsteps = 1000
zcode, fitimg, loss_col, norm_col = img_backproj(target_img, L2loss, nsteps=nsteps, return_stat=True)
#%
label = "L2_step%d" % (nsteps)
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
#(label)
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
#(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%
nsteps = 1000
zcode, fitimg, loss_col, norm_col = img_backproj(target_img, L1loss, nsteps=nsteps, return_stat=True)
#%
label = "L1_step%d" % (nsteps)
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
#(label)
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
#(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%
nsteps = 500
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
zcode, fitimg, loss_col, norm_col = img_backproj_PL(target_img, percept_net.forward, nsteps=nsteps, return_stat=True)
label = "vggPLtr_step%d" % (nsteps)
plt.figure(figsize=[4, 4.5])
plt.imshow(fitimg.cpu().numpy())
plt.title(label)
plt.axis("off")
plt.savefig(join(savedir, "Embed_%s.jpg"%label))
plt.show()
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%
nsteps1 = 300
nsteps2 = 1000
init_norm = 64
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
zcode, fitimg, loss_col, norm_col = img_backproj_L2PL(target_img, percept_net.forward, init_norm=init_norm, nsteps1=nsteps1, nsteps2=nsteps2, return_stat=True)
label = "vggL2P2_lownorm_step%d-%d" % (nsteps1, nsteps2)
plt.figure(figsize=[4, 4.5])
plt.imshow(fitimg.cpu().numpy())
plt.title(label)
plt.axis("off")
plt.savefig(join(savedir, "Embed_%s.jpg"%label))
plt.show()
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%%
nsteps1 = 300
nsteps2 = 1000
init_norm = 256
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
zcode, fitimg, loss_col, norm_col = img_backproj_L2PL(target_img, percept_net.forward, init_norm=init_norm, nsteps1=nsteps1, nsteps2=nsteps2, return_stat=True)
label = "vggL2P2_highnorm_step%d-%d" % (nsteps1, nsteps2)
plt.figure(figsize=[4, 4.5])
plt.imshow(fitimg.cpu().numpy())
plt.title(label)
plt.axis("off")
plt.savefig(join(savedir, "Embed_%s.jpg"%label))
plt.show()
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()
#%% L1 PL
nsteps1 = 300
nsteps2 = 1000
init_norm = 64
percept_net = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
zcode, fitimg, loss_col, norm_col = img_backproj_L2PL(target_img, percept_net.forward, lossfun1=L1loss, init_norm=init_norm, nsteps1=nsteps1, nsteps2=nsteps2, return_stat=True)
label = "vggL1P2_lownorm_step%d-%d" % (nsteps1, nsteps2)
plt.figure(figsize=[4, 4.5])
plt.imshow(fitimg.cpu().numpy())
plt.title(label)
plt.axis("off")
plt.savefig(join(savedir, "Embed_%s.jpg"%label))
plt.show()
plt.figure(figsize=[8, 8.5])
plt.subplot(2, 2, 1)
plt.imshow(fitimg.cpu().numpy())
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(target_img)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.plot(loss_col)
plt.ylabel("Loss")
plt.xlabel("steps")
plt.subplot(2, 2, 4)
plt.plot(norm_col)
plt.ylabel("code norm")
plt.xlabel("steps")
plt.suptitle(label)
plt.savefig(join(savedir, "Embed_traj_%s.jpg"%label))
plt.show()