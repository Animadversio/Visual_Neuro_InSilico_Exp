"""
Use the fc6 fc7 GANs implemented in pure python from GAN_utils, and hessian package in pytorch.
to compute the Hessian and eigen decomposition. 
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from hessian import hessian
import sys
sys.path.append(r"D:\Github\PerceptualSimilarity")
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
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
    feat = torch.tensor(z, requires_grad=False).cuda()
    img_ref = G.visualize(feat,)  # forward the feature vector through the GAN
    # Perturbation vector for gradient
    perturb_vec = torch.zeros((1, 4096),dtype=torch.float32).requires_grad_(True).cuda()
    # perturb_vec = 0.00001*torch.randn((1, 4096),dtype=torch.float32).requires_grad_(True).cuda()
    img_pertb = G.visualize(feat + perturb_vec)
    d_sim = percept_loss.forward(img_ref, img_pertb)
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
G.requires_grad_(False).cuda()
model_vgg.requires_grad_(False).cuda()
z = torch.randn(1, 4096) * 3
from time import time
t0 = time()
H, eigval, eigvec, gradient, d_sim = sim_hessian_computation(z, model_vgg)
print("Spent %d sec"%(time() - t0))  # compute this used 565 sec
#%%
t0 = time()
H2, eigval2, eigvec2, gradient2, d_sim2 = sim_hessian_computation(z, model_vgg)
print("Spent %d sec"%(time() - t0))  # compute this used 557 sec
#%%
from hessian_eigenthings import compute_hessian_eigenthings