"""Visualize the Visual contents of the Hessian Eigenvectors"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
from geometry_utils import SLERP, LERP
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
#%%
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
from PIL import Image
from skimage.io import imsave
# go through spectrum in batch, and plot B number of axis in a row
def vis_eigen_frame(eigvect_avg, eigv_avg, ref_code=None, figdir=figdir, page_B=50):
    if ref_code is None:
        ref_code = np.zeros((1, 4096))
    csr = 0
    img_page = []
    for eigi in range(1, 4097):
        interp_codes = LERP(ref_code, eigvect_avg[:, -eigi], 11, (-200, 200))
        img_list = G.render(interp_codes)
        img_page.extend(img_list)
        if (eigi == csr + page_B) or eigi == 4096:
            mtg = build_montages(img_page, (256, 256), (11, page_B))[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%d-%d_%.e~%.e.jpg" %
                        (csr+1, eigi, eigv_avg[-csr-1], eigv_avg[-eigi])), np.uint8(mtg * 255.0))
            img_page = []
            print("Finish printing page eigen %d-%d"%(csr, eigi))
            csr = eigi
# imgs = visualize_np(G, interp_codes)
#%%
out_dir = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace"
with np.load(join(out_dir, "Pasu_Space_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)
#%%
out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
with np.load(join(out_dir, "Evolution_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Evol"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)