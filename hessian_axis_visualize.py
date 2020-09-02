"""Visualize the Visual contents of the Hessian Eigenvectors
Major function is `vis_eigen_frame` can print out the images along an axes.
"""
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
from geometry_utils import SLERP, LERP, LExpMap
from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
#%%
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
from PIL import Image
from skimage.io import imsave
# go through spectrum in batch, and plot B number of axis in a row
def vis_eigen_frame(eigvect_avg, eigv_avg, ref_code=None, figdir=figdir, page_B=50,
                    eig_rng=(0, 4096), maxdist=120, rown=7):
    if ref_code is None:
        ref_code = np.zeros((1, 4096))
    t0=time()

    csr = 0
    img_page = []
    for eigi in range(eig_rng[0]+1, eig_rng[1]+1):
        interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi], rown, (-maxdist, maxdist))
        img_list = G.render(interp_codes)
        img_page.extend(img_list)
        if (eigi == csr + page_B) or eigi == 4096:
            mtg = build_montages(img_page, (256, 256), (rown, page_B))[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%d-%d_%.e~%.e.jpg" %
                        (csr+1, eigi, eigv_avg[-csr-1], eigv_avg[-eigi])), np.uint8(mtg * 255.0))
            img_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(csr, eigi, time()-t0))
            csr = eigi
# imgs = visualize_np(G, interp_codes)
#%% Average Hessian for the Pasupathy Patches
out_dir = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace"
with np.load(join(out_dir, "Pasu_Space_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)
#%% Average hessian for the evolved images
out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
with np.load(join(out_dir, "Evolution_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigv_avg"]
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Evol"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)
#%% use the initial gen as reference code, do the same thing
out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
with np.load(join(out_dir, "Texture_Avg_Hess.npz")) as data:
    # H_avg = data["H_avg"]
    eigvect_avg = data["eigvect_avg"]
    eigv_avg = data["eigval_avg"]

code_path = r"D:\Generator_DB_Windows\init_population\texture_init_code.npz"
with np.load(code_path) as data:
    codes_all = data["codes"]
ref_code = codes_all.mean(axis=0,keepdims=True)
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Text"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir, ref_code=ref_code,
                maxdist=120, rown=7, eig_rng=(0, 4096))
