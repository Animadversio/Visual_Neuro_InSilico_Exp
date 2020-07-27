"""
Compute Hessian matrix at different center images / codes.
Average them and do Eigen-decomposition to get the eigenvectors / basis.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings
#
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from geometry_utils import SLERP, LERP
#%%
from FeatLinModel import FeatLinModel, get_model_layers
import sys
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
sys.path.append(r"D:\Github\PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN, visualize_np
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch

# import torchvision as tv
# # VGG = tv.models.vgg16(pretrained=True)
# alexnet = tv.models.alexnet(pretrained=True).cuda()
# for param in alexnet.parameters():
#     param.requires_grad_(False)

#%% Load the pasupathy codes
from scipy.io import loadmat
code_path = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\pasu_fit_code.mat"
out_dir = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace"
data = loadmat(code_path)
pasu_codes = data['pasu_code']
#%% Compute the Hessian around a certain Pasupathy image.
t0 = time()
for imgi, code in enumerate(pasu_codes[:, :]):
    feat = torch.from_numpy(code[np.newaxis, :])
    feat.requires_grad_(False)
    metricHVP = GANHVPOperator(G, feat, model_squ)
    eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True)
    print("Finish computing img %d %.2f sec passed, max %.2e min %.2e 10th %.1e 50th %.e 100th %.1e" % (imgi,
        time() - t0, max(np.abs(eigvals)), min(np.abs(eigvals)), eigvals[-10], eigvals[-50], eigvals[-100]))
    np.savez(join(out_dir, "pasu_%03d.npz" % imgi), eigvals=eigvals, eigvects=eigvects, code=code)
#%%
imgi, imgj = 0, 1
with np.load(join(out_dir, "pasu_%03d.npz" % imgi)) as data:
    basisi = data["eigvects"]
    eigvi = data["eigvals"]
    codei = data["code"]

with np.load(join(out_dir, "pasu_%03d.npz" % imgj)) as data:
    basisj = data["eigvects"]
    eigvj = data["eigvals"]
    codej = data["code"]

#%%
from sklearn.cross_decomposition import CCA
def cca_subspace(X, Y, n_comp=50, **kwargs):
    """
    :param X, Y: should be N-by-p, N-by-q matrices, N is the dimension for the whole space, p, q are number of basis
                 vectors (Note p, q functions as number of features to be recombined, while N functions as number of
                 sampled). CCA will maximize
    :param n_comp: a integer, how many components we want to create and compare.
    :return: cca_corr, n_comp-by-n_comp matrix
       X_c, Y_c will be the linear mapped version of X, Y with shape  N-by-n_comp, N-by-n_comp shape
       cc_mat is the
    """
    cca = CCA(n_components=n_comp, **kwargs)
    X_c, Y_c = cca.fit_transform(X, Y)
    ccmat = np.corrcoef(X_c, Y_c, rowvar=False)
    cca_corr = np.diag(ccmat[n_comp:, :n_comp])  # slice out the cross corr part
    return cca_corr, cca

#%%
%%time
t0 = time()
cca_corr200, cca200 = cca_subspace(basisi[-200:, :].T, basisj[-200:, :].T, n_comp=200, max_iter=1000)
print(time() - t0) # 57.68 sec / 86.67
t0 = time()
cca_corr100, cca100 = cca_subspace(basisi[-100:, :].T, basisj[-100:, :].T, n_comp=100, max_iter=1000)
print(time() - t0) # 11.68 sec
t0 = time()
cca_corr50, cca50 = cca_subspace(basisi[-50:, :].T, basisj[-50:, :].T, n_comp=50)
print(time() - t0) # 2 sec
#%%
t0 = time()
cca_corr400, cca400 = cca_subspace(basisi[-400:, :].T, basisj[-400:, :].T, n_comp=400, max_iter=1000)
print(time() - t0) #
t0 = time()
cca_corr400_ctr, cca400_ctr = cca_subspace(np.random.randn(400, 4096).T, np.random.randn(400, 4096).T, n_comp=400, max_iter=1000)
print(time() - t0)
# 98s
#%%
t0 = time()
cca_corr200_ctr, cca200_ctr = cca_subspace(np.random.randn(200, 4096).T, np.random.randn(200, 4096).T, n_comp=200, max_iter=1000)
cca_corr100_ctr, cca100_ctr = cca_subspace(np.random.randn(100, 4096).T, np.random.randn(100, 4096).T, n_comp=100, max_iter=1000)
cca_corr50_ctr, cca50_ctr = cca_subspace(np.random.randn(50, 4096).T, np.random.randn(50, 4096).T, n_comp=50)
print(time() - t0) # 2 sec
# 509 s
#%%
plt.figure()
plt.plot(cca_corr400, label="top400 eig")
plt.plot(cca_corr200, label="top200 eig")
plt.plot(cca_corr100, label="top100 eig")
plt.plot(cca_corr50, label="top50 eig")
plt.plot(cca_corr400_ctr, label="random 400")
plt.plot(cca_corr200_ctr, label="random 200")
plt.plot(cca_corr100_ctr, label="random 100")
plt.plot(cca_corr50_ctr, label="random 50")
plt.title("Top Eigen Space Is Shared\nMeasured by CCA")
plt.ylabel("Correlation Value")
plt.xlabel("CC #")
plt.legend()
plt.savefig(join(out_dir, "Pasu12_Shared_EigenSpace.jpg"))
plt.show()

#%% Visualize effect of the Hessian Eigen vectors on the codes
cutoff = 800
visualize_np(G, pasu_codes[:20,:]@basisi[:,:].T@basisi[:,:], (4,5))
#%%
imgn = pasu_codes.shape[0]
basis_col = []
eigv_col = []
for imgi in range(imgn):
    with np.load(join(out_dir, "pasu_%03d.npz" % imgi)) as data:
        basisi = data["eigvects"]
        eigvi = data["eigvals"]
        basis_col.append(basisi)
        eigv_col.append(eigvi)
#%% Averaged Hessian matrix
avg_Hess = np.zeros((4096, 4096))
for imgi in range(imgn):
    basisi = basis_col[imgi]
    eigvi = eigv_col[imgi]
    avg_Hess = avg_Hess + (basisi.T * eigvi[np.newaxis, :] @ basisi)

avg_Hess = avg_Hess / imgn
#%% And then do Decomposition
eigv_avg, eigvect_avg = np.linalg.eigh(avg_Hess)
#%%
np.savez(join(out_dir, "Pasu_Space_Avg_Hess.npz"), H_avg=avg_Hess, eigv_avg=eigv_avg, eigvect_avg=eigvect_avg)
#%%
proj_rang = range(2000,3500)
proj_op = eigvect_avg[proj_rang,:].T @ eigvect_avg[proj_rang,:]
visualize_np(G, SLERP(pasu_codes[0:1,:]@proj_op, pasu_codes[10:11,:]@proj_op, 11),)
#%%
proj_coef = pasu_codes @ basisi.T
proj_coef[:, :400] = np.mean(proj_coef[:, :400], axis=0)
recon_code = proj_coef @ basisi
visualize_np(G, recon_code, (12, 16))
#%%
visualize_np(G, pasu_codes @ basis_col[20].T @ basis_col[20], (12, 16))
#%%
visualize_np(G, pasu_codes @ basis_col[20].T @ basis_col[20], (12, 16))
#%%
visualize_np(G, pasu_codes @ eigvect_avg[:,-100:] @ eigvect_avg[:,-100:].T, (12, 16))
#%%
proj_coef = pasu_codes @ eigvect_avg
proj_coef[:, :-150] = np.mean(proj_coef[:, :-150], axis=0)
recon_code = proj_coef @ eigvect_avg.T
visualize_np(G, recon_code, (12, 16))
#%%
pasu_mean = np.mean(pasu_codes,axis=0,keepdims=True)
visualize_np(G, pasu_mean+(pasu_codes - pasu_mean)@ eigvect_avg[:,-200:] @ eigvect_avg[:,-200:].T, (12, 16))
#%%
pasu_codes_rd = pasu_mean+(pasu_codes - pasu_mean)@ eigvect_avg[:,-200:] @ eigvect_avg[:,-200:].T
from sklearn.decomposition.pca import PCA
code_PCA = PCA().fit(pasu_codes_rd)
#%%
visualize_np(G, LERP(pasu_codes_rd[0:1,:], pasu_codes_rd[50:51,:], 11),)
#%%
evo_code_path = r"N:\Stimuli\2019-12-Evolutions\2019-12-30-Beto-03\2019-12-30-12-30-57\block032_thread000_code.mat"
data = loadmat(evo_code_path)
evo_codes = data['codes']
#%%
# visualize_np(G, evo_codes @ basisi.T @ basisi, (6, 7))
visualize_np(G, evo_codes @ eigvect_avg[:,-800:] @ eigvect_avg[:,-800:].T, (6, 7))
#%%
evo_mean = evo_codes.mean(axis=0)
evo_codes_rd = evo_mean + (evo_codes - evo_mean) @ eigvect_avg[:,-200:] @ eigvect_avg[:,-200:].T
visualize_np(G, evo_codes_rd, (7, 6))
#%%
visualize_np(G, evo_codes, (7, 6))
#%% Compute the null space for the evolved images
from os import listdir
from os.path import isdir
from glob import glob
out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
evol_dir = r"N:\Stimuli\2019-12-Evolutions"
expnames = sorted(listdir(evol_dir))
expnames = [expname for expname in expnames if ("Alfa" in expname) or ("Beto" in expname)]
expname = expnames[1]
for expi, expname in enumerate(expnames):
    subname = [fn for fn in listdir(join(evol_dir, expname)) if isdir(join(evol_dir, expname, fn))]
    if len(subname) > 0:
        matlist = sorted(glob(join(evol_dir, expname, subname[0], "*.mat")))
        evo_code_path = matlist[-2]
        data = loadmat(evo_code_path)
        evo_codes = data['codes']
        code = evo_codes[0:1, :]
        nameparts = evo_code_path.split("\\")
        t0 = time()
        feat = torch.from_numpy(code)
        feat.requires_grad_(False)
        metricHVP = GANHVPOperator(G, feat, model_squ)  # using backward Iterative method to compute Hessian.
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=800, use_gpu=True)
        print("Finish computing expi %d %.2f sec passed, max %.2e min %.2e 10th %.1e 50th %.e 100th %.1e (norm %.1f)"
          % (expi, time() - t0,max(np.abs(eigvals)), min(np.abs( eigvals)), eigvals[ -10], eigvals[-50],eigvals[-100],
          norm(code)))
        np.savez(join(out_dir, "evol_%03d.npz" % expi), eigvals=eigvals, eigvects=eigvects, code=code,
                 source=evo_code_path)
#%% Average the Hessian across the calculations
avg_Hess_evo = np.zeros((4096, 4096))
for expi in range(len(expnames)):
    with np.load(join(out_dir, "evol_%03d.npz" % expi)) as data:
        eigvects = data["eigvects"]
        eigvals = data["eigvals"]
        avg_Hess_evo += (eigvects.T * eigvals[np.newaxis, :] @ eigvects)
avg_Hess_evo /= len(expnames)
%time eigv_avg_evo, eigvect_avg_evo = np.linalg.eigh(avg_Hess_evo)
#%% Save the averaged hessian.
np.savez(join(out_dir, "Evolution_Avg_Hess.npz"), H_avg=avg_Hess_evo, eigv_avg=eigv_avg_evo,
         eigvect_avg=eigvect_avg_evo)
#%%
from os.path import join
savedir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
code_all = []
exp_src = []
for expi in range(284):#len(expnames)
    with np.load(join(savedir, "evol_%03d.npz" % expi)) as data:
        code = data["code"]
        source = data["source"]
        code_all.append(code.copy())
        exp_src.append(source.copy())
#%%
code_arr = np.concatenate(tuple(code_all), axis=0)
exp_srcs = [str(src) for src in exp_src]
np.savez(join(savedir, "evol_codes_all.npz"), code_arr=code_arr, exp_srcs=exp_srcs)