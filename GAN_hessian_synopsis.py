import os
import re
from glob import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from time import time
from os.path import join
#%% BigGAN
BGfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
BGdatadir = r"E:\Cluster_Backup\BigGANH"
eigval_col = []
eigvals_clas_col = []
eigvals_nois_col = []
for class_id in range(1000):
    fn = "Hess_cls%d.npz"%class_id
    data = np.load(join(BGdatadir, fn))
    eigvals = data["eigvals"]
    eigval_col.append(eigvals.copy())
    eigvals_clas = data["eigvals_clas"]
    eigvals_clas_col.append(eigvals_clas.copy())
    eigvals_nois = data["eigvals_nois"]
    eigvals_nois_col.append(eigvals_nois.copy())

eigval_col = np.array(eigval_col)[:,::-1]
eigvals_clas_col = np.array(eigvals_clas_col)[:,::-1]
eigvals_nois_col = np.array(eigvals_nois_col)[:,::-1]
np.savez(join(BGfigdir, "spectra_col.npz"),eigval_col=eigval_col,
							eigvals_clas_col=eigvals_clas_col,
							eigvals_nois_col=eigvals_nois_col,)
#%% BigBiGAN
BBGdir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
BBGfigdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
npzlist = glob(join(BBGdir, "*.npz"))
npzpattern = re.compile("Hess_norm(\d.*)_(\d\d\d)")
npzlist = sorted(npzlist)
H_record = []
eigval_col = []
code_all = []
for idx in range(len(npzlist)):
    npzname = os.path.split(npzlist[idx])[1]
    parts = npzpattern.findall(npzname)[0]
    vecnorm = int(parts[0])
    RNDid = int(parts[1])
    data = np.load(npzlist[idx])
    eigvals = data['eigvals']
    code = data['vect']
    realnorm = np.linalg.norm(code)
    code_all.append(code)
    eigval_col.append(eigvals)
    H_record.append([npzname, vecnorm, realnorm, RNDid,
         eigvals[-1], eigvals[-5], eigvals[-10], eigvals[-20], eigvals[-40], eigvals[-60]])

H_rec_tab = pd.DataFrame(H_record, columns=["npz","norm","realnorm","RND",] + ["eig%d"%idx for idx in [1,5,10,20,40,60]])
H_rec_tab = H_rec_tab.sort_values("norm")
H_rec_tab = H_rec_tab.reset_index()
# H_rec_tab.to_csv(join(BBGfigdir, "Hess_record.csv"))
eigval_col = np.array(eigval_col)[:, ::-1]
np.savez(join(BBGfigdir, "spectra_col.npz"), eigval_col=eigval_col, H_record=H_record)
#%% FC6GAN
FC6figdir = r"E:\Cluster_Backup\FC6GAN\summary"
FC6dir = r"E:\Cluster_Backup\FC6GAN"
labeldict = {"BP": "bpfull", "BackwardIter": "bkwlancz", "ForwardIter": "frwlancz"}
method = "BP"
labstr = labeldict[method]
space = "evol"
eigvals_col = []
code_all = []
for idx in range(284): # Note load it altogether is very slow, not recommended
    fn = "%s_%03d_%s.npz" % (space, idx, labstr)
    data = np.load(join(FC6dir, fn))
    eigvals = data['eigvals']
    code = data['code']
    eigvals_col.append(eigvals.copy())
    code_all.append(code.copy())
#%
eigvals_col = np.array(eigvals_col)[:, ::-1]
code_all = np.array(code_all)
np.savez(join(FC6figdir, "spectra_col_evol.npz"), eigval_col=eigval_col, )

#%% StyleGAN2
SGdir = r"E:\Cluster_Backup\StyleGAN2"
SGfigdir = "E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
#%
npzpaths = glob(join(Hdir, "*.npz"))
npzfns = [path.split("\\")[-1] for path in npzpaths]
eigval_col = []
for fn, path in zip(npzfns, npzpaths):
    data = np.load(path)
    evas = data["eigvals"]
    eigval_col.append(evas)
eigval_col = np.array(eigval_col)
#%% Synopsis
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
spaceD = [4096, 256, 120, 512, 512]
GANlist = ["FC6", "BigGAN", "BigBiGAN", "StyleGAN-face", "StyleGAN-cat"]
fnlist = ["FC6GAN\\spectra_col_evol.npz",
          "BigGAN\\spectra_col.npz",
          "BigBiGAN\\spectra_col_FFHQ512.npz",
          "StyleGAN2\\spectra_col_stylegan2-cat-config-f.npz"]
plt.figure()
for i, GAN in enumerate(GANlist):
    with np.load(join(rootdir, fnlist[i])) as data:
        eigval_col = data["eigval_col"]
    eva_mean = eigval_col.mean(axis=0)
    eva_std = eigval_col.std(axis=0)
    eva_lim = np.percentile(eigval_col, [5, 95], axis=0)

    plt.plot(np.arange(spaceD[i])/spaceD[i], eva_mean / eva_mean.max(), alpha=0.7)  # , eigval_arr.std(axis=0)
    plt.fill_between(np.arange(spaceD[i])/spaceD[i], eva_lim[0, :] / eva_mean.max(), eva_lim[1, :] / eva_mean.max(), alpha=0.5, label=GAN)
plt.show()
