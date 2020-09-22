import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from time import time
from os.path import join
import sys
import lpips
from GAN_hessian_compute import hessian_compute, get_full_hessian
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from GAN_utils import loadBigGAN, loadStyleGAN2, BigGAN_wrapper
from hessian_analysis_tools import plot_spectra, compute_hess_corr
#%%
BGAN = loadBigGAN()
#%% Load the real eigenvalues for the BigGAN + LPIPS
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
data = np.load(join(figdir, "spectra_col.npz"))
eigvals_col = data['eigval_col']
eva_mean = eigvals_col.mean(axis=0)
#%% BigGAN
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
plt.figure(figsize=[5, 4])
Ln = len(BGAN.generator.layers)
eva00 = np.load(join(datadir, "eig_gen_z.npz"))["eva"]  #, H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00 / eva00.max())[::-1], label="gen_z")
for blocki in range(Ln):
    eva00 = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"]
    plt.plot(np.log10(eva00 / eva00.max())[::-1], color=cm.jet((blocki+1) / (Ln+1)),
             label=("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention")
plt.plot(np.log10(eva_mean / eva_mean.max()), color=cm.jet((Ln+1) / (Ln+1)),
             label="GAN")
plt.xlim([0, len(eva00)])
plt.xlabel("eigenvalue rank")
plt.ylabel("log(eig / eig max)")
plt.title("BigGAN Hessian Spectra of\n Intermediate Layers Compared")
plt.subplots_adjust(top=0.9)
plt.legend()
plt.savefig(join(figdir, "spectrum_med_Layers_norm_cmp.png"))
plt.savefig(join(figdir, "spectrum_med_Layers_norm_cmp.pdf"))
plt.show()
#%% Collect the H and eva evc into list.
npzlabel = ["gen_z"]+["genBlock%02d"%blocki for blocki in range(Ln)]
layernames = ["gen_z"] + [("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention" for blocki in range(Ln)]
H_col = []
evc_col = []
eva_col = []
for label in npzlabel:
    data = np.load(join(datadir, "eig_%s.npz" % label))
    H_col.append(data["H"])
    evc_col.append(data["evc"])
    eva_col.append(data["eva"])
#%%
i, j = 1, 13
vHv = np.diag(evc_col[i].T @ H_col[j] @ evc_col[i])
plt.figure(figsize=[7,3.5])
plt.subplot(121)
plt.scatter(np.log10(eva_col[i]), np.log10(vHv))
plt.subplot(122)
plt.scatter(np.log10(eva_col[i]), np.log10(vHv)-np.log10(eva_col[i]))
plt.show()
#%%
i, j = 1, 12
vHv = np.diag(evc_col[i].T @ H_col[j] @ evc_col[i])
np.polyfit(np.log10(eva_col[i]), np.log10(vHv), 1)
#%%
# from hessian_analysis_tools import plot_layer_consistency_example
fig=plot_layer_consistency_example(eva_col, evc_col, layernames, layeridx=[1, 2, 3, -1], figdir=figdir, titstr="BigGAN",
                                   savelabel="BigGAN")
fig.show()