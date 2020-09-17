
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
from Hessian_analysis_tools import plot_spectra, compute_hess_corr
#%%
BGAN = loadBigGAN()
L2dist_col = []
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None
#%%
clas_vec = BGAN.embeddings.weight[:,120].detach().clone().unsqueeze(0)
feat = torch.cat((0.7*torch.randn(1, 128).cuda(), clas_vec), dim=1)
feat.requires_grad_(True)
#%%
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
L2dist_col = []
torch.cuda.empty_cache()
H1 = BGAN.generator.gen_z.register_forward_hook(Hess_hook)
img = BGAN.generator(feat, 0.7)
H1.remove()
T0 = time()
H00 = get_full_hessian(L2dist_col[0], feat)
eva00, evc00 = np.linalg.eigh(H00)
print("Spent %.2f sec computing" % (time() - T0))
np.savez(join(datadir, "eig_gen_z.npz"), H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00)[::-1])
plt.title("gen_z Linear Layer Spectra" % ())
plt.xlim([0, len(evc00)])
plt.savefig(join(datadir, "spectrum_gen_z.png" ))
plt.show()
#%%
for blocki in range(len(BGAN.generator.layers)):
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = BGAN.generator.layers[blocki].register_forward_hook(Hess_hook)
    img = BGAN.generator(feat, 0.7)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col[0], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(datadir, "eig_genBlock%02d.npz"%blocki), H=H00, eva=eva00, evc=evc00)
    plt.plot(np.log10(eva00)[::-1])
    plt.title("GenBlock %d Spectra" % (blocki,))
    plt.xlim([0, len(evc00)])
    plt.savefig(join(datadir, "spectrum_genBlock%d.png" % (blocki)))
    plt.show()
#%%
plt.figure(figsize=[7,8])
Ln = len(BGAN.generator.layers)
eva00 = np.load(join(datadir, "eig_gen_z.npz"))["eva"] #, H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00)[::-1], label="gen_z")
for blocki in range(Ln):
    eva00 = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"]
    plt.plot(np.log10(eva00)[::-1], color=cm.jet((blocki+1) / Ln),
             label=("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention")
plt.xlim([0, len(evc00)])
plt.xlabel("eigenvalue rank")
plt.ylabel("log(eig value)")
plt.title("BigGAN Hessian Spectra of Intermediate Layers Compared")
plt.subplots_adjust(top=0.9)
plt.legend()
plt.savefig(join(datadir, "spectrum_all_Layers_cmp.png"))
plt.show()
#%% Normalized
plt.figure(figsize=[7,8])
Ln = len(BGAN.generator.layers)
eva00 = np.load(join(datadir, "eig_gen_z.npz"))["eva"] #, H=H00, eva=eva00, evc=evc00)
plt.plot(np.log10(eva00/eva00.max())[::-1], label="gen_z")
for blocki in range(Ln):
    eva00 = np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"]
    plt.plot(np.log10(eva00/eva00.max())[::-1], color=cm.jet((blocki+1) / Ln),
             label=("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention")
plt.xlim([0, len(evc00)])
plt.xlabel("eigenvalue rank")
plt.ylabel("log(eig value/max(eig))")
plt.title("BigGAN Hessian Normalized Spectra of Intermediate Layers Compared")
plt.subplots_adjust(top=0.9)
plt.legend()
plt.savefig(join(datadir, "spectrum_all_Layers_maxnorm.png"))
plt.show()
#%%
eva_col = [np.load(join(datadir, "eig_gen_z.npz"))["eva"]]
evc_col = [np.load(join(datadir, "eig_gen_z.npz"))["evc"]]
for blocki in range(Ln):
    eva_col.append(np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["eva"])
    evc_col.append(np.load(join(datadir, "eig_genBlock%02d.npz" % blocki))["evc"])

#%%
# from Hessian_analysis_tools import plot_consistentcy_mat, compute_hess_corr, compute_vector_hess_corr, plot_layer_consistency_mat
corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, savelabel="BigGAN_all", figdir=datadir)
corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, savelabel="BigGAN_all", figdir=datadir)
#%%
layernames = ["gen_z"] + [("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention" for blocki in range(Ln)]
fig1, fig2, fig3 = plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, savelabel="BigGAN", figdir=datadir,
                                titstr="BigGAN", layernames=layernames)
