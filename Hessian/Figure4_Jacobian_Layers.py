import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from tqdm import tqdm
from os.path import join
from GAN_utils import loadBigGAN, BigGAN_wrapper

#%%
BGAN = loadBigGAN()
Ln = len(BGAN.generator.layers)
G = BigGAN_wrapper(BGAN)
#%% Load the real eigenvalues for the BigGAN + LPIPS
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
data = np.load(join(figdir, "spectra_col.npz"))
eigvals_col = data['eigval_col']
eva_mean = eigvals_col.mean(axis=0)
data = np.load(join(figdir, "H_avg_1000cls.npz"))
eva_Havg, evc_Havg, Havg = data['eigvals_avg'], data['eigvects_avg'], data['H_avg']
#%% BigGAN
plt.figure(figsize=[5, 4])
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
layernames = ["gen_z"] + [("GenBlock%02d" % blocki) if blocki!=8 else "SelfAttention" for blocki in range(Ln)] \
                + ["Full"]
H_col = []
evc_col = []
eva_col = []
for label in npzlabel:
    data = np.load(join(datadir, "eig_%s.npz" % label))
    H_col.append(data["H"])
    evc_col.append(data["evc"])
    eva_col.append(data["eva"])
H_col.append(Havg)
eva_col.append(eva_Havg)
evc_col.append(evc_Havg)
#%%
from Hessian.hessian_analysis_tools import plot_layer_consistency_example
fig = plot_layer_consistency_example(eva_col, evc_col, layernames, layeridx=[1, 9, 13, -1], figdir=figdir,
                                    titstr="BigGAN", savelabel="BigGAN")
fig.show()
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
embedmat = BGAN.embeddings.weight.cpu().numpy()
#%%
from pytorch_pretrained_biggan import truncated_noise_sample
from Hessian.hessian_axis_visualize import vis_eigen_explore_row
classid = 287
# noisevec = truncated_noise_sample(1, 128, 0.6)
# ref_code = np.concatenate((noisevec, embedmat[:, classid:classid+1].T), axis=1)
vis_eigen_explore_row(ref_code, evc_Havg, eva_Havg, G, figdir=figdir, namestr="BigGAN_cls%d"%classid, indivimg=True,
     eiglist=[9,16,64], maxdist=0.2, rown=5, sphere=True, )
