import numpy as np
from GAN_utils import upconvGAN
import torch
from GAN_hvp_operator import compute_hessian_eigenthings,get_full_hessian,GANForwardHVPOperator,GANHVPOperator
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from os.path import join
# import seaborn as sns
G = upconvGAN()
G.G.requires_grad_(False)
#%%
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None


#%%
feat = torch.randn(4096, requires_grad=True)

#%%
archdir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit"
#%%
layernames = [name for name, _ in G.G.named_children()]
eva_col = []
from time import time
for Li in [22, 23, 24]:#0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
    L2dist_col = []
    torch.cuda.empty_cache()
    H1 = G.G[Li].register_forward_hook(Hess_hook)
    img = G.visualize(feat)
    H1.remove()
    T0 = time()
    H10 = get_full_hessian(L2dist_col[0], feat)
    eva10, evc10 = np.linalg.eigh(H10)
    print("Layer %d, cost %.2f sec" % (Li, time() - T0))
    #%
    np.savez(join(archdir, "eig_Layer%d.npz" % (Li)), evc=evc10, eva=eva10)
    plt.plot(np.log10(eva10)[::-1])
    plt.title("Layer %d %s\n%s"%(Li, layernames[Li], G.G[Li].__repr__()))
    plt.xlim([0, 4096])
    plt.savefig(join(archdir, "spectrum_Layer%d.png" % (Li)))
    plt.show()
#%%
eva_col = []
for Li in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    with np.load(join(archdir, "eig_Layer%d.npz" % (Li))) as data:
        eva_col.append(data["eva"].copy())
#%%
np.savez(join(archdir, "spect_each_layer.npz"), eva_col=np.array(eva_col))
#%%
plt.figure(figsize=[7, 9])
for eva, name in zip(eva_col, layernames):
    plt.plot(np.log10(eva)[::-1], label=name)
plt.legend()
plt.xlim([0, 4096])
plt.title("Spectrum of Jacobian / Hessian Up to Each Layer")
plt.savefig(join(archdir, "spectrum_Layer_all.png"))
plt.savefig(join(archdir, "spectrum_Layer_all.pdf"))
plt.show()
