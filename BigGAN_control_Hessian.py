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
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
datadir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN"
#%%
BGAN = loadBigGAN()
SD = BGAN.state_dict()
#%%
shuffled_SD = {}
for name, Weight in SD.items():
    idx = torch.randperm(Weight.numel())
    W_shuf = Weight.view(-1)[idx].view(Weight.shape)
    shuffled_SD[name] = W_shuf
#%%
torch.save(shuffled_SD, join(datadir, "BigGAN_shuffle.pt"))
    # print(name, Weight.shape, Weight.mean().item(), Weight.std().item())
#%%
BGAN_sf = loadBigGAN()
BGAN_sf.load_state_dict(torch.load(join(datadir, "BigGAN_shuffle.pt")))
G_sf = BigGAN_wrapper(BGAN_sf)
#%%
img = BGAN_sf.generator(torch.randn(1, 256).cuda()*0.05, 0.7).cpu()
ToPILImage()((1+img[0])/2).show()
#%%
#%%
def Hess_hook(module, fea_in, fea_out):
    print("hooker on %s"%module.__class__)
    ref_feat = fea_out.detach().clone()
    ref_feat.requires_grad_(False)
    L2dist = torch.pow(fea_out - ref_feat, 2).sum()
    L2dist_col.append(L2dist)
    return None
triali = 0
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit\BigGAN\ctrl_Hessians"
for triali in tqdm(range(1, 100)):
    feat = torch.cat((torch.randn(128).cuda(), BGAN_sf.embeddings.weight[:, triali].clone()), dim=0)
    eigvals, eigvects, H = hessian_compute(G_sf, feat, ImDist, hessian_method="BP", )
    np.savez(join(savedir, "eig_full_trial%d.npz"%(triali)), H=H, eva=eigvals, evc=eigvects,
                 feat=feat.cpu().detach().numpy())
    feat.requires_grad_(True)
    for blocki in [0, 3, 5, 8, 10, 12]:
        L2dist_col = []
        torch.cuda.empty_cache()
        H1 = BGAN_sf.generator.layers[blocki].register_forward_hook(Hess_hook)
        img = BGAN_sf.generator(feat, 0.7)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col[0], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(savedir, "eig_genBlock%02d_trial%d.npz"%(blocki, triali)), H=H00, eva=eva00, evc=evc00,
                 feat=feat.cpu().detach().numpy())

#%%