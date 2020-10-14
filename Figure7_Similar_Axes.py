from hessian_axis_visualize import vis_eigen_frame, vis_eigen_action, vis_distance_curve, vis_eigen_explore
from hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, plot_consistentcy_mat, \
    plot_consistency_hist, plot_consistency_example, plot_spectra
from GAN_utils import loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, StyleGAN2_wrapper, \
    loadStyleGAN, StyleGAN_wrapper, upconvGAN, PGGAN_wrapper, loadPGGAN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
figsumdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\Figure7"
"""Note the loading and visualization is mostly deterministic, reproducible."""
#%% StyleGAN2 Face 1024
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "stylegan2-ffhq-config-f"
modelsnm = "Face1024"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN2_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg = data["eva_avg"], data["evc_avg"]#, data["feat_col"]
#%%
refvec = 0 * torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu()).show()
#%%
vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec.cpu(), figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
                    eiglist=None, eig_rng=(0, 20), maxdist=0.0005, rown=5, sphere=False, )



#%%
from geometry_utils import LExpMap, SExpMap
from build_montages import build_montages
def vis_eigen_explore(ref_code, eigvect_avg, eigv_avg, G, figdir="", RND=None, namestr="", transpose=True, eiglist=[1,2,4,7,16], maxdist=120, rown=5, sphere=False, ImDist=None, distrown=19, scaling=None):
    """This is small scale version of vis_eigen_frame + vis_distance_vector """
    if RND is None: RND = np.random.randint(10000)
    if eiglist is None: eiglist = list(range(len(eigv_avg)))
    if scaling is None: scaling = np.ones(len(eigv_avg))
    t0 = time()
    codes_page = []
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        scaler = scaling[idx]
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist*scaler, maxdist*scaler))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist*scaler, maxdist*scaler))
        codes_page.append(interp_codes)
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    img_page = G.render(codes_all)
    mtg = build_montages(img_page, (256, 256), (rown, len(eiglist)), transpose=transpose)[0]
    imsave(join(figdir, "%s_%d-%d_%04d.jpg" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), np.uint8(mtg * 255.0))
    plt.imsave(join(figdir, "%s_%d-%d_%04d.pdf" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), mtg, )
    print("Finish printing page (%.1fs)" % (time() - t0))
    if ImDist is not None: # if distance metric available then compute this
        distmat, ticks, fig = vis_distance_curve(ref_code, eigvect_avg, eigv_avg, G, ImDist, eiglist=eiglist,
	        maxdist=maxdist, rown=rown, distrown=distrown, sphere=sphere, figdir=figdir, RND=RND, namestr=namestr, )
        return mtg, codes_all, distmat, fig
    else:
        return mtg, codes_all
#%%
"""Note at origin of the space we need to take very small steps to reveal the content of the latent vectors"""
vis_eigen_explore(refvec.cpu(), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=2E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/2))
#%%
vis_eigen_explore(refvec.cpu(), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=.75E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = torch.randn(1, 512)
mtg = vis_eigen_explore(refvec.cpu(), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_expon"%modelsnm, transpose=False,
            eiglist=list(range(0, 20)), maxdist=6, rown=6, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))  #
#%%
refvec = torch.randn(1, 512)
mtg = vis_eigen_explore(refvec.cpu().numpy(), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_sphexpon"%modelsnm, transpose=False,
            eiglist=list(range(0, 20)), maxdist=0.35, rown=6, sphere=True, scaling=eva_avg[-1:-21:-1]**(-1/3))  #
#%% StyleGAN2 Face 512
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "ffhq-512-avg-tpurun1"
modelsnm = "Face512"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN2_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg,  = data["eva_avg"], data["evc_avg"], # feat_col = data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
# vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
#                     eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )
#%% Linear exploration from origin
vis_eigen_explore(np.zeros((1,512)), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=.75E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%% Linear exploration from origin
vis_eigen_explore(np.zeros((1,512)), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=.4E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%% Spherical exploration on shell
refvec = np.random.randn(1,512)
vis_eigen_explore(refvec, evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_sphexpon"%modelsnm, transpose=False, eiglist=list(range(0,20)), maxdist=0.2, rown=5, sphere=True, scaling=eva_avg[-1:-21:-1]**(-1/3))

#%% StyleGAN2 Face 256
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "ffhq-256-config-e-003810"
modelsnm = "Face256"
SGAN2 = loadStyleGAN2(modelnm+".pt")
SG2 = StyleGAN2_wrapper(SGAN2)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, = data["eva_avg"], data["evc_avg"], # feat_col = data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG2.visualize(refvec)
ToPILImage()(make_grid(img).cpu()).show()
#%%
# vis_eigen_frame(evc_avg, eva_avg, SG2, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN2_%s"%modelsnm, transpose=False,
#         eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )
#%% Linear exploration from origin
vis_eigen_explore(np.zeros((1,512)), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=.75E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%% Linear exploration from origin
vis_eigen_explore(np.zeros((1,512)), evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_orig_expon"%modelsnm, transpose=False,
            eiglist=list(range(0,20)), maxdist=.4E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = np.random.randn(1,512)
vis_eigen_explore(refvec, evc_avg, eva_avg, SG2, figdir=figsumdir, namestr="StyleGAN2_%s_sphexpon"%modelsnm, transpose=False, eiglist=list(range(0,20)), maxdist=0.25, rown=5, sphere=True, scaling=eva_avg[-1:-21:-1]**(-1/3))



#%% StyleGAN
"""StyleGAN for Faces"""
Hessdir = join(rootdir, 'StyleGAN')
modelnm = "StyleGAN"
SGAN = loadStyleGAN()
SG = StyleGAN_wrapper(SGAN)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, = data["eva_avg"], data["evc_avg"], # feat_col = data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = SG.visualize(refvec)
ToPILImage()(make_grid(img).cpu()).show()
#%%
# vis_eigen_frame(evc_avg, eva_avg, SG, ref_code=refvec, figdir=figsumdir, namestr="StyleGAN", transpose=False,
#                     eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )
#%% Linear exploration from origin
_=vis_eigen_explore(np.zeros((1, 512)), evc_avg, eva_avg, SG, figdir=figsumdir, namestr="StyleGAN_orig_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=.8E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/2.5))
#%%
_=vis_eigen_explore(np.zeros((1, 512)), evc_avg, eva_avg, SG, figdir=figsumdir, namestr="StyleGAN_orig_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=1E-3, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = np.random.randn(1, 512)
_=vis_eigen_explore(refvec, evc_avg, eva_avg, SG, figdir=figsumdir, namestr="StyleGAN_sphexpon", transpose=False,
            eiglist=list(range(0, 20)), maxdist=0.4, rown=5, sphere=True, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = np.random.randn(1, 512)
_=vis_eigen_explore(refvec, evc_avg, eva_avg, SG, figdir=figsumdir, namestr="StyleGAN_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=8, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))

#%% ProgGAN
"""Progressive Growing GAN! PG"""
Hessdir = join(rootdir, 'PGGAN')
modelnm = "PGGAN"
PGAN = loadPGGAN()
PG = PGGAN_wrapper(PGAN)
with np.load(join(Hessdir, "H_avg_%s.npz"%modelnm)) as data:
    eva_avg, evc_avg, = data["eva_avg"], data["evc_avg"], #feat_col = data["feat_col"]
#%%
refvec = torch.randn(1, 512).cuda()
img = PG.visualize(refvec)
ToPILImage()(make_grid(img).cpu())
#%%
# vis_eigen_frame(evc_avg, eva_avg, PG, ref_code=refvec, figdir=figsumdir, namestr="PGGAN", transpose=False,
#                     eiglist=None, eig_rng=(0, 20), maxdist=1, rown=5, sphere=False, )
#%% Linear exploration from origin
_=vis_eigen_explore(np.zeros((1, 512)), evc_avg, eva_avg, PG, figdir=figsumdir, namestr="PGGAN_orig_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=1E-4, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/2.5))
#%% Linear exploration from origin
_=vis_eigen_explore(np.zeros((1, 512)), evc_avg, eva_avg, PG, figdir=figsumdir, namestr="PGGAN_orig_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=.5E-4, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = np.random.randn(1, 512)
_=vis_eigen_explore(refvec, evc_avg, eva_avg, PG, figdir=figsumdir, namestr="PGGAN_sphexpon", transpose=False,
            eiglist=list(range(0, 20)), maxdist=0.25, rown=5, sphere=True, scaling=eva_avg[-1:-21:-1]**(-1/3))
#%%
refvec = np.random.randn(1, 512)
_=vis_eigen_explore(refvec, evc_avg, eva_avg, PG, figdir=figsumdir, namestr="PGGAN_expon", transpose=False,
            eiglist=list(range(0,20)), maxdist=6, rown=5, sphere=False, scaling=eva_avg[-1:-21:-1]**(-1/3))