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
from PIL import Image
from skimage.io import imsave
from build_montages import build_montages, color_framed_montages
from geometry_utils import SLERP, LERP, LExpMap
from GAN_utils import upconvGAN
#%% FC6 GAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
#%%
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
# go through spectrum in batch, and plot B number of axis in a row
def vis_eigen_frame(eigvect_avg, eigv_avg, G, ref_code=None, figdir=figdir, page_B=50,
                    eig_rng=(0, 4096), eiglist=None, maxdist=120, rown=7, transpose=True):
    if ref_code is None:
        ref_code = np.zeros((1, 4096))
    t0 = time()
    if eiglist is not None:
        if type(eiglist) is not list:
            raise
    else:
        eiglist = list(range(eig_rng[0], eig_rng[1]))
    csr = 0
    img_page = []
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        img_list = G.render(interp_codes)
        img_page.extend(img_list)
        if (idx == csr + page_B - 1) or idx + 1 == len(eiglist):
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%d-%d_%.e~%.e.jpg" %
                        (eiglist[csr]+1, eigi+1, eigv_avg[-eiglist[csr]-1], eigv_avg[-eigi])), np.uint8(mtg * 255.0))
            img_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(eiglist[csr], eigi, time()-t0))
            csr = idx
#%%
def vis_eigen_action(eigvec, ref_codes, G, figdir=figdir, page_B=50,
                    maxdist=120, rown=7, transpose=True, RND=None, namestr=""):
    if ref_codes is None:
        ref_codes = np.zeros((1, 4096))
    reflist = list(ref_codes)
    t0 = time()
    csr = 0
    img_page = []
    for idx, ref_code in enumerate(reflist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        interp_codes = LExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        img_list = G.render(interp_codes)
        img_page.extend(img_list)
        if (idx == csr + page_B - 1) or idx + 1 == len(reflist):
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            if RND is None: RND = np.random.randint(10000)
            imsave(join(figdir, "%s_ref_%d-%d_%d.jpg" %
                        (namestr, csr, idx, RND)), np.uint8(mtg * 255.0))
            img_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(csr, idx, time()-t0))
            csr = idx + 1
    return mtg
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
#%%
code_path = r"D:\Generator_DB_Windows\init_population\texture_init_code.npz"
with np.load(code_path) as data:
    codes_all = data["codes"]
ref_code = codes_all.mean(axis=0, keepdims=True)
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Text"
vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir, ref_code=ref_code,
                maxdist=120, rown=7, eig_rng=(0, 4096))
#%%

figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN"
vis_eigen_frame(eigvect_avg, eigv_avg, ref_code=None, figdir=figdir, page_B=50,
                eiglist=[0,1,2,5,10,20,30,50,100,200,300,400,600,800,1000,2000,3000,4000], maxdist=240, rown=5,
                transpose=False)
#%%
vis_eigen_action(eigvect_avg[:, -5], np.random.randn(10,4096), figdir=figdir, page_B=50,
                    maxdist=20, rown=5, transpose=False)
#%%
vis_eigen_action(eigvect_avg[:, -5], None, figdir=figdir, page_B=50,
                    maxdist=20, rown=5, transpose=False)

#%% BigGAN
from GAN_utils import BigGAN_wrapper, loadBigGAN
from pytorch_pretrained_biggan import BigGAN
from torchvision.transforms import ToPILImage
BGAN = loadBigGAN("biggan-deep-256").cuda()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
#%%
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
Hessdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
evc_nois = data['eigvects_nois_avg']
evc_clas = data['eigvects_clas_avg']
#%%
imgs = BG.render(np.random.randn(1, 256)*0.06)
#%%
eigi = 5
refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
vis_eigen_action(evc_BG[:, -eigi], refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=0.5, rown=5, transpose=False, namestr="eig%d"%eigi)
#%% Effect of eigen vectors within the noise space
eigi = 3
tanvec = np.hstack((evc_nois[:, -eigi], np.zeros(128)))
refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=2, rown=5, transpose=False, namestr="eig_nois%d"%eigi)
#%%
eigi = 3
tanvec = np.hstack((np.zeros(128), evc_clas[:, -eigi]))
refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=0.4, rown=5, transpose=False, namestr="eig_clas%d"%eigi)
#%%
eigi = 120
tanvec = np.hstack((np.zeros(128), evc_clas[:, -eigi]))
refvecs = np.vstack((EmbedMat[:, np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=2, rown=5, transpose=False, namestr="eig_clas%d"%eigi)

#%%
from GAN_utils import BigBiGAN_wrapper, loadBigBiGAN
from torchvision.transforms import ToPILImage
#%%
BBGAN = loadBigBiGAN().cuda()
BBG = BigBiGAN_wrapper(BBGAN)
# EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
#%%
from lpips import LPIPS
ImDist = LPIPS(net="squeeze")
#%%
from GAN_hessian_compute import hessian_compute, get_full_hessian
# from Hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_spectra
npzdir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_norm9_(\d*).npz", evakey='eigvals', evckey='eigvects', featkey="vect")
feat_arr = np.array(feat_col).squeeze()
#%%
eigid = 20
figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
mtg = vis_eigen_action(eigvec=eigvec_col[12][:, -eigid-1], ref_codes=feat_arr[[12, 0, 2, 4, 6, 8, 10, 12, ], :], G=BBG, maxdist=2, rown=5, transpose=False, namestr="BigBiGAN_norm9_eig%d"%eigid, figdir=figdir)