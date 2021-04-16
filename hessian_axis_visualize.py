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
from imageio import imwrite, imsave
from PIL import Image
from build_montages import build_montages, color_framed_montages
from geometry_utils import SLERP, LERP, LExpMap, SExpMap
from GAN_utils import upconvGAN, loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, \
    StyleGAN2_wrapper
from GAN_hessian_compute import hessian_compute, get_full_hessian
from hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_spectra, average_H
#%%
# figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
# go through spectrum in batch, and plot B number of axis in a row
def vis_eigen_frame(eigvect_avg, eigv_avg, G, ref_code=None, figdir="", RND=None, namestr="", page_B=50, transpose=True,
                    eiglist=None, eig_rng=(0, 4096), maxdist=120, rown=7, sphere=False, ):
    if ref_code is None: ref_code = np.zeros((1, eigvect_avg.shape[0]))
    if RND is None: RND = np.random.randint(10000)
    if eiglist is None: eiglist = list(range(eig_rng[0], eig_rng[1]))
    t0 = time()
    csr = 0
    codes_page = []
    codes_col = []
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        if (idx == csr + page_B - 1) or idx + 1 == len(eiglist):
            codes_all = np.concatenate(tuple(codes_page), axis=0)
            img_page = G.render(codes_all)
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%s_%d-%d_%.e~%.e_%04d.jpg" %
            (namestr, eiglist[csr]+1, eigi+1, eigv_avg[-eiglist[csr]-1], eigv_avg[-eigi], RND)), np.uint8(mtg * 255.0))
            codes_col.append(codes_all)
            codes_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(eiglist[csr], eigi, time()-t0))
            csr = idx + 1
    return mtg, codes_col

def vis_eigen_explore(ref_code, eigvect_avg, eigv_avg, G, figdir="", RND=None, namestr="", transpose=True, save=True,
                      eiglist=[1,2,4,7,16], maxdist=120, rown=5, sphere=False, ImDist=None, distrown=19, scaling=None):
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
    if save:
        imsave(join(figdir, "%s_%d-%d_%04d.jpg" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), np.uint8(mtg * 255.0))
        plt.imsave(join(figdir, "%s_%d-%d_%04d.pdf" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), mtg, )
    print("Finish printing page (%.1fs)" % (time() - t0))
    if ImDist is not None: # if distance metric available then compute this
        distmat, ticks, fig = vis_distance_curve(ref_code, eigvect_avg, eigv_avg, G, ImDist, eiglist=eiglist,
	        maxdist=maxdist, rown=rown, distrown=distrown, sphere=sphere, figdir=figdir, RND=RND, namestr=namestr, )
        return mtg, codes_all, distmat, fig
    else:
        return mtg, codes_all

def vis_eigen_explore_row(ref_code, eigvect_avg, eigv_avg, G, figdir="", RND=None, namestr="", indivimg=False,
     transpose=True, eiglist=[1,2,4,7,16], maxdist=120, rown=5, sphere=False, save=True):  # ImDist=None, distrown=19
    """This is small scale version of vis_eigen_frame + vis_distance_vector """
    if RND is None: RND = np.random.randint(10000)
    if eiglist is None: eiglist = list(range(len(eigv_avg)))
    t0 = time()
    codes_page = []
    mtg_col = []
    ticks = np.linspace(-maxdist, maxdist, rown)
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        img_page = G.render(interp_codes)
        mtg = build_montages(img_page, (256, 256), (rown, 1), transpose=transpose)[0]
        if save:
            imsave(join(figdir, "%s_eig%d_%04d.jpg" % (namestr, eigi+1, RND)), np.uint8(mtg * 255.0))
            plt.imsave(join(figdir, "%s_eig%d_%04d.pdf" % (namestr, eigi+1, RND)), mtg, )
        mtg_col.append(mtg)
        if indivimg and save:
            for deviation, img in zip(ticks, img_page):
                imsave(join(figdir, "%s_eig%d_%.1e_%04d.jpg" % (namestr,eigi+1, deviation, RND)), np.uint8(img * 255.0))
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    print("Finish printing page (%.1fs)" % (time() - t0))
    # if ImDist is not None: # if distance metric available then compute this
    #     distmat, ticks, fig = vis_distance_curve(ref_code, eigvect_avg, eigv_avg, G, ImDist, eiglist=eiglist,
	#         maxdist=maxdist, rown=rown, distrown=distrown, sphere=sphere, figdir=figdir, RND=RND, namestr=namestr, )
    #     return mtg, codes_all, distmat, fig
    # else:
    return mtg_col, codes_all

def vis_distance_curve(ref_code, eigvect_avg, eigvals_avg, G, ImDist, eiglist=[1,2,4,7,16],
	    maxdist=0.3, rown=3, distrown=19, sphere=False, figdir="", RND=None, namestr="", ):
    refimg = G.visualize_batch_np(ref_code.reshape(1, -1))
    if RND is None: RND = np.random.randint(10000)
    codes_page = []
    ticks = np.linspace(-maxdist, maxdist, distrown, endpoint=True)
    visticks = np.linspace(-maxdist, maxdist, rown, endpoint=True)
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi - 1], distrown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi - 1], distrown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        # if (idx == csr + page_B - 1) or idx + 1 == len(eiglist):
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    img_page = G.visualize_batch_np(codes_all)
    with torch.no_grad():
        dist_all = ImDist(refimg, img_page).squeeze()
    distmat = dist_all.reshape(-1, distrown).numpy()
    fig = plt.figure(figsize=[5, 3])
    for idx, eigi in enumerate(eiglist):
        plt.plot(ticks, distmat[idx, :], label="eig%d %.E" % (eigi + 1, eigvals_avg[-eigi - 1]), lw=2.5, alpha=0.7)
    plt.xticks(visticks)
    plt.ylabel("Image distance")
    plt.xlabel("L2 in latent space" if not sphere else "Angle (rad) in latent space")
    plt.legend()
    plt.subplots_adjust(left=0.14, bottom=0.14)
    plt.savefig(join(figdir, "%s_imdistcrv_%04d.jpg" % (namestr, RND)) )
    plt.savefig(join(figdir, "%s_imdistcrv_%04d.pdf" % (namestr, RND)) )
    plt.show()
    return distmat, ticks, fig
#%%
def vis_eigen_action(eigvec, ref_codes, G, figdir="", page_B=50,
                    maxdist=120, rown=7, transpose=True, RND=None, namestr="", sphere=False):
    if ref_codes is None:
        ref_codes = np.zeros(eigvec.size)
    if RND is None: RND = np.random.randint(10000)
    reflist = list(ref_codes)
    t0 = time()
    csr = 0
    codes_page = []
    codes_col = []
    for idx, ref_code in enumerate(reflist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        if (idx == csr + page_B - 1) or idx + 1 == len(reflist):
            codes_all = np.concatenate(tuple(codes_page), axis=0)
            img_page = G.render(codes_all)
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            imsave(join(figdir, "%s_ref_%d-%d_%04d.jpg" %
                        (namestr, csr, idx, RND)), np.uint8(mtg * 255.0))
            codes_col.append(codes_all)
            codes_page = []
            print("Finish printing page vector %d-%d (%.1fs)"%(csr, idx, time()-t0))
            csr = idx + 1
    return mtg, codes_col
#%%
def vis_eigen_action_row(eigvec, ref_codes, G, figdir="", indivimg=False,
                    maxdist=120, rown=7, transpose=True, RND=None, namestr="", sphere=False):
    if ref_codes is None:
        ref_codes = np.zeros(eigvec.size)
    if RND is None: RND = np.random.randint(10000)
    reflist = list(ref_codes)
    t0 = time()
    codes_col = []
    mtg_col = []
    ticks = np.linspace(-maxdist, maxdist, rown)
    for idx, ref_code in enumerate(reflist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        img_page = G.render(interp_codes)
        mtg = build_montages(img_page, (256, 256), (rown, 1), transpose=transpose)[0]
        imsave(join(figdir, "%s_ref_%d_%04d.jpg" %
                    (namestr, idx, RND)), np.uint8(mtg * 255.0))
        codes_col.append(interp_codes)
        mtg_col.append(mtg)
        if indivimg:
            for div, img in zip(ticks, img_page):
                imsave(join(figdir, "%s_ref_%d_%.1e_%04d.jpg" % (namestr, idx, div, RND)), np.uint8(img * 255.0))
        print("Finish printing along vector %d (%.1fs)"%(idx, time()-t0))
    return mtg_col, codes_col

#%% imgs = visualize_np(G, interp_codes)
if __name__ == "__main__":
    # %%
    from lpips import LPIPS
    ImDist = LPIPS(net="squeeze")
    #%% FC6 GAN on ImageNet
    G = upconvGAN("fc6")
    G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
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

    #%% BigGAN on ImageNet Class Specific
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

    #%% BigBiGAN on ImageNet
    from GAN_utils import BigBiGAN_wrapper, loadBigBiGAN
    from torchvision.transforms import ToPILImage
    BBGAN = loadBigBiGAN().cuda()
    BBG = BigBiGAN_wrapper(BBGAN)
    # EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
    #%%
    from GAN_hessian_compute import hessian_compute, get_full_hessian
    from hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_spectra
    npzdir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_norm9_(\d*).npz", evakey='eigvals', evckey='eigvects', featkey="vect")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 20
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
    mtg = vis_eigen_action(eigvec=eigvec_col[12][:, -eigid-1], ref_codes=feat_arr[[12, 0, 2, 4, 6, 8, 10, 12, ], :], G=BBG, maxdist=2, rown=5, transpose=False, namestr="BigBiGAN_norm9_eig%d"%eigid, figdir=figdir)
    #%% StyleGAN2
    from GAN_hessian_compute import hessian_compute
    from GAN_utils import loadStyleGAN, StyleGAN_wrapper
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
    #%% Cats
    modelname = "stylegan2-cat-config-f"
    npzdir = r"E:\Cluster_Backup\StyleGAN2\stylegan2-cat-config-f"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=2)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 5
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=3, rown=5, transpose=False, namestr="SG2_Cat_eig%d"%eigid, figdir=figdir)
    #%% Animation
    modelname = "2020-01-11-skylion-stylegan2-animeportraits"
    npzdir = r"E:\Cluster_Backup\StyleGAN2\2020-01-11-skylion-stylegan2-animeportraits"
    SGAN = loadStyleGAN(modelname+".pt", size=512, channel_multiplier=2)
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 3
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=10, rown=5, transpose=False, namestr="SG2_anime_eig%d"%eigid, figdir=figdir)
    #%% Faces 256
    modelname = 'ffhq-256-config-e-003810'
    npzdir = r"E:\Cluster_Backup\StyleGAN2\ffhq-256-config-e-003810"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=1)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 14
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=10, rown=5, transpose=False, namestr="SG2_Face256_eig%d"%eigid, figdir=figdir)
    #%% Faces
    modelname = 'ffhq-256-config-e-003810'
    npzdir = r"E:\Cluster_Backup\StyleGAN2\ffhq-256-config-e-003810"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=1)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    def average_H(eigval_col, eigvec_col):
        """Compute the average Hessian over a bunch of positions"""
        nH = len(eigvec_col)
        dimen = eigval_col.shape[1]
        H_avg = np.zeros((dimen, dimen))
        for iH in range(nH):
            H = (eigvec_col[iH] * eigval_col[iH][np.newaxis, :]) @ eigvec_col[iH].T
            H_avg += H
        H_avg /= nH
        eva_avg, evc_avg = np.linalg.eigh(H_avg)
        return H_avg, eva_avg, evc_avg

    H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
    #%%
    maxang = 1.5
    figdir = "E:\Cluster_Backup\StyleGAN2_axis\Face256"
    for eigid in list(range(20))+list(range(20,60,2))+list(range(60,200,4)):
        mtg = vis_eigen_action(eigvec=evc_avg[:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 40, 60], :],
                       G=G, sphere=True, maxdist=1.5, rown=5, transpose=False, namestr="SG2_Face256_AVGeig%d_Sph%.1f"%(eigid, maxang), figdir=figdir)
        print("Finish printing eigenvalue %d"%eigid)
        # if eigid==5:
        #     break

    maxdis = 6
    figdir = "E:\Cluster_Backup\StyleGAN2_axis\Face256"
    for eigid in list(range(20))+list(range(20,60,2))+list(range(60,200,4)):
        mtg = vis_eigen_action(eigvec=evc_avg[:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 40, 60], :],
                               G=G, maxdist=maxdis, rown=7, transpose=False, namestr="SG2_Face256_AVGeig%d_Lin%1.f"%(eigid, maxdis), figdir=figdir)
        print("Finish printing eigenvalue %d"%eigid)

    #%

