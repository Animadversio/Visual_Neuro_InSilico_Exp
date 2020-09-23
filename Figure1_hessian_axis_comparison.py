from hessian_axis_visualize import vis_eigen_frame, vis_eigen_action, vis_distance_curve, vis_eigen_explore
from hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, plot_consistentcy_mat, \
    plot_consistency_hist, plot_consistency_example, plot_spectra
from GAN_utils import loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, StyleGAN2_wrapper, upconvGAN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
"""Note the loading and visualization is fully deterministic, reproducible."""
#%%
from lpips import LPIPS
ImDist = LPIPS("squeeze")
#%% BigGAN
BGAN = loadBigGAN("biggan-deep-256").cuda()
BG = BigGAN_wrapper(BGAN)
EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
figdir = join(rootdir, 'BigGAN')
Hessdir = join(rootdir, 'BigGAN')
data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
eva_BG = data['eigvals_avg']
evc_BG = data['eigvects_avg']
evc_nois = data['eigvects_nois_avg']
evc_clas = data['eigvects_clas_avg']
eva_nois = data['eigvals_nois_avg']
eva_clas = data['eigvals_clas_avg']
evc_clas_f = np.vstack((np.zeros((128, 128)), evc_clas, ))
evc_nois_f = np.vstack((evc_nois, np.zeros((128, 128)), ))
#%% Class specific visualization
# refvecs = np.vstack((0.5*np.random.randn(128,10)), EmbedMat[:,np.random.randint(0, 1000, 10)]).T
classid = 230#np.random.randint(0, 1000, 1)[0]
RND = np.random.randint(1E4)
# refvec = np.vstack((0.7*np.random.randn(128,1), EmbedMat[:, [classid]])).T
with np.load(join(figdir, "cur_refvec_%d.npz"%classid)) as data:
    refvec = data["refvec"]
    classid = data["classid"]
print(["%.1e"%eig for eig in eva_clas[-np.array([1,2,4,7,16])-1]])
print(["%.1e"%eig for eig in eva_nois[-np.array([1,2,4,7,16])-1]])
# ['5.4e+03', '3.7e+03', '2.3e+03', '1.3e+03', '1.7e+01']
# ['1.6e+02', '1.3e+02', '4.8e+01', '9.7e+00', '2.7e+00']
mtg, codes_all, distmat, fig = vis_eigen_explore(refvec, evc_clas_f, eva_clas, BG, figdir=figdir, RND=RND,
            namestr="spect_avg_class%d"%classid, eiglist=[1,2,4,7,16], maxdist=0.3, rown=3,  ImDist=ImDist)

mtg, codes_all, distmat, fig = vis_eigen_explore(refvec, evc_clas_f, eva_clas, BG, figdir=figdir, RND=RND,
            namestr="spect_avg_class%d_5"%classid, eiglist=[1,2,4,7,16], maxdist=0.5, rown=5,  ImDist=ImDist)

mtg, codes_all, distmat, fig = vis_eigen_explore(refvec, evc_nois_f, eva_nois, BG, figdir=figdir, RND=RND,
        namestr="spect_avg_noise%d"%classid, eiglist=[1,2,4,7,16], maxdist=0.18, rown=3,  ImDist=ImDist, sphere=True)

mtg, codes_all, distmat, fig = vis_eigen_explore(refvec, evc_nois_f, eva_nois, BG, figdir=figdir, RND=RND,
        namestr="spect_avg_noise%d_5"%classid, eiglist=[1,2,4,7,16], maxdist=0.5, rown=5,  ImDist=ImDist, sphere=True)
# mtg, codes_all = vis_eigen_frame(evc_clas_f, eva_clas, BG, ref_code=refvec, figdir=figdir,namestr="spect_avg_class%d"%classid,
#              eiglist=[1,2,4,7,16], maxdist=0.3, rown=3, transpose=True, RND=RND)
# plt.imsave(join(figdir, "spect_class_lin_%d_0-64.pdf"%classid), mtg,)
# distmat, ticks, fig = vis_distance_vector(refvec, evc_clas_f, eva_clas, BG, ImDist, figdir=figdir, namestr="spect_avg_class%d"%classid,
#           eiglist = [1,2,4,7,16], maxdist=0.3, rown=3, distrown=19, sphere=False, RND=RND)
#
# mtg, codes_all = vis_eigen_frame(evc_clas_f, eva_clas, BG, ref_code=refvec, figdir=figdir,namestr="spect_avg_class%d"%classid,
#              eiglist=[1,2,4,7,16], maxdist=0.5, rown=5, transpose=True, RND=RND)
# plt.imsave(join(figdir, "spect_class_lin5_%d_0-64.pdf"%classid), mtg, )
# distmat, ticks, fig = vis_distance_curve(refvec, evc_clas_f, eva_clas, BG, ImDist, figdir=figdir, namestr="spect_avg_class%d"%classid,
#           eiglist=[1,2,4,7,16], maxdist=0.5, rown=5, distrown=19, sphere=False, RND=RND)
#
# mtg, codes_all = vis_eigen_frame(evc_nois_f, eva_nois, BG, ref_code=refvec, figdir=figdir, namestr="spect_avg_noise_sph%d"%classid,
#             eiglist=[1,2,4,7,16], maxdist=0.18, rown=3, transpose=True, sphere=True, RND=RND)
# plt.imsave(join(figdir, "spect_noise_sph_%d_0-64.pdf"%classid), mtg,)
# distmat, ticks, fig = vis_distance_curve(refvec, evc_nois_f, eva_nois, BG, ImDist, figdir=figdir, namestr="spect_avg_noise_sph%d"%classid,
#           eiglist=[1,2,4,7,16], maxdist=0.18, rown=3, distrown=19, sphere=False, RND=RND)
#
# mtg, codes_all = vis_eigen_frame(evc_nois_f, eva_nois, BG, ref_code=refvec, figdir=figdir, namestr="spect_avg_noise_sph%d"%classid,
#             eiglist=[1,2,4,7,16], maxdist=0.3, rown=5, transpose=True, sphere=True, RND=RND)
# plt.imsave(join(figdir, "spect_noise_sph5_%d_0-64.pdf"%classid), mtg,)
# distmat, ticks, fig = vis_distance_curve(refvec, evc_nois_f, eva_nois, BG, ImDist, figdir=figdir, namestr="spect_avg_noise_sph%d"%classid,
#           eiglist=[1,2,4,7,16], maxdist=0.3, rown=5, distrown=19, sphere=False, RND=RND)
#
# np.savez(join(figdir, "cur_refvec_%d.npz"%classid), refvec=refvec, classid=classid)
#%% Visualize Action of Vectors on Some Position
eigi = 5
tanvec = np.hstack((evc_nois[:, -eigi], np.zeros(128)))
refvecs = np.vstack((0.7*np.random.randn(128, 10), EmbedMat[:,np.random.randint(0, 1000, 10)])).T
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=1, rown=5, transpose=False, namestr="eig_nois%d_sph"%eigi, sphere=True)
# using spherical exploration is much better than linear
vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                 maxdist=0.4, rown=5, transpose=False, namestr="eig_nois%d"%eigi)
#%%
# refvec = np.vstack((0.7*np.random.randn(128,1), EmbedMat[:,np.random.randint(0, 1000, 1)])).T
mtg, codes_all = vis_eigen_frame(evc_BG, eva_BG, BG, ref_code=refvec, figdir=figdir, namestr="spect_all",
                eiglist=[1,2,4,6,8,10,15,20,30,40,60,80,120], maxdist=0.3, rown=5, transpose=True)


#%% StyleGAN2
"""StyleGAN2 model"""
from hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, plot_consistentcy_mat, plot_consistency_example
figdir = join(rootdir, 'StyleGAN2')
Hessdir = join(rootdir, 'StyleGAN2')
dataroot = r"E:\Cluster_Backup\StyleGAN2"


#%% Face 512 model
modelnm = "ffhq-512-avg-tpurun1"
modelsnm = "Face512"
SGAN = loadStyleGAN2(modelnm+".pt")
SG = StyleGAN2_wrapper(SGAN)
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
#%%
mtg,codes_all = vis_eigen_frame(evc_avg, eva_avg, SG, ref_code=feat_col[5,:], figdir=figdir,
                                namestr="spect_avg_%s"%modelnm, eiglist=[1,2,4,6,8,10,15,20,30,40,60,80,120],
                                maxdist=5.0, rown=5, transpose=True, sphere=False)
#%%
#%% Final Version
veci = 5
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_lin_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,64], maxdist=4.5, rown=3, transpose=True, sphere=False)
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_lin5_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,64], maxdist=4.5, rown=5, transpose=True, sphere=False)
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_sph_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,32,64], maxdist=0.2, rown=3, transpose=True, sphere=True)
print(["%.1e"%eig for eig in eigval_col[veci][-np.array([2,4,8,16,64])-1]])
#%% Final Version
veci = 5
mtg,codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                    namestr="spect_indiv_lin_%s_%d"%(modelsnm, veci),
                eiglist=[2,4,8,16,64], maxdist=4.5, rown=3, transpose=True, sphere=False)
mtg,codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                    namestr="spect_indiv_lin5_%s_%d"%(modelsnm, veci),
                eiglist=[2,4,8,16,64], maxdist=4.5, rown=5, transpose=True, sphere=False)
mtg,codes_all = vis_eigen_frame(eigvec_col[5], eigval_col[5], SG, ref_code=feat_col[5,:], figdir=figdir,
                    namestr="spect_indiv_sph_%s_%d"%(modelsnm, veci),
                eiglist=[2,4,8,16,32,64], maxdist=0.2, rown=3, transpose=True, sphere=True)
print(["%.1e"%eig for eig in eigval_col[veci][-np.array([2,4,8,16,64])-1]])
# ['7.6e+00', '1.2e+00', '1.8e-01', '1.4e-02', '9.3e-05']
#%% Cat Datasets StyleGAN2
modelnm = "stylegan2-cat-config-f"
modelsnm = "Cat256"
SGAN = loadStyleGAN2(modelnm+".pt", size=256,)
SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)

#%% Final Version
veci = 40
RND = np.random.randint(1E4)
print(["%.1e"%eig for eig in eigval_col[veci][-np.array([2,4,8,16,64])-1]])
# ['7.5e+01', '2.7e+01', '1.5e+00', '5.2e-02', '-1.3e-05']
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_lin_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,64], maxdist=2.5, rown=3, transpose=True, sphere=False)
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_lin5_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,64], maxdist=2.5, rown=5, transpose=True, sphere=False)
mtg,codes_all, distmat, fig = vis_eigen_explore(feat_col[veci,:], eigvec_col[veci], eigval_col[veci], SG, figdir=figdir,
                    namestr="spect_indiv_sph_%s_%d"%(modelsnm, veci), ImDist=ImDist,
                eiglist=[2,4,8,16,64], maxdist=0.15, rown=3, transpose=True, sphere=True)

#%% Final version
veci = 40
RND = np.random.randint(1E4)
print(["%.1e"%eig for eig in eigval_col[veci][-np.array([2,4,8,16,64])-1]])
# ['7.5e+01', '2.7e+01', '1.5e+00', '5.2e-02', '-1.3e-05']
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_lin_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=2.5, rown=3, transpose=True,
                                sphere=False, RND=RND)
plt.imsave(join(figdir, "spect_indiv_lin_Cat256_%d_2-64.pdf"%veci), mtg, )
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_lin_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=2.5, rown=5, transpose=True,
                                sphere=False, RND=RND)
plt.imsave(join(figdir, "spect_indiv_lin5_Cat256_%d_2-64.pdf"%veci), mtg, )
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_sph_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=0.15, rown=3, transpose=True,
                                sphere=True, RND=RND)
plt.imsave(join(figdir, "spect_indiv_sph_Cat256_%d_2-64.pdf"%veci), mtg, )

#%% Animation Portraits
modelnm = "2020-01-11-skylion-stylegan2-animeportraits"
modelsnm = "Anime512"
SGAN = loadStyleGAN2(modelnm+".pt", size=512,)
SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
#%% Final version
veci = 13
print(["%.1e"%eig for eig in eigval_col[veci][-np.array([2,4,8,16,64])-1]])
# ['4.0e+01', '2.0e+01', '4.0e+00', '6.1e-01', '1.2e-03']
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_lin_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=2.5, rown=3, transpose=True,
                                sphere=False)
plt.imsave(join(figdir, "spect_indiv_lin_Anime512_%d_2-64.pdf"%veci), mtg, )
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_lin_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=2.5, rown=5, transpose=True,
                                sphere=False)
plt.imsave(join(figdir, "spect_indiv_lin5_Anime512_%d_2-64.pdf"%veci), mtg, )
mtg, codes_all = vis_eigen_frame(eigvec_col[veci], eigval_col[veci], SG, ref_code=feat_col[veci,:], figdir=figdir,
                namestr="spect_indiv_sph_%s"%modelnm, eiglist=[2,4,8,16,64], maxdist=0.15, rown=3, transpose=True,
                                sphere=True)
plt.imsave(join(figdir, "spect_indiv_sph_Anime512_%d_2-64.pdf"%veci), mtg, )


#%% Face1024
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "stylegan2-ffhq-config-f"
modelsnm = "Face1024"
# SGAN = loadStyleGAN2(modelnm+".pt", size=1024,)
# SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
fig0 = plot_spectra(eigval_col=eigval_col, savename="%s_spectrum"%modelnm)
corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, figdir=figdir, use_cuda=False, savelabel=modelnm, )
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eigval_col, eigvec_col, figdir=figdir, nsamp=5, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
#%% "ImageNet512"
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "model.ckpt-533504"
modelsnm = "ImageNet512"
# SGAN = loadStyleGAN2(modelnm+".pt", size=512,)
# SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
fig0 = plot_spectra(eigval_col=eigval_col, savename="%s_spectrum"%modelnm)
corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, figdir=figdir, use_cuda=True, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eigval_col, eigvec_col, figdir=figdir, nsamp=5, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
#%% "Face256"
Hessdir = join(rootdir, 'StyleGAN2')
modelnm = "ffhq-256-config-e-003810"
modelsnm = "Face256"
# SGAN = loadStyleGAN2(modelnm+".pt", size=512,)
# SG = StyleGAN2_wrapper(SGAN, )
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
fig0 = plot_spectra(eigval_col=eigval_col, savename="%s_spectrum"%modelnm)
corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, figdir=figdir, use_cuda=True, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
fig3 = plot_consistency_example(eigval_col, eigvec_col, figdir=figdir, nsamp=5, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)

#%%
dataroot = r"E:\Cluster_Backup\StyleGAN2"
modelnm = "stylegan2-car-config-f"
modelsnm = "Car512"
# SGAN = loadStyleGAN2(modelname+".pt", size=512, channel_multiplier=2)
# SG = StyleGAN2_wrapper(SGAN)
eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(join(dataroot, modelnm), "Hess_BP_(\d*).npz", featkey="feat")
feat_col = np.array(feat_col).squeeze()
H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
np.savez(join(Hessdir, "H_avg_%s.npz"%modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
fig0 = plot_spectra(eigval_col=eigval_col, savename="%s_spectrum"%modelnm)
corr_mat_log, corr_mat_lin = compute_hess_corr(eigval_col, eigvec_col, figdir=figdir, use_cuda=True, savelabel=modelnm)
fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)
fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="StyleGAN2 %s"%modelnm,
                                    savelabel=modelnm)
fig3 = plot_consistency_example(eigval_col, eigvec_col, figdir=figdir, nsamp=5, titstr="StyleGAN2 %s"%modelnm, savelabel=modelnm)

