"""
The Script is designed to compare the PCA of the image datasets and the
corresponding Hessian eigenvectors.

"""
from GAN_utils import BigGAN_wrapper, loadBigGAN, StyleGAN2_wrapper, loadStyleGAN2, loadPGGAN, PGGAN_wrapper
import torch, numpy as np
from os.path import join
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pylab as plt
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds, eigs
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist, pdist
from load_hessian_data import load_Haverage
"""The major challenge is memory"""
figroot = r"E:\OneDrive - Washington University in St. Louis\GAN_PCA"
#%% PGGAN
modelsnm = "PGGAN"
PGAN = loadPGGAN()
G = PGGAN_wrapper(PGAN)
savedir = join(figroot, "PGGAN")
#%%
vecn = 4000
codes = np.random.randn(vecn, 512)
imgs = G.visualize_batch_np(codes)
#% Compute sparse SVD
U, S, VH = svds(imgs.view(vecn, -1).numpy(), k=30)
#%%
for PCi in range(30):
    img_mean = VH[-1, :].reshape((3, 256, 256)).transpose((1,2,0)) * np.sign(VH[-1, :].mean()) # sign may be inverted
    img_dev = VH[-PCi, :].reshape((3, 256, 256)).transpose((1,2,0))
    plt.figure(figsize=[7,3])
    plt.subplot(1, 3, 1)
    plt.imshow((img_mean-0.4*img_dev)*255)
    plt.axis(False)
    plt.subplot(1, 3, 2)
    plt.imshow((img_mean)*255)
    plt.axis(False)
    plt.subplot(1, 3, 3)
    plt.imshow((img_mean+0.4*img_dev)*255)
    plt.axis(False)
    plt.suptitle("image space PC%02d"%PCi)
    plt.savefig(join(savedir, "im_PC%02d.png"%PCi))
    plt.show()
#%%
regr = LinearRegression()
regr.fit(codes, U)
PCregaxes = regr.coef_

imgs_reg_ax = G.visualize_batch_np(PCregaxes)
ToPILImage()(make_grid(imgs_reg_ax)).show()
imgs_reg_ax = G.visualize_batch_np(-PCregaxes)
ToPILImage()(make_grid(imgs_reg_ax)).show()
#%%
PCWeiAvg = (U.T @ codes)

imgs_reg_ax = G.visualize_batch_np(PCWeiAvg)
ToPILImage()(make_grid(imgs_reg_ax)).show()
imgs_reg_ax = G.visualize_batch_np(-PCWeiAvg)
ToPILImage()(make_grid(imgs_reg_ax)).show()
"""
Basically the dimensions found by weighted averaging code or by linear regression is the same.
They generates similar transitions. 
"""
#%%
H, eva, evc = load_Haverage("PGGAN")
#%%
imgs_PCs = G.visualize_batch_np(-evc[:, -25:].T)
ToPILImage()(make_grid(imgs_PCs)).show()
#%%
corrmat1 = 1 - cdist(PCregaxes, evc[:, -25:].T, metric="correlation")
#%%
unitPCregaxes = PCregaxes / np.linalg.norm(PCregaxes, axis=1, keepdims=True)
#%%
proj_coef = unitPCregaxes @ evc[:,::-1]
plt.plot(proj_coef.T**2)
plt.show()
#%%
ccmat1 = abs(1 - cdist(unitPCregaxes, evc[:, -30:].T, metric="cosine"))
plt.figure()
plt.matshow(ccmat1,0)
plt.ylabel("Hessian eigenvector")
plt.xlabel("PC regress axes")
cbar = plt.colorbar()
cbar.ax.set_ylabel("abs(cosine)")
plt.show()
#%%
ccmat2 = 1 - cdist(PCWeiAvg, evc[:, -30:].T, metric="correlation")
#%%
#%%
SGAN = loadStyleGAN2()
G = StyleGAN2_wrapper(SGAN)




