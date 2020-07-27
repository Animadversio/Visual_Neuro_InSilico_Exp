# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:08:54 2020

@author: Binxu Wang

Find important Nuisanced transformations in Noise space for a Class evolved image. 
"""

# backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_integrated\2020-06-01-09-46-37"
# Put the backup folder and the thread to analyze here 
#backup_dir = r"C:\Users\Poncelab-ML2a\Documents\monkeylogic2\generate_BigGAN\2020-07-22-10-14-22"
#backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2020-07-22-11-05-32"
backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2020-07-23-09-34-47"
threadid = 2
#evolspace = "all"


#%% Prepare the generator model and perceptual loss networks
from time import time
import os
from os.path import join
import sys
if os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
    Python_dir = r"C:\Users\Ponce lab\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
    Python_dir = r"C:\Users\Poncelab-ML2a\Documents\Python"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':
    Python_dir = r"E:\Github_Projects"
elif os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':
    Python_dir = r"D:\Github"

sys.path.append(join(Python_dir,"Visual_Neuro_InSilico_Exp"))
sys.path.append(join(Python_dir,"PerceptualSimilarity"))
import torch
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from GAN_utils import upconvGAN
from hessian_eigenthings.lanczos import lanczos
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, GANForwardMetricHVPOperator, \
    compute_hessian_eigenthings, get_full_hessian
from skimage.io import imsave
from torchvision.utils import make_grid
from build_montages import build_montages
from torchvision.transforms import ToPILImage, ToTensor
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
T00 = time()
import models # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
# model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
ImDist.cuda()
for param in ImDist.parameters():
    param.requires_grad_(False)
BGAN = BigGAN.from_pretrained("biggan-deep-256")
for param in BGAN.parameters():
    param.requires_grad_(False)
embed_mat = BGAN.embeddings.parameters().__next__().data
BGAN.cuda()

#%%
def LExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis]
    interp_vects = steps @ tangvect + refvect
    return interp_vects

def SExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis] * np.pi / 2
    interp_vects = steps @ tangvect + refvect
    return interp_vects

class BigGAN_wrapper():#nn.Module
    def __init__(self, BigGAN, space="class"):
        self.BigGAN = BigGAN
        self.space = space

    def visualize(self, code, scale=1.0, truncation=0.7):
        imgs = self.BigGAN.generator(code, truncation) # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation=0.7, B=5):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                           truncation=truncation, ).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
                clear_output(wait=True)
                progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

G = BigGAN_wrapper(BGAN)

#%% Test code for hessian eigendecomposition
#t0 = time()
#feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
#eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, ImDist,
#    num_eigenthings=300, mode="lanczos", use_gpu=True,)
#print(time() - t0,"\n")  # 81.02 s 
#%% Load the codes from the Backup folder 
import os
from  scipy.io import loadmat
import re
def load_codes_mat(backup_dir, threadnum=None, savefile=False):
    """ load all the code mat file in the experiment folder and summarize it into nparrays
    threadnum: can select one thread of the code if it's a parallel evolution. Usually, 0 or 1. 
        None for all threads. 
    """
    # make sure enough codes for requested size
    if "codes_all.npz" in os.listdir(backup_dir):
        # if the summary table exist, just read from it!
        with np.load(join(backup_dir, "codes_all.npz")) as data:
            codes_all = data["codes_all"]
            generations = data["generations"]
        return codes_all, generations
    if threadnum is None:
        codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "_code.mat" in fn])
    else:
        codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "thread%03d_code.mat"%(threadnum) in fn])
    codes_all = []
    img_ids = []
    for i, fn in enumerate(codes_fns[:]):
        matdata = loadmat(join(backup_dir, fn))
        codes_all.append(matdata["codes"])
        img_ids.extend(list(matdata["ids"]))

    codes_all = np.concatenate(tuple(codes_all), axis=0)
    img_ids = np.concatenate(tuple(img_ids), axis=0)
    img_ids = [img_ids[i][0] for i in range(len(img_ids))]
    generations = [int(re.findall("gen(\d+)", img_id)[0]) if 'gen' in img_id else -1 for img_id in img_ids]
    if savefile:
        np.savez(join(backup_dir, "codes_all.npz"), codes_all=codes_all, generations=generations)
    return codes_all, generations

#%% Load up the codes
from sklearn.decomposition import PCA 
import numpy as np
import matplotlib.pylab as plt
from imageio import imwrite

newimg_dir = join(backup_dir,"Hess_imgs")
summary_dir = join(backup_dir,"Hess_imgs","summary")
os.makedirs(newimg_dir,exist_ok=True)
os.makedirs(summary_dir,exist_ok=True)

print("Loading the codes from experiment folder %s", backup_dir)
codes_all, generations = load_codes_mat(backup_dir, threadnum=threadid) 
generations = np.array(generations) 
print("Shape of codes", codes_all.shape) 

#%% PCA of the Existing code (adopted from Manifold experiment)
final_gen_codes = codes_all[generations==max(generations), :]
final_gen_norms = np.linalg.norm(final_gen_codes, axis=1)
final_gen_norm = final_gen_norms.mean()
print("Average norm of the last generation samples %.2f" % final_gen_norm)
sphere_norm = final_gen_norm
print("Set sphere norm to the last generations norm!")
#% Do PCA and find the major trend of evolution
print("Computing PCs")
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1

PC1_vect = PC1_sign * PC_vectors[0,:] 
#%% From the setting of the bhv2 files find the fixed noise vectors
space_data = loadmat(join(backup_dir, "space_opts.mat"))["space_opts"]
evolspace = space_data[0,threadid]["name"][0]
print("Evolution happens in %s space, load the fixed code in `space_opts`" % evolspace)
if evolspace == "BigGAN_class":
    ref_noise_vec = space_data[0,threadid]['fix_noise_vec']
    #% Choose a random final generation codes as our reference
    ref_class_vec = final_gen_codes[0:1, :]
elif evolspace == "BigGAN":
    ref_vec = final_gen_codes[0:1, :]
    ref_noise_vec = ref_vec[:, :128]
    ref_class_vec = ref_vec[:, 128:]
#%% If you want to regenerate the images here. 
Demo = False
if Demo:
    imgs = G.visualize(torch.from_numpy(np.concatenate((ref_noise_vec, ref_class_vec), axis=1)).float().cuda()).cpu()
    ToPILImage()(imgs[0,:,:,:].cpu()).show()
    #%% If you want to regenerate the images here. 
    imgs = G.visualize(torch.from_numpy(np.concatenate((ref_noise_vec.repeat(5,axis=0), final_gen_codes[:5,:]), axis=1)).float().cuda()).cpu()
    ToPILImage()(imgs[0,:,:,:].cpu()).show()
    ToPILImage()(make_grid(imgs.cpu())).show()
#%% Compute Hessian decomposition and get the vectors
Hess_method = "BP"  # "BackwardIter"
t0 = time()
if Hess_method == "BP":
    print("Computing Hessian Decomposition Through auto-grad and full eigen decomposition.")
    classvec = torch.from_numpy(ref_class_vec).float().cuda()  # embed_mat[:, class_id:class_id+1].cuda().T
    noisevec = torch.from_numpy(ref_noise_vec).float().cuda()
    ref_vect = torch.cat((noisevec, classvec, ), dim=1).detach().clone()
    mov_vect = ref_vect.detach().clone().requires_grad_(True)
    #%
    imgs1 = G.visualize(ref_vect)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H = get_full_hessian(dsim, mov_vect)  # 77sec to compute a Hessian. # 114sec on ML2a
    # ToPILImage()(imgs[0,:,:,:].cpu()).show()
    eigvals, eigvects = np.linalg.eigh(H)  # 75 ms
    #%
    noisevec.requires_grad_(True)
    classvec.requires_grad_(False)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_nois = get_full_hessian(dsim, noisevec)  # 39.3 sec to compute a Hessian.# 59 sec on ML2a
    eigvals_nois, eigvects_nois = np.linalg.eigh(H_nois)  # 75 ms
    #%
    noisevec.requires_grad_(False)
    classvec.requires_grad_(True)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_clas = get_full_hessian(dsim, classvec)  # 39.3 sec to compute a Hessian.
    eigvals_clas, eigvects_clas = np.linalg.eigh(H_clas)  # 75 ms
    classvec.requires_grad_(False)

    np.savez(join(summary_dir, "Hess_mat.npz"), H=H, H_nois=H_nois, H_clas=H_clas, eigvals=eigvals,
             eigvects=eigvects, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, eigvals_nois=eigvals_nois,
             eigvects_nois=eigvects_nois, vect=ref_vect.cpu().numpy(),
             noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())

elif Hess_method == "BackwardIter":
    print("Computing Hessian Decomposition Through Lanczos decomposition on Backward HVP operator.")
    feat = torch.from_numpy(sphere_norm * PC1_vect).float().requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, ImDist,
        num_eigenthings=800, mode="lanczos", use_gpu=True)
    eigenvals = eigenvals[::-1]
    eigenvecs = eigenvecs[::-1, :]

elif Hess_method == "ForwardIter": 
    print("Computing Hessian Decomposition Through Lanczos decomposition on Forward HVP operator.")
    pass

print("%.2f sec"% (time() - t0))  # 31.75 secs for 300 eig, 87.52 secs for 800 eigs. 
#%% Visualize spectrum
plt.figure(figsize=[8,5])
plt.subplot(1,2,1)
plt.plot(eigvals[::-1], label="all")
plt.plot(eigvals_clas[::-1], label="class")
plt.plot(eigvals_nois[::-1], label="noise")
plt.ylabel("eigval");plt.legend()
plt.subplot(1,2,2)
plt.plot(np.log10(eigvals[::-1]), label="all")
plt.plot(np.log10(eigvals_clas[::-1]), label="class")
plt.plot(np.log10(eigvals_nois[::-1]), label="noise")
plt.ylabel("log(eigval)");plt.legend()
plt.savefig(join(summary_dir, "spectrum.jpg"))
#%% Angle with PC1 vector
if evolspace == "BigGAN_class":
    innerprod2PC1 = PC1_vect @ eigvects_clas.T
elif evolspace == "BigGAN_noise":
    innerprod2PC1 = PC1_vect @ eigvects_nois.T
elif evolspace == "BigGAN":
    innerprod2PC1 = PC1_vect @ eigvects.T
print("Eigen vector: Innerproduct max %.3E min %.3E std %.3E"% (innerprod2PC1.max(), innerprod2PC1.min(), innerprod2PC1.std()))
print("EigenDecomposition of Hessian of Image Similarity Metric\nEigen value: max %.3E min %.3E std %.3E"%
      (eigvals.max(), eigvals.min(), eigvals.std(), )) 

#%% Do interpolation

#% Interpolation in the class space
codes_all = []
img_names = []
for eigi in [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40, 60, 80]:#range(20):  # eigvects.shape[1]
    interp_class = LExpMap(classvec.cpu().numpy(), eigvects_clas[:, -eigi-1], 11, (-2.5, 2.5))
    interp_codes = np.hstack((noisevec.cpu().numpy().repeat(11, axis=0), interp_class, ))
    codes_all.append(interp_codes.copy())
    img_names.extend("class_eig%d_lin%.1f.jpg"%(eigi, dist) for dist in np.linspace(-2.5, 2.5, 11))
codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
imggrid = make_grid(img_all, nrow=11)
PILimg2 = ToPILImage()(imggrid)#.show()
PILimg2.save(join(summary_dir, "eigvect_clas_interp.jpg"))
npimgs = img_all.permute([2,3,1,0]).numpy()
for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))

#% Interpolation in the noise space
codes_all = []
img_names = []
for eigi in [0, 1, 2, 3, 4, 6, 10, 15, 20, 40]:#range(20):#eigvects_nois.shape[1]
    interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-4.5, 4.5))
    interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
    codes_all.append(interp_codes.copy())
    img_names.extend("noise_eig%d_lin%.1f.jpg"%(eigi, dist) for dist in np.linspace(-4.5, 4.5, 11))

codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
imggrid = make_grid(img_all, nrow=11)
PILimg3 = ToPILImage()(imggrid)#.show()
PILimg3.save(join(summary_dir, "eigvect_nois_interp.jpg"))
npimgs = img_all.permute([2,3,1,0]).numpy()
for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
print("Spent %.1f sec from start" % (time() - T00))
#%% Interpolation in the noise space
#codes_all = []
#img_names = []
#for eigi in range(20):#eigvects_nois.shape[1]
#    interp_noise = SExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-1, 1))
#    interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
#    codes_all.append(interp_codes.copy())
#    img_names.extend("noise_eig%d_sphang%d.jpg"%(eigi, angle) for angle in np.linspace(-90,90,11))
#
#codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
#img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
#imggrid = make_grid(img_all, nrow=11)
#PILimg4 = ToPILImage()(imggrid)#.show()
#PILimg4.save(join(summary_dir, "eigvect_nois_sph_interp.jpg"))
#npimgs = img_all.permute([2,3,1,0]).numpy()
#for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))

#%% 
#vec_norm = 220# sphere_norm
#eig_id = 0
#perturb_vect = eigenvecs[eig_id,:] # PC_vectors[1,:]    
#ang_step = 180 / 10
#theta_arr_deg = ang_step * np.arange(-5, 6)
#theta_arr = ang_step * np.arange(-5, 6) / 180 * np.pi
#codes_arc = np.array([np.cos(theta_arr), 
#                      np.sin(theta_arr) ]).T @ np.array([PC1_vect, perturb_vect])
#norms = np.linalg.norm(codes_arc, axis=1)
#codes_arc = codes_arc / norms[:,np.newaxis] * vec_norm
#imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
#
#npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()
#for i in range(npimgs.shape[3]):
#    angle = theta_arr_deg[i]
#    imwrite(join(newimg_dir, "norm%d_eig%d_ang%d.jpg"%(vec_norm, eig_id, angle)), npimgs[:,:,:,i])
#
#img_list = [npimgs[:,:,:,i] for i in range(npimgs.shape[3])]
#mtg1 = build_montages(img_list, [256, 256], [11, 1])[0]
##imwrite(join(backup_dir, "norm%d_eig%d.jpg"%(vec_norm, eig_id)),mtg1)
#imwrite(join(newimg_dir, "norm%d_eig%d.jpg"%(vec_norm, eig_id)),mtg1)