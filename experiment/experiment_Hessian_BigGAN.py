# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:08:54 2020

@author: Binxu Wang

Find important Nuisanced + Class transformations in Noise + Class space for a BigGAN evolved image. 
"""

# backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_integrated\2020-06-01-09-46-37"
# Put the backup folder and the thread to analyze here 
#backup_dir = r"C:\Users\Poncelab-ML2a\Documents\monkeylogic2\generate_BigGAN\2020-07-22-10-14-22"
# backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2020-08-06-10-18-55"#2020-08-04-09-54-25"#
backup_dir = r"C:\Users\Ponce lab\Documents\ml2a-monk\generate_BigGAN\2020-08-10-09-59-48"
threadid = 1

score_rank_avg = False  # If True, it will try to read "scores_record.mat", from the backup folder and read "scores_record"
                        # Else, it will use the unweighted mean code of the last generation as the center vector. 
                        # Need to run the BigGAN postHoc Analysis to save the `scores_record` mat and use this flag

exact_distance = True   # Control if exact distance search is used or approximate heuristic rule is used.
target_distance = [0.08, 0.16, 0.24, 0.32, 0.40]
#target_distance = [0.09, 0.18, 0.27, 0.36, 0.45]  # if exact_distance is True it will search for images with these
                                                  # distance to reference image along each eigenvector.
                                                  
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
from tqdm import tqdm
T00 = time()
import models # from PerceptualSimilarity folder
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
# model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
ImDist.cuda()
for param in ImDist.parameters():
    param.requires_grad_(False)
#%%
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
#%% Use Rank weight like CMAES
def rankweight(popsize):
    weights_pad =np.zeros(popsize)
    mu = popsize/2
    weights = np.log(mu + 1 / 2) - (np.log(np.arange(1, 1 + np.floor(mu))))
    weights = weights / sum(weights)
    mu = int(mu)
    weights_pad[:mu] = weights
    return weights_pad
#%% Compute Image distance using the ImDist
def Hess_img_distmat(ImDist, img_all, nrow=11):
    """
    img_all: a torch 4D array of shape [N, 3, 256, 256] on cpu()
    nrow: specify how many images are generated in one axis.
        It will arrange the images in `img_all` in a matrix and calculate the distance to the center image in each row.
    """
    distmat = torch.zeros(img_all.shape[0]).view(-1,nrow)
    nTot = distmat.shape[0]
    for irow in range(nTot):
        rstr = irow * nrow
        rend = nrow + rstr
        rmid = (nrow-1)//2 + rstr
        with torch.no_grad():
            dists = ImDist(img_all[rstr:rend,:],img_all[rmid,:]).squeeze()
        distmat[irow, :]=dists.cpu()
    return distmat
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
evo_codes_all, generations = load_codes_mat(backup_dir, threadnum=threadid) 
generations = np.array(generations) 
print("Shape of codes", evo_codes_all.shape) 
# Use penultimate generation to generate the center
final_gen_codes = evo_codes_all[generations==max(generations)-1, :]
final_gen_norms = np.linalg.norm(final_gen_codes, axis=1)
final_gen_norm = final_gen_norms.mean()
print("Average norm of the last generation samples %.2f" % final_gen_norm)
#%% If there is score of images, load them up here. And compute the weighted average code
if score_rank_avg:
    try:
        scores_record = loadmat(join(backup_dir, "scores_record.mat"))["scores_record"]
        scores_thread = scores_record[:,threadid]
        assert len(scores_thread)==max(generations)-min(generations) # block of scores is the number of block of codes -1  
        final_gen_scores = scores_thread[-1].squeeze()
        assert len(final_gen_scores)==final_gen_codes.shape[0]
        print("Loading scores successful, use the Score Rank Weighted mean as center code.")
        sort_idx = np.argsort( - final_gen_scores)
        weights = rankweight(len(final_gen_scores))
        w_avg_code = weights[np.newaxis,:] @ final_gen_codes
    except Exception as e:
        score_rank_avg = False
        print(e)
        print("Loading scores not successful, use the unweighted mean instead.")

#%% PCA of the Existing code (adopted from Manifold experiment)
sphere_norm = final_gen_norm
print("Set sphere norm to the last generations norm!")
#% Do PCA and find the major trend of evolution
print("Computing PCs")
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(evo_codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1

PC1_vect = PC1_sign * PC_vectors[0,:] 
#%% Prepare the center vector to use in Hessian computation. 
#   Use this vector as the reference vector (center) in the Hessian computation
#   Before Aug. 6th, it's designed to use one code / image from the last generation and explore around it. 
#       this can be dangerous, sometimes one image will lose the major feature that we want in the population. (lose the 2 balls)
#   From Aug. 6th on, we decided to use the mean code from the last generation, which has a higher probability of 
#   From the setting of the bhv2 files find the fixed noise vectors
space_data = loadmat(join(backup_dir, "space_opts.mat"))["space_opts"]
evolspace = space_data[0,threadid]["name"][0]
print("Evolution happens in %s space, load the fixed code in `space_opts`" % evolspace)
if evolspace == "BigGAN_class":
    ref_noise_vec = space_data[0,threadid]['fix_noise_vec']
    #% Choose the mean final generation codes as our reference
    ref_class_vec = final_gen_codes.mean(axis=0, keepdims=True)  # final_gen_codes[0:1, :]
    if score_rank_avg:
        ref_class_vec = w_avg_code
if evolspace == "BigGAN_noise":
    ref_class_vec = space_data[0,threadid]['fix_class_vec']
    ref_noise_vec = final_gen_codes.mean(axis=0, keepdims=True)  # final_gen_codes[0:1, :]
    if score_rank_avg:
        ref_noise_vec = w_avg_code
elif evolspace == "BigGAN":
    ref_vec = final_gen_codes.mean(axis=0, keepdims=True)  # final_gen_codes[0:1, :]
    if score_rank_avg:
        ref_vec = w_avg_code
    ref_noise_vec = ref_vec[:, :128]
    ref_class_vec = ref_vec[:, 128:]
## View image correspond to the reference code
ref_vect = torch.from_numpy(np.concatenate((ref_noise_vec, ref_class_vec), axis=1)).float().cuda()
refimg = G.visualize(ref_vect).cpu()
centimg = ToPILImage()(refimg[0,:,:,:])
centimg.show(title="Center Reference Image")
#%% Visualize the Final Generation  together  with the center reference image. 
VisFinalGen = True
if VisFinalGen:
    #% If you want to regenerate the images from last generation here.
    print("Review the last generation codes w.r.t. the center code for manifold.")
    imgs_final = G.visualize_batch_np(np.concatenate((ref_noise_vec.repeat(25,axis=0), final_gen_codes[:,:]), axis=1))
    ToPILImage()(make_grid(imgs_final,nrow=5)).show()
    #G.visualize(torch.from_numpy(np.concatenate((ref_noise_vec.repeat(5,axis=0), final_gen_codes[:5,:]), axis=1)).float().cuda()).cpu()
    #ToPILImage()(make_grid(imgs.cpu())).show()
#%% Compute Hessian decomposition and get the vectors
Hess_method = "BP"  # "BackwardIter" "ForwardIter"
Hess_all = False # Set to False to reduce computation time. 
t0 = time()
if Hess_method == "BP":
    print("Computing Hessian Decomposition Through auto-grad and full eigen decomposition.")
    classvec = torch.from_numpy(ref_class_vec).float().cuda()  # embed_mat[:, class_id:class_id+1].cuda().T
    noisevec = torch.from_numpy(ref_noise_vec).float().cuda()
    ref_vect = torch.cat((noisevec, classvec, ), dim=1).detach().clone()
    mov_vect = ref_vect.detach().clone().requires_grad_(True)
    #%
    imgs1 = G.visualize(ref_vect)
    if Hess_all:
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
    if Hess_all:
        np.savez(join(summary_dir, "Hess_mat.npz"), H=H, eigvals=eigvals, eigvects=eigvects, 
             H_clas=H_clas, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, 
             H_nois=H_nois, eigvals_nois=eigvals_nois, eigvects_nois=eigvects_nois, 
             vect=ref_vect.cpu().numpy(), noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())
    else:
        np.savez(join(summary_dir, "Hess_mat.npz"), #H=H, eigvals=eigvals, eigvects=eigvects, 
             H_clas=H_clas, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, 
             H_nois=H_nois, eigvals_nois=eigvals_nois, eigvects_nois=eigvects_nois, 
             vect=ref_vect.cpu().numpy(), noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())

elif Hess_method == "BackwardIter":
    print("Computing Hessian Decomposition Through Lanczos decomposition on Backward HVP operator.")
    feat = torch.from_numpy(sphere_norm * PC1_vect).float().requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, ImDist,
        num_eigenthings=128, mode="lanczos", use_gpu=True)
    eigenvals = eigenvals[::-1]
    eigenvecs = eigenvecs[::-1, :]

elif Hess_method == "ForwardIter": 
    print("Computing Hessian Decomposition Through Lanczos decomposition on Forward HVP operator.")
    pass

print("%.2f sec"% (time() - t0))  # 31.75 secs for 300 eig, 87.52 secs for 800 eigs. 
#%% Visualize spectrum
plt.figure(figsize=[8,5])
plt.subplot(1,2,1)
if Hess_all: plt.plot(eigvals[::-1], label="all")
plt.plot(eigvals_clas[::-1], label="class")
plt.plot(eigvals_nois[::-1], label="noise")
plt.ylabel("eigval");plt.legend()
plt.subplot(1,2,2)
if Hess_all: plt.plot(np.log10(eigvals[::-1]), label="all")
plt.plot(np.log10(eigvals_clas[::-1]), label="class")
plt.plot(np.log10(eigvals_nois[::-1]), label="noise")
plt.ylabel("log(eigval)");plt.legend()
plt.savefig(join(summary_dir, "spectrum.jpg"))
#%% Optional: Angle with PC1 vector
if evolspace == "BigGAN_class":
    innerprod2PC1 = PC1_vect @ eigvects_clas.T
elif evolspace == "BigGAN_noise":
    innerprod2PC1 = PC1_vect @ eigvects_nois.T
elif evolspace == "BigGAN":
    innerprod2PC1 = PC1_vect @ eigvects.T
print("Eigen vector: Innerproduct max %.3E min %.3E std %.3E"% (innerprod2PC1.max(), innerprod2PC1.min(), innerprod2PC1.std()))
print("EigenDecomposition of Hessian of Image Similarity Metric\nEigen value: Class space max %.3E min %.3E std %.3E; Noise space max %.3E min %.3E std %.3E"%
      (eigvals_clas.max(), eigvals_clas.min(), eigvals_clas.std(), eigvals_nois.max(), eigvals_nois.min(), eigvals_nois.std(), )) 
if Hess_all: 
    print("EigenDecomposition of Hessian of Image Similarity Metric\nEigen value: All: max %.3E min %.3E std %.3E"%
      (eigvals.max(), eigvals.min(), eigvals.std(),))

#%% Do interpolation along each axes
#%%
if not exact_distance:
    #% Interpolation in the class space, but inversely scale the step size w.r.t. eigenvalue
    codes_all = []
    img_names = []
    scale = 5
    expon = 2.5
    for eigi in [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40,]:#range(20):  # eigvects.shape[1] # 60, 80
        interp_class = LExpMap(classvec.cpu().numpy(), eigvects_clas[:, -eigi-1], 11, (-scale * eigvals_clas[-eigi-1] ** (-1/expon), scale * eigvals_clas[-eigi-1] ** (-1/expon)))
        interp_codes = np.hstack((noisevec.cpu().numpy().repeat(11, axis=0), interp_class, ))
        codes_all.append(interp_codes.copy())
        img_names.extend("class_eig%d_exp%.1f_lin%.1f.jpg"%(eigi, expon, dist) for dist in np.linspace(-scale, scale, 11))

    codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
    img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
    imggrid = make_grid(img_all, nrow=11)
    PILimg2 = ToPILImage()(imggrid)#.show()
    PILimg2.save(join(summary_dir, "eigvect_clas_interp_exp%.1f_d%d.jpg"%(expon, scale)))
    npimgs = img_all.permute([2,3,1,0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
#%
    nrow = 11
    distmat = Hess_img_distmat(ImDist, img_all, nrow=11)
    plt.figure(figsize=[6, 6])
    plt.matshow(distmat, fignum=0)
    plt.colorbar()
    plt.title("Perceptual distance metric along each row\nclass space exponent %.1f Scale%d "%(expon, scale, ))
    plt.savefig(join(summary_dir, "distmat_eigvect_clas_interp_exp%.1f_d%d.jpg"%(expon, scale)))
    plt.show()
    #% Interpolation in the noise space
    codes_all = []
    img_names = []
    scale = 6
    expon = 3
    for eigi in [0, 1, 2, 3, 4, 6, 10, 15, 20, 40]:#range(20):#eigvects_nois.shape[1]
    #    interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-4.5, 4.5))
        interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-scale * eigvals_nois[-eigi-1] ** (-1/expon), scale * eigvals_nois[-eigi-1] ** (-1/expon)))
        interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
        codes_all.append(interp_codes.copy())
        img_names.extend("noise_eig%d_exp%.1f_lin%.1f.jpg"%(eigi, expon, dist) for dist in np.linspace(-scale, scale, 11))

    codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
    img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
    imggrid = make_grid(img_all, nrow=11)
    PILimg3 = ToPILImage()(imggrid)#.show()
    PILimg3.save(join(summary_dir, "eigvect_nois_interp_exp%.1f_d%d.jpg"%(expon, scale)))
    npimgs = img_all.permute([2,3,1,0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
    #
    nrow = 11
    distmat = Hess_img_distmat(ImDist, img_all, nrow=11)
    # Show the image distance from center reference to each image around it.
    plt.figure(figsize=[6,6])
    plt.matshow(distmat, fignum=0)
    plt.colorbar()
    plt.title("Perceptual distance metric along each row\nnoise space  Scale%d exponent %.1f"%(scale, expon))
    plt.savefig(join(summary_dir, "distmat_eigvect_nois_interp_exp%.1f_d%d.jpg"%(expon, scale)))
    plt.show()

else:  # exact_distance by line search
    from ImDist_Line_Searcher import find_level_step
    targ_val = np.array(target_distance)
    ref_vect = torch.from_numpy(np.concatenate((ref_noise_vec, ref_class_vec), axis=1)).float().cuda()
    refimg = G.visualize(ref_vect)
    evc_clas_tsr = torch.from_numpy(eigvects_clas[:, ::-1].copy()).float().cuda()
    evc_nois_tsr = torch.from_numpy(eigvects_nois[:, ::-1].copy()).float().cuda()

    space = "noise"
    imgall = None
    xtick_col = []
    dsim_col = []
    vecs_col = []
    img_names = []
    tick_labels = list(-targ_val[::-1]) + [0] + list(targ_val)  # -0.5, -0.4 ...  0.4, 0.5
    t0 = time()
    eiglist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80]
    for eigid in tqdm(eiglist):  # range(128):  # #
        if space == "class":
            tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid + 1].t()), dim=1)
        elif space == "noise":
            tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid + 1].t(), torch.zeros(1, 128).cuda()), dim=1)
        xtar_pos, ytar_pos, stepimgs_pos = find_level_step(BGAN, ImDist, targ_val, ref_vect, tan_vec, refimg, iter=20,
                                                           pos=True, maxdist=30)
        xtar_neg, ytar_neg, stepimgs_neg = find_level_step(BGAN, ImDist, targ_val, ref_vect, tan_vec, refimg, iter=20,
                                                           pos=False, maxdist=30)
        imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
        xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
        dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
        vecs_row = torch.tensor(xticks_row).float().cuda().view(-1, 1) @ tan_vec + ref_vect

        xtick_col.append(xticks_row)
        dsim_col.append(dsim_row)
        vecs_col.append(vecs_row.cpu().numpy())
        img_names.extend("noise_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in tick_labels)  # dsim_row)
        imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))

    mtg1 = ToPILImage()(make_grid(imgall, nrow=11).cpu())  # 20sec for 13 rows not bad
    mtg1.show()
    mtg1.save(join(summary_dir, "noise_space_all_var.jpg"))
    npimgs = imgall.permute([2, 3, 1, 0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]),
                                                  np.uint8(npimgs[:, :, :, imgi] * 255))
    print(time() - t0)
    # %
    xtick_arr = np.array(xtick_col)
    dsim_arr = np.array(dsim_col)
    vecs_arr = np.array(vecs_col)
    np.savez(join(summary_dir, "noise_ImDist_root_data.npz"), xtick_arr=xtick_arr, dsim_arr=dsim_arr, vecs_arr=vecs_arr,
             targ_val=targ_val, eiglist=eiglist)
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(xtick_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("L2 deviation from center")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Distance Travel Along Given Eigen vector to achieve certain Image Distance")
    plt.savefig(join(summary_dir, "noise_code_deviation.jpg"))
    plt.show()
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(dsim_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Image Distance")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Achieved Image Distance Along Each Axis")
    plt.savefig(join(summary_dir, "noise_space_dist_curv.jpg"))
    plt.show()
    # %
    plt.figure()
    plt.matshow(dsim_arr, fignum=0)
    plt.colorbar()
    plt.title("Perceptual distance metric along each row\nnoise space")
    plt.savefig(join(summary_dir, "noise_space_distmat.jpg"))
    plt.show()
    #%
    space = "class"
    imgall = None
    xtick_col = []
    dsim_col = []
    vecs_col = []
    img_names = []
    tick_labels = list(-targ_val[::-1]) + [0] + list(targ_val)
    t0 = time()
    eiglist = [0, 1, 2, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40, 50, 60, 70, 80]
    for eigid in tqdm(eiglist):  # [0,1,2,3,4,5,6,7,8,10,20,30,
        # 40]:#
        if space == "class":
            tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid + 1].t()), dim=1)
        elif space == "noise":
            tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid + 1].t(), torch.zeros(1, 128).cuda()), dim=1)
        xtar_pos, ytar_pos, stepimgs_pos = find_level_step(BGAN, ImDist, targ_val, ref_vect, tan_vec, refimg, iter=20,
                                                           pos=True, maxdist=30)
        xtar_neg, ytar_neg, stepimgs_neg = find_level_step(BGAN, ImDist, targ_val, ref_vect, tan_vec, refimg, iter=20,
                                                           pos=False, maxdist=30)
        imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
        xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
        dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
        vecs_row = torch.tensor(xticks_row).cuda().view(-1, 1) @ tan_vec + ref_vect
        xtick_col.append(xticks_row)
        dsim_col.append(dsim_row)
        vecs_col.append(vecs_row.cpu().numpy())
        img_names.extend(
            "class_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in tick_labels)  # np.linspace(-0.4, 0.4,11))
        #
        imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))

    mtg2 = ToPILImage()(make_grid(imgall, nrow=11).cpu())  # 20sec for 13 rows not bad
    mtg2.show()
    mtg2.save(join(summary_dir, "class_space_all_var.jpg"))
    npimgs = imgall.permute([2, 3, 1, 0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]),
                                                  np.uint8(npimgs[:, :, :, imgi] * 255))
    print(time() - t0)
    # %
    xtick_arr = np.array(xtick_col)
    dsim_arr = np.array(dsim_col)
    vecs_arr = np.array(vecs_col)
    np.savez(join(summary_dir, "class_ImDist_root_data.npz"), xtick_arr=xtick_arr, dsim_arr=dsim_arr, vecs_arr=vecs_arr,
             targ_val=targ_val)
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(xtick_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("L2 deviation from center")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Distance Travel Along Given Eigen vector to achieve certain Image Distance")
    plt.savefig(join(summary_dir, "class_code_deviation.jpg"))
    plt.show()
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(dsim_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Image Distance")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Achieved Image Distance Along Each Axis")
    plt.savefig(join(summary_dir, "class_space_dist_curv.jpg"))
    plt.show()
    # %
    plt.figure()
    plt.matshow(dsim_arr, fignum=0)
    plt.colorbar()
    plt.savefig(join(summary_dir, "class_space_distmat.jpg"))
    plt.show()
#%%

##% Interpolation in the class space
#codes_all = []
#img_names = []
#for eigi in [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40,]:#range(20):  # eigvects.shape[1] # 60, 80
#    interp_class = LExpMap(classvec.cpu().numpy(), eigvects_clas[:, -eigi-1], 11, (-2.5, 2.5))
#    interp_codes = np.hstack((noisevec.cpu().numpy().repeat(11, axis=0), interp_class, ))
#    codes_all.append(interp_codes.copy())
#    img_names.extend("class_eig%d_lin%.1f.jpg"%(eigi, dist) for dist in np.linspace(-2.5, 2.5, 11))
#codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
#img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
#imggrid = make_grid(img_all, nrow=11)
#PILimg2 = ToPILImage()(imggrid)#.show()
#PILimg2.save(join(summary_dir, "eigvect_clas_interp.jpg"))
#npimgs = img_all.permute([2,3,1,0]).numpy()
#for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
##%%
##% Interpolation in the noise space
#codes_all = []
#img_names = []
#for eigi in [0, 1, 2, 3, 4, 6, 10, 15, 20, 40]:#range(20):#eigvects_nois.shape[1]
#    interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-4.5, 4.5))
#    interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
#    codes_all.append(interp_codes.copy())
#    img_names.extend("noise_eig%d_lin%.1f.jpg"%(eigi, dist) for dist in np.linspace(-4.5, 4.5, 11))
#
#codes_all_arr = np.concatenate(tuple(codes_all), axis=0)
#img_all = G.visualize_batch_np(codes_all_arr, truncation=0.7, B=10)
#imggrid = make_grid(img_all, nrow=11)
#PILimg3 = ToPILImage()(imggrid)#.show()
#PILimg3.save(join(summary_dir, "eigvect_nois_interp.jpg"))
#npimgs = img_all.permute([2,3,1,0]).numpy()
#for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]), np.uint8(npimgs[:,:,:,imgi]*255))
#print("Spent %.1f sec from start" % (time() - T00))
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
#%%
