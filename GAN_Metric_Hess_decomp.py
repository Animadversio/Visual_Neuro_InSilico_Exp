"""
Demo code for computing the Hessian of Neuron's tuning w.r.t underlying code
Or The Hessian of image dissimilarity metric w.r.t. underlying code
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings
#%%
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
#%% Prepare the Networks:
# #    Perceptual Similarity Net
#      Gan
#      alexnet or VGG
import sys
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
sys.path.append(r"D:\Github\PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda() # this notation is incorrect in older pytorch

import torchvision as tv
# VGG = tv.models.vgg16(pretrained=True)
alexnet = tv.models.alexnet(pretrained=True).cuda()
for param in alexnet.parameters():
    param.requires_grad_(False)
#%% Set up hook and the linear network based on the CNN
from FeatLinModel import FeatLinModel

#%% Compute the full hessian
from hessian_eigenthings.utils import progress_bar
def get_full_hessian(loss, param):
    # from https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    # modified from hessian_eigenthings repo. api follows hessian.hessian
    hessian_size = param.size(0)
    hessian = torch.zeros(hessian_size, hessian_size)
    loss_grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
    for idx in range(hessian_size):
        progress_bar(
            idx, hessian_size, "full hessian columns: %d of %d" % (idx, hessian_size)
        )
        grad2rd = torch.autograd.grad(loss_grad[idx], param, create_graph=False, retain_graph=True, only_inputs=True)
        hessian[idx] = grad2rd[0].view(-1)
    return hessian.cpu().data.numpy()

def torch_corr(vec1, vec2):
    return torch.mean((vec1 - vec1.mean()) * (vec2 - vec2.mean())) / vec1.std(unbiased=False) / vec2.std(unbiased=False)
#%%
summary_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
def tuning_plot(G, preprocess, objective, eigvals, eigvects, eig_id_arr=(0, 1, 5, 10, 15, 20, 40, 60, 80,99,150,200,250,299,450),
        save_indiv=False, save_row=False, ticks=21, summary_dir=summary_dir, veclabel="eig", titlestr="",
        pad=24, cmap=plt.cm.viridis):
    RND = np.random.randint(100)
    vec_norm = feat.norm().item()
    ref_vect = (feat / vec_norm).cpu().numpy()
    theta_arr_deg =  np.linspace(-90, 90, ticks) # np.arange(-5, 6)
    theta_arr = theta_arr_deg / 180 * np.pi
    img_list_all = []
    scores_col = [] # array version of scores
    scores_all = [] # list version of scores
    # eig_id_arr = [0, 1, 5, 10, 15, 20, 40, 60, 80,99,150,200,250,299,450]
    for eig_id in eig_id_arr: #,600,799]:
        # eig_id = 0
        perturb_vect = eigvects[eig_id,:]  # PC_vectors[1,:]
        codes_arc = np.array([np.cos(theta_arr),
                              np.sin(theta_arr) ]).T @ np.array([ref_vect, perturb_vect])
        norms = np.linalg.norm(codes_arc, axis=1)
        codes_arc = codes_arc / norms[:, np.newaxis] * vec_norm
        imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
        scores = - objective(preprocess(imgs), scaler=False)
        scores_col.append(scores.cpu().numpy())
        scores_all.extend(scores.cpu().squeeze().tolist())
        npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()

        if save_indiv:
            for i in range(npimgs.shape[3]):
                angle = theta_arr_deg[i]
                imwrite(join(newimg_dir, "norm%d_%s%d_ang%d.jpg" % (vec_norm, veclabel, eig_id, angle)), npimgs[:, :, :, i])

        img_list = [npimgs[:, :, :, i] for i in range(npimgs.shape[3])]
        img_list_all.extend(img_list)
        if save_row:
            mtg1 = build_montages(img_list, [256, 256], [len(theta_arr), 1])[0]
            imwrite(join(summary_dir, "norm%d_%s_%d.jpg" % (vec_norm, veclabel, eig_id)), mtg1)
    mtg_all = build_montages(img_list_all, [256, 256], [len(theta_arr), int(len(img_list_all) // len(theta_arr))])[0]
    imwrite(join(summary_dir, "norm%d_%s_all_opt_%d.jpg" % (vec_norm, veclabel, RND)), mtg_all)
    print("Write to ", join(summary_dir, "norm%d_%s_all_opt_%d.jpg" % (vec_norm, veclabel, RND)))

    mtg_frm = color_framed_montages(img_list_all, [256, 256], [len(theta_arr), int(len(img_list_all) // len(theta_arr))], scores_all, pad=pad, cmap=cmap)[0]
    imwrite(join(summary_dir, "norm%d_%s_all_opt_framed_%d.jpg" % (vec_norm, veclabel, RND)), mtg_frm)
    print("Write to ", join(summary_dir, "norm%d_%s_all_opt_framed_%d.jpg" % (vec_norm, veclabel, RND)))

    scores_col = np.array(scores_col)
    plt.matshow(scores_col)
    plt.axis('image')
    plt.title("Neural Tuning Towards Different Eigen Vectors of Activation")
    plt.xlabel("Angle")
    plt.ylabel("Eigen Vector #")
    eiglabel = ["%d %.3f"%(id,eig) for id, eig in zip(eig_id_arr, eigvals[eig_id_arr])]
    plt.yticks(range(len(eig_id_arr)), eiglabel) # eig_id_arr
    plt.ylim(top=-0.5, bottom=len(eig_id_arr) - 0.5)
    plt.colorbar()
    plt.suptitle(titlestr)
    plt.savefig(join(summary_dir, "norm%d_%s_score_mat_%02d.jpg" % (vec_norm, veclabel, RND)) )
    plt.show()
    print("Write to ", join(summary_dir, "norm%d_%s_score_mat_%02d.jpg" % (vec_norm, veclabel, RND)) )

#%%  Apply the Approximate Forward Differencing
#%%  Test the new forward mode HVP computation (see test_Hess_Decomp.py)
#%%  Simulated activation maximization
from torchvision.transforms import Normalize, Compose
RGB_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
RGB_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()
preprocess = Compose([lambda img: (F.interpolate(img, (224, 224), mode='bilinear', align_corners=True) - RGB_mean) / RGB_std])
# weight = torch.randn(256, 13, 13).cuda()
# objective = FeatLinModel(alexnet, layername='features_10', type="weight", weight=weight)
objective = FeatLinModel(alexnet, layername='features_10', type="neuron", chan=slice(None), pos=(10, 10))
feat = 5 * torch.randn(4096).cuda()
activHVP = GANForwardHVPOperator(G, feat, objective, preprocess=preprocess)
activHVP.apply(1*torch.randn(4096).requires_grad_(False).cuda())
#%% Optimize the hidden code
feat = 5*torch.randn(4096).cuda()
feat.requires_grad_(True)
optimizer = optim.Adam([feat], lr=5e-2)
for step in range(200):
    optimizer.zero_grad()
    obj = objective(preprocess(G.visualize(feat)))
    obj.backward()
    optimizer.step()
    if np.mod((step + 1), 10) == 0:
        print("step %d: %.2f"%(step, obj.item()))
#%%
feat.requires_grad_(False)
activHVP = GANForwardHVPOperator(G, feat, objective, preprocess=preprocess)
activHVP.apply(1*torch.randn(4096).requires_grad_(False).cuda())
#%%
t0 = time()
eigvals, eigvects = lanczos(activHVP, num_eigenthings=2000, use_gpu=True)
print(time() - t0)  # 40 sec 146sec for 2000 eigens
eigvals = eigvals[::-1]
eigvects = eigvects[::-1, :]

#%% compute vHv for all
metricHVP = GANHVPOperator(G, feat, model_squ)
vHv_arr = np.zeros_like(eigvals)
vHv_arr.fill(np.nan)
for i, vec in enumerate(eigvects):
    vHv_arr[i] = metricHVP.vHv_form(torch.tensor(vec).cuda())
#%%
vHv_log = np.log10(np.abs(vHv_arr))
eig_log = np.log10(np.abs(eigvals))
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(eig_log[eig_log > -8].reshape(-1, 1), vHv_log[eig_log > -8].reshape(-1, 1))
residue = (vHv_log - reg.predict(eig_log.reshape(-1,1)).reshape(-1))
#%%
sort_idx = residue.argsort()
eig_id_arr = sort_idx[-10:]
tuning_plot(G, preprocess, objective, eigvals, eigvects, eig_id_arr, veclabel="top_inv", ticks=11)
#%%
sort_idx_flt = sort_idx[eig_log[sort_idx] > -8]
eig_id_arr = sort_idx_flt[:10]
tuning_plot(G, preprocess, objective, eigvals, eigvects, eig_id_arr, veclabel="low_inv", ticks=21)
#%%
sort_idx_flt = sort_idx[(eig_log[sort_idx] > -2.9) & (vHv_log[sort_idx] > -3.1)]
eig_id_arr = sort_idx_flt[-10:]
tuning_plot(G, preprocess, objective, eigvals, eigvects, eig_id_arr, veclabel="top_inv", ticks=21, titlestr="top log(eig) - log(metric vHv) vectors for both > 1E-3")

#%%
# plt.scatter(eig_log, reg.predict(eig_log.reshape(-1,1)))
plt.scatter(eig_log, vHv_log, alpha=0.5, c=residue)
plt.xlabel("Eigenvalue of Activation function (log)")
plt.ylabel("vHv of metric function (log)")
plt.show()
#%%
RND = np.random.randint(100)
savedir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
plt.scatter((np.abs(eigvals)), (np.abs(vHv_arr)))
plt.xlabel("Eigenvalue of Activation function")
plt.ylabel("vHv of metric function")
plt.savefig(join(savedir, "Activ-vHv-Ratio_lin_%02d.jpg"%(RND)))
plt.show()
#%
plt.scatter(np.log10(np.abs(eigvals)), np.log10(np.abs(vHv_arr)))
plt.xlabel("Eigenvalue of Activation function (log)")
plt.ylabel("vHv of metric function (log)")
plt.savefig(join(savedir, "Activ-vHv-Ratio_log_%02d.jpg"%(RND)))
plt.show()
#%%
plt.scatter(np.log10(np.abs(eigvals)), np.log10(np.abs(vHv_arr)), c=np.log10(np.abs(vHv_arr)) - np.log10(np.abs(eigvals)))
plt.xlabel("Eigenvalue of Activation function (log)")
plt.ylabel("vHv of metric function (log)")
# plt.savefig(join(savedir, "Activ-vHv-Ratio_log_%02d.jpg"%(RND)))
plt.show()
#%%
feat.requires_grad_(False)
metricHVP = GANHVPOperator(G, feat, model_squ)
t0 = time()
eigvals, eigvects = lanczos_generalized(metricHVP, metric_operator=activHVP, num_eigenthings=1, tol=1E-1, max_steps=20, use_gpu=True)
print(time() - t0)  # 40 sec
eigvals = eigvals[::-1]
eigvects = eigvects[::-1, :]

#%%
feat.requires_grad_(False)
metricHVP = GANHVPOperator(G, feat, model_squ)
t0 = time()
eigvals, eigvects = lanczos_generalized(activHVP, metric_operator=metricHVP, num_eigenthings=500, use_gpu=True)
print(time() - t0)  # 40 sec
eigvals = eigvals[::-1]
eigvects = eigvects[::-1, :]
#%%
summary_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
#%% Visualize the perturbed codes.
RND = np.random.randint(100)
ref_vect = (feat / feat.norm()).cpu().numpy()
save_indiv = False
save_row = False
vec_norm = feat.norm().item()
ang_step = 180 / 10
theta_arr_deg = ang_step * np.linspace(-5, 5, 21)# np.arange(-5, 6)
theta_arr = theta_arr_deg / 180 * np.pi
img_list_all = []
scores_col = []
eig_id_arr = [0, 1, 5, 10, 15, 20, 40, 60, 80,99,150,200,250,299,450]
for eig_id in eig_id_arr:#,600,799]:
    # eig_id = 0
    perturb_vect = eigvects[eig_id,:]  # PC_vectors[1,:]
    codes_arc = np.array([np.cos(theta_arr),
                          np.sin(theta_arr) ]).T @ np.array([ref_vect, perturb_vect])
    norms = np.linalg.norm(codes_arc, axis=1)
    codes_arc = codes_arc / norms[:, np.newaxis] * vec_norm
    imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
    scores = - objective(preprocess(imgs), scaler=False)
    scores_col.append(scores.cpu().numpy())
    npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()

    if save_indiv:
        for i in range(npimgs.shape[3]):
            angle = theta_arr_deg[i]
            imwrite(join(newimg_dir, "norm%d_eig%d_ang%d.jpg" % (vec_norm, eig_id, angle)), npimgs[:, :, :, i])

    img_list = [npimgs[:, :, :, i] for i in range(npimgs.shape[3])]
    img_list_all.extend(img_list)
    if save_row:
        mtg1 = build_montages(img_list, [256, 256], [len(theta_arr), 1])[0]
        imwrite(join(summary_dir, "norm%d_eig_%d.jpg" % (vec_norm, eig_id)), mtg1)
mtg_all = build_montages(img_list_all, [256, 256], [len(theta_arr), int(len(img_list_all) // len(theta_arr))])[0]
imwrite(join(summary_dir, "norm%d_eig_all_opt_%d.jpg" % (vec_norm, RND)), mtg_all)
#%
scores_col = np.array(scores_col)
plt.matshow(scores_col)
plt.axis('image')
plt.title("Neural Tuning Towards Different Eigen Vectors of Activation")
plt.xlabel("Angle")
plt.ylabel("Eigen Vector #")
eiglabel = ["%d %.3f"%(id,eig) for id, eig in zip(eig_id_arr, eigvals[eig_id_arr])]
plt.yticks(range(len(eig_id_arr)), eiglabel) # eig_id_arr
plt.ylim(top=-0.5, bottom=len(eig_id_arr) - 0.5)
plt.colorbar()
plt.savefig(join(summary_dir, "norm%d_score_mat_%02d.jpg" % (vec_norm, RND)) )
plt.show()
#%%
scores_col = []
for eig_id in [0,1,5,10,15,20,40,60,80,99,150,200,250,299,450,600,799]:
    # eig_id = 0
    perturb_vect = eigvects[eig_id,:] # PC_vectors[1,:]
    codes_arc = np.array([np.cos(theta_arr),
                          np.sin(theta_arr) ]).T @ np.array([ref_vect, perturb_vect])
    norms = np.linalg.norm(codes_arc, axis=1)
    codes_arc = codes_arc / norms[:,np.newaxis] * vec_norm
    imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
    scores = objective(F.interpolate(imgs, (224, 224), mode='bilinear', align_corners=True), scaler=False)
    scores_col.append(scores.cpu().numpy())

scores_col = np.array(scores_col)
#%%
plt.matshow(scores_col)
plt.axis('image')
plt.colorbar()
plt.show()
#%%
#%% Try to compute the full Hessian first and use its inverse to investigate
#%  Fast method of computing Hessian for underlying metric
perturb_vect = 1E-6 * torch.randn(4096).float().cuda()
perturb_vect.requires_grad_(True)
d_sim = model_squ(G.visualize(feat), G.visualize(feat+perturb_vect))
t0 = time()
H = get_full_hessian(d_sim, perturb_vect) # it's pretty symmetric
print(time() - t0)  # 362 sec
#%%
Hinv = np.linalg.pinv(H, hermitian=True)  # Wall time: 32.3 s
#%%
t0 = time()
eigvals_g, eigvects_g = lanczos_generalized(activHVP, metric_operator=H, metric_inv_operator=Hinv, num_eigenthings=100, use_gpu=True, max_steps=40)
print(time() - t0)
#%%
%time met_eigval, met_eigvec = np.linalg.eigh(H)
#%% Cut off the spectrum of H and find the pseudo inverse of H by suppressing the null space of it.
cutoff = 1e-5
msk = np.abs(met_eigval) < cutoff
met_eigval_cutoff = met_eigval.copy()
met_eigval_cutoff[msk] = 0
met_eiginv_cutoff = met_eigval_cutoff.copy()
met_eiginv_cutoff[~msk] = 1 / met_eigval_cutoff[~msk] #

Hpinv = met_eigvec @ np.diag(met_eiginv_cutoff) @ met_eigvec.T
# np.diag(H @ Hpinv)
#%%
np.sum(np.round(met_eiginv_cutoff * met_eigval_cutoff))
#%%
# (np.mean(np.abs(H - met_eigvec @ np.diag(met_eigval) @ met_eigvec.T)))
plt.figure()
plt.hist(np.log10(np.abs(met_eigval)), bins=20)
plt.show()
#%%
t0 = time()
eigvals_g, eigvects_g = lanczos_generalized(activHVP, metric_operator=H, metric_inv_operator=Hpinv, num_eigenthings=300, use_gpu=True, max_steps=60, tol=1E-5)
print(time() - t0)
#%%

tuning_plot(G, preprocess, objective, eigvals_g, eigvects_g, eig_id_arr=[5,6,7,8,9,10,11,12])