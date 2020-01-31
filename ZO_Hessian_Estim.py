from matplotlib import use as use_backend
use_backend("Agg")
import matplotlib.pylab as plt
plt.ioff()
# from insilico_Exp import *
from ZO_HessAware_Optimizers import *
import time
import sys
orig_stdout = sys.stdout
# model_unit = ('caffe-net', 'fc6', 1)
# CNN = CNNmodel(model_unit[0])  # 'caffe-net'
# CNN.select_unit(model_unit)
from numpy import sqrt, zeros, abs
from numpy.random import randn
#%%
class HessEstim_Gauss:
    """Code to generate samples and estimate Hessian from it"""
    def __init__(self, space_dimen):
        self.dimen = space_dimen
        self.HB = 0
        self.std = 2

    def GaussSampling(self, xmean, batch=100, std=2):
        xmean = xmean.reshape(1, -1)
        self.std = std
        self.HB = batch
        self.HinnerU = randn(self.HB, self.dimen) # / sqrt(self.dimen)  # make it unit var along the code vector dimension
        H_pos_samples = xmean + self.std * self.HinnerU
        H_neg_samples = xmean - self.std * self.HinnerU
        new_samples = np.concatenate((xmean, H_pos_samples, H_neg_samples), axis=0)
        return new_samples

    def HessEstim(self, scores):
        fbasis = scores[0]
        fpos = scores[-2 * self.HB:-self.HB]
        fneg = scores[-self.HB:]
        weights = abs(
            (fpos + fneg - 2 * fbasis) / 2 / self.std ** 2 / self.HB)  # use abs to enforce positive definiteness
        C = sqrt(weights[:, np.newaxis]) * self.HinnerU  # or the sqrt may not work.
        # H = C^TC + Lambda * I
        self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        # self.HessV.shape = (HB, HB); self.HessD.shape = (HB,), self.HessUC.shape = (HB, dimen)
        # self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        print("Hessian Samples Spectrum", self.HessD)
        print("Hessian Samples Full Power:%f" % ((self.HessD ** 2).sum()))
        return self.HessV, self.HessD, self.HessUC

#%%

# #%%
# unit = ('caffe-net', 'fc6', 1)
# optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
# experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
# experiment.run()
# #%%
# #optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
# experiment2 = ExperimentEvolve(unit, max_step=71)
# experiment2.run()
# experiment2.visualize_trajectory(show=True)
# experiment2.visualize_best(show=True)
# #%%
unit = ('caffe-net', 'fc8', 1)
optim = HessAware_Gauss(4096, population_size=40, lr=2, mu=0.5, Lambda=0.9, Hupdate_freq=5, maximize=True)
experiment3 = ExperimentEvolve(unit, max_step=50, optimizer=optim)
experiment3.run()
experiment3.visualize_trajectory(show=True)
experiment3.visualize_best(show=True)
# #%%
# np.save("tmpcodes.npy", experiment3.codes_all)
# #%%
# t0 = time()
# sample_num = 1000
# meancode = experiment3.codes_all[-5, :]
# HEstim = HessEstim_Gauss(4096)
# codes = HEstim.GaussSampling(meancode, batch=sample_num, std=2)
# #CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
# #CNNmodel.select_unit(model_unit)
# BSize = 100
# scores_all = []
# for i in range(np.int32(np.ceil(codes.shape[0] / BSize))):
#     cur_images = render(codes[i*BSize:min((i+1)*BSize, codes.shape[0]), :])
#     scores = CNNmodel.score(cur_images)
#     scores_all.extend(list(scores))
# scores_all = np.array(scores_all)
# HV, HD, HU = HEstim.HessEstim(scores_all)
# np.savez("HEstim_VDU3.npz", V=HV, D=HD, U=HU, innerU=HEstim.HinnerU, scores=scores, xmean=meancode)
# t1 = time()
# print(t1- t0, 'secs')
# #%%
# imgs = render(experiment3.codes_all[-40:,:])
# experiment3.CNNmodel.score(imgs)
#%%
# #%%
# # Hessian Estimate in Pytorch
# sys.path.append("D:\Github\pytorch-caffe")
# sys.path.append("D:\Github\pytorch-receptive-field")
# from torch_receptive_field import receptive_field, receptive_field_for_unit
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from caffenet import *  # Pytorch-caffe converter
# from hessian import hessian
# print(torch.cuda.current_device())
# # print(torch.cuda.device(0))
# if torch.cuda.is_available():
#     print(torch.cuda.device_count(), " GPU is available:", torch.cuda.get_device_name(0))
# from torch_net_utils import load_caffenet,load_generator
# net = load_caffenet()
# Generator = load_generator()
# import net_utils
# detfmr = net_utils.get_detransformer(net_utils.load('generator'))
# tfmr = net_utils.get_transformer(net_utils.load('caffe-net'))
#
# #%% Pytorch Reference Hessian Computation
# t0 = time()
# blobs = Generator(feat)  # forward the feature vector through the GAN
# out_img = blobs['deconv0']  # get raw output image from GAN
# resz_out_img = F.interpolate(out_img, (224, 224), mode='bilinear', align_corners=True) # Differentiable resizing
# blobs_CNN = net(resz_out_img)
# if len(unit) == 5:
#     neg_activ = - blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
# elif len(unit) == 3:
#     neg_activ = - blobs_CNN[unit[1]][0, unit[2]]
# else:
#     neg_activ = - blobs_CNN['fc8'][0, 1]
# gradient = torch.autograd.grad(neg_activ, feat, retain_graph=True)[0] # First order gradient
# H = hessian(neg_activ, feat, create_graph=False) # Second order gradient
# t1 = time()
# print(t1-t0, " sec, computing Hessian") # Each Calculation may take 1050s esp for deep layer in the network!
# eigval, eigvec = np.linalg.eigh(H.detach().numpy()) # eigen decomposition for a symmetric array! ~ 5.7 s
# g = gradient.numpy()
# g = np.sort(g)
# t2 = time()
# print(t2-t1, " sec, eigen factorizing hessian")
# np.savez(join(output_dir, "hessian_result_%s_%d.npz"%(unit[1], unit[2])),
#          z=feat.detach().numpy(),
#          activation=-neg_activ.detach().numpy(),
#          grad=gradient.numpy(),H=H.detach().numpy(),
#          heig=eigval,heigvec=eigvec)