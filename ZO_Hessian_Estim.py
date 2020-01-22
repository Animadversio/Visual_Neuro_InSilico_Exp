from insilico_Exp import *
plt.ioff()
import matplotlib

model_unit = ('caffe-net', 'fc6', 1)
CNN = CNNmodel(model_unit[0])  # 'caffe-net'
CNN.select_unit(model_unit)
from numpy import sqrt, zeros
from numpy.random import randn

class HessAware_ADAM:
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, nu=0.9, maximize=True):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # sample step size, just like sigma
        self.nu = nu  # update learning rate for D
        self.lr = lr  # learning rate of moving along gradient
        self.grad = np.zeros((1, self.dimen))  # estimated gradient
        self.D = np.ones((1, self.dimen))  # running average of gradient square
        self.Hdiag = np.ones((1, self.dimen))  # Diagonal of estimated Hessian
        self.innerU = np.zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = np.zeros((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}
        self.xcur = np.zeros((1, self.dimen))  # current base point
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.fcur = 0  # f(xcur)
        self.fnew = 0  # f(xnew)
        self._istep = 0  # step counter
        self.maximize = maximize  # maximize / minimize the function

    def step_simple(self, scores, codes):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        # set short name for everything to simplify equations
        N = self.dimen
        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            self.xcur = codes[0:1, :]
            self.xnew = codes[0:1, :]
        else:
            # self.xcur = self.xnew # should be same as following
            self.xcur = codes[0:1, :]
            self.weights = (scores - scores[0]) / self.mu

            HAgrad = self.weights[1:] @ (codes[1:] - self.xcur) / self.B  # it doesn't matter if it includes the 0 row!
            if self.maximize is True:
                self.xnew = self.xcur + self.lr * HAgrad  # add - operator it will do maximization.
            else:
                self.xnew = self.xcur - self.lr * HAgrad
            self.D = self.nu * self.D + (1 - self.nu) * HAgrad # running average of gradient square
            self.Hdiag = self.D / (1 - self.nu ** self._istep)  # Diagonal of estimated Hessian

        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.B + 1, N))
        self.innerU = randn(self.B, N)  # save the random number for generating the code.
        self.outerV = self.innerU / sqrt(self.Hdiag)  # H^{-1/2}U
        new_samples[0:1, :] = self.xnew
        new_samples[1: , :] = self.xnew + self.mu * self.outerV  # m + sig * Normal(0,C)
        self._istep += 1
        return new_samples
#%%
unit = ('caffe-net', 'fc6', 1)
optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
experiment.run()
#%%
#optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
experiment2 = ExperimentEvolve(unit, max_step=100)
experiment2.run()
experiment2.visualize_trajectory(show=True)
experiment2.visualize_best(show=True)

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