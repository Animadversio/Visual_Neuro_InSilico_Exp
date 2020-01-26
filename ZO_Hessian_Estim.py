from insilico_Exp import *
plt.ioff()
import matplotlib

model_unit = ('caffe-net', 'fc6', 1)
CNN = CNNmodel(model_unit[0])  # 'caffe-net'
CNN.select_unit(model_unit)
from numpy import sqrt, zeros, abs
from numpy.random import randn

class HessAware_ADAM:
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, nu=0.9, maximize=True):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of estimating gradient
        self.nu = nu  # update rate for D
        self.lr = lr  # learning rate (step size) of moving along gradient
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
class HessAware_Gauss:
    """Gaussian Sampling method for estimating Hessian"""
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, Lambda=0.9, Hupdate_freq=5, maximize=True):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        self.Lambda = Lambda  # diagonal regularizer for Hessian matrix
        self.lr = lr  # learning rate (step size) of moving along gradient
        self.grad = np.zeros((1, self.dimen))  # estimated gradient
        self.innerU = np.zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = np.zeros((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        self.xcur = np.zeros((1, self.dimen))  # current base point
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.fcur = 0  # f(xcur)
        self.fnew = 0  # f(xnew)
        self.Hupdate_freq = int(Hupdate_freq)  # Update Hessian (add additional samples every how many generations)
        self.HB = population_size  # Batch size of samples to estimate Hessian, can be different from self.B
        self.HinnerU = np.zeros((self.HB, self.dimen))  # sample deviation vectors for Hessian construction
        # SVD of the weighted HinnerU for Hessian construction
        self.HessUC = np.zeros((self.HB, self.dimen))  # Basis vector for the linear subspace defined by the samples
        self.HessD  = np.zeros(self.HB)  # diagonal values of the Lambda matrix
        self.HessV  = np.zeros((self.HB, self.HB))  # seems not used....
        self.HUDiag = np.zeros(self.HB)
        self.hess_comp = False
        self._istep = 0  # step counter
        self.maximize = maximize  # maximize / minimize the function

    def step_hessian(self, scores):
        '''Currently only use part of the samples to estimate hessian, maybe need more '''
        fbasis = scores[0]
        fpos = scores[-2*self.HB:-self.HB]
        fneg = scores[-self.HB:]
        weights = abs((fpos + fneg - 2 * fbasis) / 2 / self.mu ** 2 / self.HB)  # use abs to enforce positive definiteness
        C = sqrt(weights[:, np.newaxis]) * self.HinnerU  # or the sqrt may not work.
        # H = C^TC + Lambda * I
        self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        print("Hessian Samples Spectrum", self.HessD)
        print("Hessian Samples Full Power:%f \nLambda:%f" % ((self.HessD ** 2).sum(), self.Lambda) )


    def step_simple(self, scores, codes):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        # set short name for everything to simplify equations
        N = self.dimen
        if self.hess_comp:  # if this flag is True then more samples have been added to the trial
            self.step_hessian(scores)
            # you should only get images for gradient estimation, get rid of the Hessian samples, or make use of it to estimate gradient
            codes = codes[:self.B+1, :]
            scores = scores[:self.B+1]
            self.hess_comp = False

        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            self.xcur = codes[0:1, :]
            self.xnew = codes[0:1, :]
        else:
            # self.xcur = self.xnew # should be same as following line
            self.xcur = codes[0:1, :]
            self.weights = (scores - scores[0]) / self.mu
            # estimate gradient from the codes and scores
            HAgrad = self.weights[1:] @ (codes[1:] - self.xcur) / self.B  # it doesn't matter if it includes the 0 row!
            print("Estimated Gradient Norm %f"%np.linalg.norm(HAgrad))
            if self.maximize is True:
                self.xnew = self.xcur + self.lr * HAgrad  # add - operator it will do maximization.
            else:
                self.xnew = self.xcur - self.lr * HAgrad
        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.B + 1, N))
        self.innerU = randn(self.B, N)  # Isotropic gaussian distributions
        self.outerV = self.innerU / sqrt(self.Lambda) + ((self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC # H^{-1/2}U
        new_samples[0:1, :] = self.xnew
        new_samples[1: , :] = self.xnew + self.mu * self.outerV  # m + sig * Normal(0,C)
        if self._istep % self.Hupdate_freq == 0:
            # add more samples to next batch for hessian computation
            self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((new_samples, H_pos_samples, H_neg_samples), axis=0)
        self._istep += 1
        return new_samples
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
unit = ('caffe-net', 'fc6', 1)
optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
experiment.run()
#%%
#optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
experiment2 = ExperimentEvolve(unit, max_step=71)
experiment2.run()
experiment2.visualize_trajectory(show=True)
experiment2.visualize_best(show=True)
#%%
unit = ('caffe-net', 'fc8', 1)
optim = HessAware_Gauss(4096, population_size=40, lr=2, mu=0.5, Lambda=0.9, Hupdate_freq=5, maximize=True)
experiment3 = ExperimentEvolve(unit, max_step=50, optimizer=optim)
experiment3.run()
experiment3.visualize_trajectory(show=True)
experiment3.visualize_best(show=True)
#%%
np.save("tmpcodes.npy", experiment3.codes_all)
#%%
t0 = time()
sample_num = 1000
meancode = experiment3.codes_all[-5, :]
HEstim = HessEstim_Gauss(4096)
codes = HEstim.GaussSampling(meancode, batch=sample_num, std=2)
#CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
#CNNmodel.select_unit(model_unit)
BSize = 100
scores_all = []
for i in range(np.int32(np.ceil(codes.shape[0] / BSize))):
    cur_images = render(codes[i*BSize:min((i+1)*BSize, codes.shape[0]), :])
    scores = CNNmodel.score(cur_images)
    scores_all.extend(list(scores))
scores_all = np.array(scores_all)
HV, HD, HU = HEstim.HessEstim(scores_all)
np.savez("HEstim_VDU3.npz", V=HV, D=HD, U=HU, innerU=HEstim.HinnerU, scores=scores, xmean=meancode)
t1 = time()
print(t1- t0, 'secs')
#%%
imgs = render(experiment3.codes_all[-40:,:])
experiment3.CNNmodel.score(imgs)
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