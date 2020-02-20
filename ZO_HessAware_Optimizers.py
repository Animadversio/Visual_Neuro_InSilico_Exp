from matplotlib import use as use_backend
use_backend("Agg")
import matplotlib.pylab as plt
plt.ioff()
from insilico_Exp import *
import time
import sys
import utils
import numpy as np
from numpy.linalg import norm
import os
orig_stdout = sys.stdout
# model_unit = ('caffe-net', 'fc6', 1)
# CNN = CNNmodel(model_unit[0])  # 'caffe-net'
# CNN.select_unit(model_unit)
from numpy import sqrt, zeros, abs, floor, log
from numpy.random import randn

#%%
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
            self.D = self.nu * self.D + (1 - self.nu) * HAgrad ** 2  # running average of gradient square # Missing square before
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
def rankweight(lambda_, mu=None):
    """ Rank weight inspired by CMA-ES code
    mu is the cut off number, how many samples will be kept while `lambda_ - mu` will be ignore
    """
    if mu is None:
        mu = lambda_ / 2  # number of parents/points for recombination
        #  Defaultly Select half the population size as parents
    weights = zeros(int(lambda_))
    mu_int = int(floor(mu))
    weights[:mu_int] = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
    weights = weights / sum(weights)
    return weights

class HessAware_Gauss_Spherical:
    """Gaussian Sampling method for estimating Hessian"""
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, Lambda=0.9, Hupdate_freq=5, 
            sphere_norm=300, maximize=True, rankweight=False):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        self.Lambda = Lambda  # diagonal regularizer for Hessian matrix
        self.lr = lr  # learning rate (step size) of moving along gradient
        self.sphere_norm = sphere_norm
        self.tang_codes = zeros((self.B, self.dimen))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, self.dimen))
        self.ps = zeros((1, self.dimen))  # evolution paths for C and sigma
        self.weights = np.log(mu + 1 / 2) - (np.log(np.arange(1, 1 + np.floor(mu))))  # muXone array for weighted recombination
        self.fmu = int(np.floor(mu))
        self.weights = self.weights / sum(self.weights)  # normalize recombination weights array
        self.weights.shape = (1, -1)  # Add the 1st dim 1 to the weights mat

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
        self.rankweight = rankweight # Switch between using raw score as weight VS use rank weight as score
        print("Spereical Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\n Learning rate: %.3f"
              % (self.dimen, self.B, self.mu, self.lr))
    
    def step_hessian(self, scores):
        '''Currently not implemented in Spherical Version.'''
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
            print('First generation\n')
            self.xcur = codes[0:1, :]
            self.xnew = codes[0:1, :]
            # No reweighting as there should be a single code
        else:
            # self.xcur = self.xnew # should be same as following line
            self.xcur = codes[0:1, :]
            if self.rankweight is False: # use the score difference as weight
                # B normalizer should go here larger cohort of codes gives more estimates 
                self.weights = (scores[1:] - scores[0]) / self.B # / self.mu 
            else:  # use a function of rank as weight, not really gradient. 
                if self.maximize is False: # note for weighted recombination, the maximization flag is here. 
                    code_rank = np.argsort(np.argsort( scores[1:]))  # add - operator it will do maximization.
                else:
                    code_rank = np.argsort(np.argsort(-scores[1:]))
                # Consider do we need to consider the basis code and score here? Or no? 
                # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
                self.weights = rankweight(len(scores)-1)[code_rank] # map the rank to the corresponding weight of recombination

            # estimate gradient from the codes and scores
            # HAgrad = self.weights[1:] @ (codes[1:] - self.xcur) / self.B  # it doesn't matter if it includes the 0 row!
            HAgrad = self.weights[np.newaxis, :] @ self.tang_codes
            print("Estimated Gradient Norm %f" % np.linalg.norm(HAgrad))
            if self.rankweight is False:
                if self.maximize is True:
                    self.xnew = ExpMap(self.xcur,   self.lr * HAgrad) # add - operator it will do maximization.
                else:
                    self.xnew = ExpMap(self.xcur, - self.lr * HAgrad)
            else:
                self.xnew = ExpMap(self.xcur,   self.lr * HAgrad) 
            # vtan_new = VecTransport(self.xcur, self.xnew, vtan_old)
            # uni_vtan_old = vtan_old / np.linalg.norm(vtan_old);
            # uni_vtan_new = vtan_new / np.linalg.norm(vtan_new);  # uniform the tangent vector

        # Generate new sample by sampling from Gaussian distribution
        self.tang_codes = zeros((self.B, N))  # Tangent vectors of exploration
        new_samples = zeros((self.B + 1, N))
        self.innerU = randn(self.B, N)  # Isotropic gaussian distributions
        self.outerV = self.innerU / sqrt(self.Lambda) + ((self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC # H^{-1/2}U
        new_samples[0:1, :] = self.xnew
        self.tang_codes[: , :] = self.mu * self.outerV  # m + sig * Normal(0,C)
        new_samples[1:, ] = ExpMap(self.xnew, self.tang_codes)
        if (self._istep + 1) % self.Hupdate_freq == 0:
            # add more samples to next batch for hessian computation
            self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((new_samples, H_pos_samples, H_neg_samples), axis=0)
        self._istep += 1
        self._curr_samples = new_samples / norm(new_samples, axis=1)[:, np.newaxis] * self.sphere_norm
        return self._curr_samples

def ExpMap(x, tang_vec, EPS = 1E-4):
    angle_dist = sqrt((tang_vec ** 2).sum(axis=1))  # vectorized
    angle_dist = angle_dist[:, np.newaxis]
    print("Angular distance for Exponentiation ", angle_dist[:,0])
    uni_tang_vec = tang_vec / angle_dist
    # x = repmat(x, size(tang_vec, 1), 1); # vectorized
    xnorm = np.linalg.norm(x)
    assert(xnorm > EPS, "Exponential Map from a basis point at origin is degenerate, examine the code. (May caused by 0 initialization)")
    y = (np.cos(angle_dist) @ (x[:] / xnorm) + np.sin(angle_dist) * uni_tang_vec) * xnorm
    return y

def VecTransport(xold, xnew, v):
    xold = xold / np.linalg.norm(xold)
    xnew = xnew / np.linalg.norm(xnew)
    x_symm_axis = xold + xnew
    v_transport = v - 2 * v @ x_symm_axis.T / np.linalg.norm(
        x_symm_axis) ** 2 * x_symm_axis  # Equation for vector parallel transport along geodesic
    # Don't use dot in numpy, it will have wierd behavior if the array is not single dimensional
    return v_transport
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
def radial_proj(codes, max_norm):
    if max_norm is np.inf:
        return codes
    else:
        assert max_norm >= 0
        code_norm = np.linalg.norm(codes, axis=1)
        proj_norm = np.minimum(code_norm, max_norm)
        return codes / code_norm[:, np.newaxis] * proj_norm[:, np.newaxis]

class HessAware_Gauss_DC:
    """Gaussian Sampling method for estimating Hessian"""
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, Lambda=0.9, Hupdate_freq=5, 
        maximize=True, max_norm=300, rankweight=False, nat_grad=False):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        self.Lambda = Lambda  # diagonal regularizer for Hessian matrix
        self.lr = lr  # learning rate (step size) of moving along gradient
        self.grad = np.zeros((1, self.dimen))  # estimated gradient
        self.innerU = np.zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = np.zeros((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.xscore = 0
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
        self.code_stored = np.array([]).reshape((0, self.dimen))
        self.score_stored = np.array([])
        self.N_in_samp = 0
        self.max_norm = max_norm
        self.nat_grad = nat_grad # use the natural gradient definition, or normal gradient. 
        self.rankweight = rankweight
        
    def new_generation(self, init_score, init_code):
        self.xscore = init_score
        self.score_stored = np.array([])
        self.xnew = init_code
        self.code_stored = np.array([]).reshape((0, self.dimen))
        self.N_in_samp = 0

    def compute_hess(self, scores, Lambda_Frac=100):
        '''Currently only use part of the samples to estimate hessian, maybe need more '''
        fbasis = self.xscore
        fpos = scores[:self.HB]
        fneg = scores[-self.HB:]
        weights = abs((fpos + fneg - 2 * fbasis) / 2 / self.mu ** 2 / self.HB)  # use abs to enforce positive definiteness
        C = sqrt(weights[:, np.newaxis]) * self.HinnerU  # or the sqrt may not work.
        # H = C^TC + Lambda * I
        self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        self.Lambda = (self.HessD ** 2).sum() / Lambda_Frac
        self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        print("Hessian Samples Spectrum", self.HessD)
        print("Hessian Samples Full Power:%f \nLambda:%f" % ((self.HessD ** 2).sum(), self.Lambda) )

    def compute_grad(self, scores):
        # add the new scores to storage
        self.score_stored = np.concatenate((self.score_stored, scores), axis=0) if self.score_stored.size else scores
        if self.rankweight is False: # use the score difference as weight
            # B normalizer should go here larger cohort of codes gives more estimates 
            self.weights = (self.score_stored - self.xscore) / self.score_stored.size # / self.mu 
            # assert(self.N_in_samp == self.score_stored.size)
        else:  # use a function of rank as weight, not really gradient. 
            # Note descent check **could be** built into ranking weight? 
            # If not better just don't give weights to that sample 
            if self.maximize is False: # note for weighted recombination, the maximization flag is here. 
                code_rank = np.argsort(np.argsort( self.score_stored))  # add - operator it will do maximization.
            else:
                code_rank = np.argsort(np.argsort(-self.score_stored))
            # Consider do we need to consider the basis code and score here? Or no? 
            # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
            self.weights = rankweight(len(scores)-1,mu=20)[code_rank] # map the rank to the corresponding weight of recombination
            # only keep the top 20 codes and recombine them. 

        if self.nat_grad: # if or not using the Hessian to rescale the codes 
            hagrad = self.weights @ (self.code_stored - self.xnew) # /self.mu
        else:
            Hdcode = self.Lambda * (self.code_stored - self.xnew) + (
                    ((self.code_stored - self.xnew) @ self.HessUC.T) * self.HessD **2) @ self.HessUC
            hagrad = self.weights @ Hdcode  # /self.mu

        print("Gradient Norm %.2f" % (np.linalg.norm(hagrad)))
        # if self.rankweight is False:
        #     if self.maximize:
        #         ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        #     else:
        #         ynew = radial_proj(self.xnew - self.lr * hagrad, max_norm=self.max_norm)
        # else: # if using rankweight, then the maximization if performed in the recombination step. 
        #     ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        mov_sign = -1 if (not self.maximize) and (not self.rankweight) else 1 
        ynew = radial_proj(self.xnew + mov_sign * self.lr * hagrad, max_norm=self.max_norm)
        return ynew

    def generate_sample(self, samp_num=None, hess_comp=False):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        N = self.dimen
        # Generate new sample by sampling from Gaussian distribution
        if hess_comp:
            # self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((H_pos_samples, H_neg_samples), axis=0)
            # new_samples = radial_proj(new_samples, self.max_norm)
        else:
            new_samples = zeros((samp_num, N))
            self.innerU = randn(samp_num, N)  # Isotropic gaussian distributions
            self.outerV = self.innerU / sqrt(self.Lambda) + (
                        (self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC  # H^{-1/2}U
            # new_samples[0:1, :] = self.xnew
            new_samples[:, :] = self.xnew + self.mu * self.outerV  # m + sig * Normal(0,C) self.mu *
            new_samples = radial_proj(new_samples, self.max_norm)
            self.code_stored = np.concatenate((self.code_stored, new_samples), axis=0) if self.code_stored.size else new_samples
            self.N_in_samp += samp_num
        return new_samples
        # set short name for everything to simplify equations

#%%
class HessAware_ADAM_DC:
    """Gaussian Sampling method for estimating Hessian"""
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, nu=0.9, maximize=True, max_norm=300):
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
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.xscore = 0
        self._istep = 0  # step counter
        self.maximize = maximize  # maximize / minimize the function
        self.Hupdate_freq = np.nan  # disable Hessian update here!
        # self.hess_comp = False
        self.code_stored = np.array([]).reshape((0, self.dimen))
        self.score_stored = np.array([])
        self.N_in_samp = 0
        self.max_norm = max_norm

    def new_generation(self, init_score, init_code):
        self.xscore = init_score
        self.score_stored = np.array([])
        self.xnew = init_code
        self.code_stored = np.array([]).reshape((0, self.dimen))
        self.N_in_samp = 0
        self._istep += 1 

    def compute_hess(self, scores):
        ''' ADAM doesn't need to compute Hessian separately '''
        print("Hess Diagonal Estimate: Mean %.2f, Max %.2f, Min %.2f" % (
            self.Hdiag.mean(), self.Hdiag.max(), self.Hdiag.min()))

    def compute_grad(self, scores, nat_grad=True):
        # add the new scores to storage
        self.score_stored = np.concatenate((self.score_stored, scores), axis=0) if self.score_stored.size else scores
        if nat_grad:
            hagrad = (self.score_stored - self.xscore) /self.mu @ (self.code_stored - self.xnew) / self.N_in_samp # /self.mu
        else:
            hagrad = (self.score_stored - self.xscore) / self.mu @ (self.code_stored - self.xnew) * self.Hdiag / self.N_in_samp  # non nat_grad
        self.D = self.nu * self.D + (1 - self.nu) * hagrad**2  # running average of gradient square
        self.Hdiag = self.D / (1 - self.nu ** (self._istep+1))  # Diagonal of estimated Hessian
        print("Gradient Norm %.2f" % (np.linalg.norm(hagrad)))
        print("Hess Diagonal Estimate: Mean %.2f, Max %.2f, Min %.2f" % (self.Hdiag.mean(), self.Hdiag.max(), self.Hdiag.min()))
        if self.maximize:
            ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        else:
            ynew = radial_proj(self.xnew - self.lr * hagrad, max_norm=self.max_norm)
        return ynew

    def generate_sample(self, samp_num=None, hess_comp=False):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        N = self.dimen
        # Generate new sample by sampling from Gaussian distribution
        if hess_comp:
            new_samples = np.array([]).reshape((0, self.dimen))
            pass
        else:
            new_samples = zeros((samp_num, N))
            self.innerU = randn(samp_num, N)  # Isotropic gaussian distributions
            self.outerV = self.innerU / sqrt(self.Hdiag)  # H^{-1/2}U
            new_samples[:, :] = self.xnew + self.mu * self.outerV  # m + sig * Normal(0,C) self.mu *
            new_samples = radial_proj(new_samples, self.max_norm)
            self.code_stored = np.concatenate((self.code_stored, new_samples), axis=0) if self.code_stored.size else new_samples
            self.N_in_samp += samp_num
        return new_samples
        # set short name for everything to simplify equations

class HessAware_Gauss_Spherical_DC:
    """Gaussian Sampling method for estimating Hessian"""
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, Lambda=0.9, Hupdate_freq=5, 
            maximize=True, max_norm=300, rankweight=False, nat_grad=False):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        self.Lambda = Lambda  # diagonal regularizer for Hessian matrix
        self.lr = lr  # learning rate (step size) of moving along gradient
        self.grad = np.zeros((1, self.dimen))  # estimated gradient
        self.innerU = np.zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = np.zeros((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.xscore = 0
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
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.score_stored = np.array([])
        self.N_in_samp = 0
        self.max_norm = max_norm
        # Options for "Gradient computation"
        self.rankweight = rankweight # Switch between using raw score as weight VS use rank weight as score
        self.nat_grad = nat_grad # use the natural gradient definition, or normal gradient. 

    def new_generation(self, init_score, init_code):
        self.xscore = init_score
        self.score_stored = np.array([])
        self.xnew = init_code
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.N_in_samp = 0
        self._istep += 1 

    def compute_hess(self, scores, Lambda_Frac=100):
        ''' Not implemented in spherical setting '''
        fbasis = self.xscore
        fpos = scores[:self.HB]
        fneg = scores[-self.HB:]
        weights = abs((fpos + fneg - 2 * fbasis) / 2 / self.mu ** 2 / self.HB)  # use abs to enforce positive definiteness
        C = sqrt(weights[:, np.newaxis]) * self.HinnerU  # or the sqrt may not work.
        # H = C^TC + Lambda * I
        self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        self.Lambda = (self.HessD ** 2).sum() / Lambda_Frac
        self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        print("Hessian Samples Spectrum", self.HessD)
        print("Hessian Samples Full Power:%f \nLambda:%f" % ((self.HessD ** 2).sum(), self.Lambda) )

    def compute_grad(self, scores):
        # add the new scores to storage
        # refer to the original one, adapt from the HessAware_Gauss version. 
        self.score_stored = np.concatenate((self.score_stored, scores), axis=0) if self.score_stored.size else scores
        if self.rankweight is False: # use the score difference as weight
            # B normalizer should go here larger cohort of codes gives more estimates 
            self.weights = (self.score_stored - self.xscore) / self.score_stored.size # / self.mu 
            # assert(self.N_in_samp == self.score_stored.size)
        else:  # use a function of rank as weight, not really gradient. 
            # Note descent check **could be** built into ranking weight? 
            # If not better just don't give weights to that sample 
            if self.maximize is False: # note for weighted recombination, the maximization flag is here. 
                code_rank = np.argsort(np.argsort( self.score_stored))  # add - operator it will do maximization.
            else:
                code_rank = np.argsort(np.argsort(-self.score_stored))
            # Consider do we need to consider the basis code and score here? Or no? 
            # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
            self.weights = rankweight(self.score_stored.size, mu=self.B / 2)[code_rank] # map the rank to the corresponding weight of recombination
            self.weights = self.weights[np.newaxis, :]
            # only keep the top 20 codes and recombine them.

        if self.nat_grad: # if or not using the Hessian to rescale the codes 
            # hagrad = self.weights @ (self.code_stored - self.xnew) # /self.mu
            hagrad = self.weights @ self.tang_code_stored # /self.mu
        else:
            Hdcode = self.Lambda * self.tang_code_stored + (
                    (self.tang_code_stored @ self.HessUC.T) * self.HessD **2) @ self.HessUC
            hagrad = self.weights @ Hdcode  # /self.mu

        print("Gradient Norm %.2f" % (np.linalg.norm(hagrad)))
        # if self.rankweight is False:
        #     if self.maximize:
        #         ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        #     else:
        #         ynew = radial_proj(self.xnew - self.lr * hagrad, max_norm=self.max_norm)
        # else: # if using rankweight, then the maximization if performed in the recombination step. 
        #     ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        if self.rankweight is False:
            if self.maximize:
                ynew = ExpMap(self.xnew, self.lr * hagrad)
            else:
                ynew = ExpMap(self.xnew, - self.lr * hagrad)
        else: # if using rankweight, then the maximization if performed in the recombination step. 
            ynew = ExpMap(self.xnew, self.lr * hagrad)

        ynew = ynew / norm(ynew) * self.max_norm
        return ynew

    def generate_sample(self, samp_num=None, hess_comp=False):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        N = self.dimen
        # Generate new sample by sampling from Gaussian distribution
        if hess_comp: 
            # Not implemented yet 
            # self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((H_pos_samples, H_neg_samples), axis=0)
            # new_samples = radial_proj(new_samples, self.max_norm)
        else:
            # new_samples = zeros((samp_num, N))
            self.innerU = randn(samp_num, N)  # Isotropic gaussian distributions
            self.outerV = self.innerU / sqrt(self.Lambda) + (
                        (self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC  # H^{-1/2}U
            tang_codes = self.mu * self.outerV # TODO: orthogonalize 
            new_samples = ExpMap(self.xnew, tang_codes) # m + sig * Normal(0,C) self.mu *
            new_samples = radial_proj(new_samples, self.max_norm)
            self.tang_code_stored = np.concatenate((self.tang_code_stored, tang_codes), axis=0) if self.tang_code_stored.size else tang_codes  # only store the tangent codes.
            self.N_in_samp += samp_num
        return new_samples
        # set short name for everything to simplify equations

class HessAware_Gauss_Hybrid_DC:
    """Gaussian Sampling method Hessian Aware Algorithm
    With a automatic switch between linear exploration and spherical exploration
    Our ultimate Optimizer. It's an automatic blend of Spherical class and normal class.
    TODO: Add automatic tuning of mu and step size according to the norm of the mean vector.
    """
    def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, lr_sph=2, mu_sph=0.005, 
            Lambda=0.9, Hupdate_freq=5, maximize=True, max_norm=300, rankweight=False, nat_grad=False):
        self.dimen = space_dimen  # dimension of input space
        self.B = population_size  # population batch size
        self.mu = mu  # scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        self.Lambda = Lambda  # diagonal regularizer for Hessian matrix
        self.lr = lr  # learning rate (step size) of moving along gradient
        self.grad = np.zeros((1, self.dimen))  # estimated gradient
        self.innerU = np.zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = np.zeros((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        self.xnew = np.zeros((1, self.dimen))  # new base point
        self.xscore = 0
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
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.score_stored = np.array([])
        self.N_in_samp = 0
        self.max_norm = max_norm
        # Options for `compute_grad` part
        self.rankweight = rankweight # Switch between using raw score as weight VS use rank weight as score
        self.nat_grad = nat_grad # use the natural gradient definition, or normal gradient. 
        self.sphere_flag = False # initialize the whole system as linear?
        self.lr_sph = lr_sph
        self.mu_sph = mu_sph

    def new_generation(self, init_score, init_code):
        self.xscore = init_score
        self.score_stored = np.array([])
        self.xnew = init_code
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.N_in_samp = 0
        self._istep += 1 

    def compute_hess(self, scores, Lambda_Frac=100):
        ''' Not implemented in spherical setting '''
        fbasis = self.xscore
        fpos = scores[:self.HB]
        fneg = scores[-self.HB:]
        weights = abs((fpos + fneg - 2 * fbasis) / 2 / self.mu ** 2 / self.HB)  # use abs to enforce positive definiteness
        C = sqrt(weights[:, np.newaxis]) * self.HinnerU  # or the sqrt may not work.
        # H = C^TC + Lambda * I
        self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        self.Lambda = (self.HessD ** 2).sum() / Lambda_Frac
        self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        print("Hessian Samples Spectrum", self.HessD)
        print("Hessian Samples Full Power:%f \nLambda:%f" % ((self.HessD ** 2).sum(), self.Lambda) )

    def compute_grad(self, scores):
        # add the new scores to storage
        # refer to the original one, adapt from the HessAware_Gauss version. 
        self.score_stored = np.concatenate((self.score_stored, scores), axis=0) if self.score_stored.size else scores
        if self.rankweight is False: # use the score difference as weight
            # B normalizer should go here larger cohort of codes gives more estimates 
            self.weights = (self.score_stored - self.xscore) / self.score_stored.size # / self.mu 
            # assert(self.N_in_samp == self.score_stored.size)
        else:  # use a function of rank as weight, not really gradient. 
            # Note descent check **could be** built into ranking weight? 
            # If not better just don't give weights to that sample 
            if self.maximize is False: # note for weighted recombination, the maximization flag is here. 
                code_rank = np.argsort(np.argsort( self.score_stored))  # add - operator it will do maximization.
            else:
                code_rank = np.argsort(np.argsort(-self.score_stored))
            # Consider do we need to consider the basis code and score here? Or no? 
            # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
            self.weights = rankweight(self.score_stored.size, mu=self.B / 2)[code_rank] # map the rank to the corresponding weight of recombination
            self.weights = self.weights[np.newaxis, :]
            # only keep the top 20 codes and recombine them.

        if self.nat_grad: # if or not using the Hessian to rescale the codes 
            # hagrad = self.weights @ (self.code_stored - self.xnew) # /self.mu
            hagrad = self.weights @ self.tang_code_stored # /self.mu
        else:
            Hdcode = self.Lambda * self.tang_code_stored + (
                    (self.tang_code_stored @ self.HessUC.T) * self.HessD **2) @ self.HessUC
            hagrad = self.weights @ Hdcode  # /self.mu

        print("Gradient Norm %.2f" % (np.linalg.norm(hagrad)))
        # if self.rankweight is False:
        #     if self.maximize:
        #         ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        #     else:
        #         ynew = radial_proj(self.xnew - self.lr * hagrad, max_norm=self.max_norm)
        # else: # if using rankweight, then the maximization if performed in the recombination step. 
        #     ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        # Summarize the conditional above to the following line.
        # If it's minimizing and it's not using rankweight, then travel inverse to the hagrad. 
        mov_sign = -1 if (not self.maximize) and (not self.rankweight) else 1 
        if not self.sphere_flag: 
            ynew = self.xnew + mov_sign * self.lr * hagrad # Linear space ExpMap reduced to just normal linear addition.
            if norm(ynew) > self.max_norm:
                print("Travelling outside the spherer\n Changing to Spherical mode at step %d",self._istep)
                self.sphere_flag = True 
                self.mu = self.mu_sph # changing the new parameters
                self.lr = self.lr_sph # changing the new parameters
                # maybe save the old ones
        else:  # Spherically move mean
            ynew = ExpMap(self.xnew, mov_sign * self.lr * hagrad)
            ynew = ynew / norm(ynew) * self.max_norm # exact normalization to the sphere.
        # ynew = radial_proj(ynew, max_norm=self.max_norm) # Projection 
        return ynew

    def generate_sample(self, samp_num=None, hess_comp=False):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        N = self.dimen
        # Generate new sample by sampling from Gaussian distribution
        if hess_comp: 
            # Not implemented yet 
            # self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((H_pos_samples, H_neg_samples), axis=0)
            # new_samples = radial_proj(new_samples, self.max_norm)
        else:
            # new_samples = zeros((samp_num, N))
            self.innerU = randn(samp_num, N)  # Isotropic gaussian distributions
            self.outerV = self.innerU / sqrt(self.Lambda) + (
                        (self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC  # H^{-1/2}U
            tang_codes = self.mu * self.outerV # TODO: orthogonalize 
            if not self.sphere_flag: 
                new_samples = self.xnew + tang_codes # Linear space ExpMap reduced to just normal linear addition.
            else:  # Spherically move mean
                new_samples = ExpMap(self.xnew, tang_codes) # m + sig * Normal(0,C) self.mu *
            new_samples = radial_proj(new_samples, self.max_norm)
            self.tang_code_stored = np.concatenate((self.tang_code_stored, tang_codes), axis=0) if self.tang_code_stored.size else tang_codes  # only store the tangent codes.
            self.N_in_samp += samp_num
        return new_samples
        # set short name for everything to simplify equations


class ExperimentEvolve_DC:
    """
    Default behavior is to use the current CMAES optimizer to optimize for 200 steps for the given unit.
    This Experimental Class is defined to test out the new Descent checking 
    """
    def __init__(self, model_unit, max_step=200, optimizer=None, nat_grad=True):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        if optimizer is None:  # Default optimizer is this
            self.optimizer = HessAware_Gauss_DC(space_dimen=4096, )    # , optim_params=optim_params
            # CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
            #                                            init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq)
        else:
            # assert issubclass(type(optimizer), Optimizer)
            self.optimizer = optimizer
        self.max_steps = max_step
        self.istep = 0

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        x_img = render(init_code)
        x_score = self.CNNmodel.score(x_img) # initial code and image and score
        self.optimizer.new_generation(x_score, init_code)
        MAX_IN_ITER = 100
        Batch_Size = 40
        INCRE_NUM = 10
        samp_num = Batch_Size
        self.codes_all = init_code
        self.scores_all = np.array([x_score])
        self.generations = np.array([self.istep])
        while True:
            new_codes = self.optimizer.generate_sample(samp_num) # self.optimizer.N_in_samp += samp_num
            new_imgs = render(new_codes)
            new_scores = self.CNNmodel.score(new_imgs)
            y_code = self.optimizer.compute_grad(new_scores)
            y_img = render(y_code)
            y_score = self.CNNmodel.score(y_img)
            self.codes_all = np.concatenate((self.codes_all, new_codes, y_code), axis=0)
            self.scores_all = np.concatenate((self.scores_all, new_scores[:, np.newaxis], y_score[:, np.newaxis]), axis=0)
            self.generations = np.concatenate((self.generations, np.array([self.istep] * (samp_num + 1))), axis=0)
            print('Step {}\nsynthetic img scores: mean {}, all {}'.format(self.istep, np.nanmean(new_scores), new_scores))
            if y_score < x_score and self.optimizer.N_in_samp <= MAX_IN_ITER:
                samp_num = INCRE_NUM
            else:
                print("Accepted basis score: mean %.2f" % y_score)
                print("Accepted basis code: norm %.2f" % np.linalg.norm(y_code))
                x_code = y_code
                x_score = y_score
                self.istep += 1
                self.optimizer.new_generation(x_score, x_code)  # clear score_store code_store N_in_samp
                samp_num = Batch_Size
                if self.istep > self.max_steps:
                    break
                if not self.istep % self.optimizer.Hupdate_freq:
                    Hess_codes = self.optimizer.generate_sample(samp_num, hess_comp=True)
                    Hess_imgs = render(Hess_codes)
                    Hess_scores = self.CNNmodel.score(Hess_imgs)
                    self.optimizer.compute_hess(Hess_scores)
                    self.codes_all = np.concatenate((self.codes_all, Hess_codes), axis=0)
                    self.scores_all = np.concatenate((self.scores_all, Hess_scores[:, np.newaxis]), axis=0)
                    self.generations = np.concatenate((self.generations, np.array([self.istep] * len(Hess_scores))),
                                                      axis=0)
        self.scores_all = self.scores_all[:, 0]
        print("Summary\nGenerations: %d, Image samples: %d, Best score: %.2f" % (self.istep, self.codes_all.shape[0], self.scores_all.max()))

    def visualize_exp(self, show=False, title_str=""):
        """ Visualize the experiment by showing the maximal activating images and the scores in each generations
        """
        idx_list = []
        for geni in range(min(self.generations), max(self.generations) + 1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = render(select_code)
        fig = utils.visualize_img_list(img_select, score_select, show=show, nrow=None, title_str=title_str)
        if show:
            fig.show()
        return fig

    def visualize_best(self, show=False, title_str=""):
        """ Just Visualize the best Images for the experiment """
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx + 1, :]
        score_select = self.scores_all[idx]
        img_select = render(select_code)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select)+title_str, fontsize=16)
        if show:
            plt.show()
        return fig

    def visualize_codenorm(self, show=True, title_str=""):
        code_norm = np.sqrt((self.codes_all ** 2).sum(axis=1))
        figh = plt.figure()
        plt.scatter(self.generations, code_norm, s=16, alpha=0.6, label="all score")
        plt.title("Optimization Trajectory of Code Norm\n" + title_str)
        if show:
            plt.show()
        return figh

    def visualize_trajectory(self, show=True, title_str=""):
        """ Visualize the Score Trajectory """
        gen_slice = np.arange(min(self.generations), max(self.generations) + 1)
        AvgScore = np.zeros_like(gen_slice)
        MaxScore = np.zeros_like(gen_slice)
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n" + title_str)
        plt.legend()
        if show:
            plt.show()
        return figh
#%%

# savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# unit = ('caffe-net', 'fc8', 1)
# expdir = join(savedir, "%s_%s_%d_nohess" % unit)
# os.makedirs(expdir, exist_ok=True)
# for trial_i in range(2):
#     optim = HessAware_Gauss_DC(4096, population_size=40, lr=0.2, mu=0.6, Lambda=0.2, Hupdate_freq=101, maximize=True,max_norm=1000)
#     fn_str = "lr%.1f_mu%.1f_Labda%s_uf%d_tr%d" % (optim.lr, optim.mu, optim.Lambda, optim.Hupdate_freq, trial_i)
#     f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#     sys.stdout = f
#     experiment = ExperimentEvolve_DC(unit, max_step=100, optimizer=optim)
#     experiment.run(init_code=np.zeros((1, 4096)))
#     param_str = "lr=%.1f, mu=%.1f, Lambda=%.1f, Hupdate_freq=%d" % (optim.lr, optim.mu, optim.Lambda, optim.Hupdate_freq)
#     fig1 = experiment.visualize_trajectory(show=True, title_str=param_str)
#     fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
#     fig2 = experiment.visualize_best(show=True, title_str=param_str)
#     fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
#     fig3 = experiment.visualize_exp(show=True, title_str=param_str)
#     fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
#     fig4 = experiment.visualize_codenorm(show=True, title_str=param_str)
#     fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
#     # plt.close(fig1)
#     # plt.close(fig2)
#     # plt.close(fig3)
#     # plt.close(fig4)
#     plt.show(block=False)
#     time.sleep(5)
#     plt.close('all')
#     sys.stdout = orig_stdout
#     f.close()
#
# #%%
# savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# unit = ('caffe-net', 'fc8', 1)
# expdir = join(savedir, "%s_%s_%d_lmbdadp" % unit)
# os.makedirs(expdir, exist_ok=True)
# lr_list = [0.2, 0.5, 1, 2, 5, 10, 20]
# mu_list = [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 20]
# UF_list = [5, 10, 20, 2]
# Lambda_list = []
# for i, UF in enumerate(UF_list):
#     for j, lr in enumerate(lr_list):
#         for k, mu in enumerate(mu_list):
#             idxno = k + j * len(mu_list) + i * len(lr_list) * len(mu_list)
#             if idxno < 77:
#                 continue
#             for trial_i in range(2):
#                 fn_str = "lr%.1f_mu%.1f_Labda%s_uf%d_tr%d" % (lr, mu, "Adp", UF, trial_i)
#                 f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#                 sys.stdout = f
#                 optim = HessAware_Gauss_DC(4096, population_size=40, lr=lr, mu=mu, Lambda=0.1, Hupdate_freq=UF, maximize=True)
#                 experiment = ExperimentEvolve_DC(unit, max_step=50, optimizer=optim)
#                 experiment.run(init_code=np.zeros((1, 4096)))
#                 param_str = "lr=%.1f, mu=%.1f, Lambda=%.1f. Hupdate_freq=%d" % (optim.lr, optim.mu, optim.Lambda, optim.Hupdate_freq)
#                 fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
#                 fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
#                 fig2 = experiment.visualize_best(show=False, title_str=param_str)
#                 fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
#                 fig3 = experiment.visualize_exp(show=False, title_str=param_str)
#                 fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
#                 fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
#                 fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
#                 # plt.close(fig1)
#                 # plt.close(fig2)
#                 # plt.close(fig3)
#                 # plt.close(fig4)
#                 plt.show(block=False)
#                 time.sleep(5)
#                 plt.close('all')
#                 sys.stdout = orig_stdout
#                 f.close()
#%%
# savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# unit = ('caffe-net', 'fc8', 1)
# expdir = join(savedir, "%s_%s_%d_ADAM_DC_real" % unit)
# os.makedirs(expdir, exist_ok=True)
# lr_list = [1, 2, 0.5, 5, 0.2, 10, 20]
# mu_list = [1, 2, 0.5, 5, 0.2, 10, 0.1, ]
# nu_list = [0.9, 0.99, 0.8, 0.999]
# for i, nu in enumerate(nu_list):
#     for j, lr in enumerate(lr_list):
#         for k, mu in enumerate(mu_list):
#             idxno = k + j * len(mu_list) + i * len(lr_list) * len(mu_list)
#             for trial_i in range(1):
#                 fn_str = "lr%.1f_mu%.1f_nu%.2f_tr%d" % (lr, mu, nu, trial_i)
#                 f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#                 sys.stdout = f
#                 optim = HessAware_ADAM_DC(4096, population_size=40, lr=lr, mu=mu, nu=nu, maximize=True, max_norm=400)
#                 experiment = ExperimentEvolve_DC(unit, max_step=100, optimizer=optim)
#                 experiment.run(init_code=np.zeros((1, 4096)))
#                 param_str = "lr=%.1f, mu=%.1f, nu=%.1f." % (optim.lr, optim.mu, optim.nu)
#                 fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
#                 fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
#                 fig2 = experiment.visualize_best(show=False, title_str=param_str)
#                 fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
#                 fig3 = experiment.visualize_exp(show=False, title_str=param_str)
#                 fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
#                 fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
#                 fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
#                 plt.show(block=False)
#                 time.sleep(5)
#                 plt.close('all')
#                 sys.stdout = orig_stdout
#                 f.close()
#%%
# # #%%
# # unit = ('caffe-net', 'fc6', 1)
# # optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
# # experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
# # experiment.run()
# # #%%
# # #optim = HessAware_ADAM(4096, population_size=40, lr=2, mu=0.5, nu=0.99, maximize=True)
# # experiment2 = ExperimentEvolve(unit, max_step=71)
# # experiment2.run()
# # experiment2.visualize_trajectory(show=True)
# # experiment2.visualize_best(show=True)
# # #%%
# unit = ('caffe-net', 'fc8', 1)
# optim = HessAware_Gauss(4096, population_size=40, lr=2, mu=0.5, Lambda=0.9, Hupdate_freq=5, maximize=True)
# experiment3 = ExperimentEvolve(unit, max_step=50, optimizer=optim)
# experiment3.run()
# experiment3.visualize_trajectory(show=True)
# experiment3.visualize_best(show=True)
# %% HessAware_Gauss_Spherical Testing
# unit = ('caffe-net', 'fc8', 1)
# savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# expdir = join(savedir, "%s_%s_%d_Gauss_Sph" % unit)
# os.makedirs(expdir, exist_ok=True)
# # lr=0.25; mu=0.01; Lambda=0.99; trial_i=0
# UF = 200
# lr_list = [1,2] # lr_list = [0.1, 0.05, 0.5, 0.25, 0.01]
# mu_list = [0.01, 0.005] # mu_list = [0.01, 0.02, 0.005, 0.04, 0.002, 0.001]
# Lambda_list = [1]
# for i, Lambda in enumerate(Lambda_list):
#     for j, lr in enumerate(lr_list):
#         for k, mu in enumerate(mu_list):
#             idxno = k + j * len(mu_list) + i * len(lr_list) * len(mu_list)
#             for trial_i in range(3):
#                 fn_str = "lr%.2f_mu%.3f_Lambda%.1f_UF%d_tr%d" % (lr, mu, Lambda, UF, trial_i)
#                 f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#                 sys.stdout = f
#                 optim = HessAware_Gauss_Spherical(4096, population_size=40, lr=lr, mu=mu, Lambda=Lambda, Hupdate_freq=UF, sphere_norm=300, maximize=True)
#                 experiment = ExperimentEvolve(unit, max_step=100, optimizer=optim)
#                 experiment.run(init_code=np.random.randn(1, 4096))
#                 experiment.visualize_trajectory(show=True)
#                 experiment.visualize_best(show=True)
#                 param_str = "lr=%.2f, mu=%.3f, Lambda=%.1f, UpdateFreq=%d" % (optim.lr, optim.mu, optim.Lambda, optim.Hupdate_freq)
#                 fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
#                 fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
#                 fig2 = experiment.visualize_best(show=False)# , title_str=param_str)
#                 fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
#                 fig3 = experiment.visualize_exp(show=False, title_str=param_str)
#                 fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
#                 fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
#                 fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
#                 time.sleep(5)
#                 plt.close('all')
#                 sys.stdout = orig_stdout
#                 f.close()

# %% HessAware_Gauss_Spherical_DC Testing
# unit = ('caffe-net', 'fc8', 1)
# # savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# expdir = join(savedir, "%s_%s_%d_Gauss_DC_Sph" % unit)
# os.makedirs(expdir, exist_ok=True)
# lr = 3; mu = 0.002; Lambda=1;trial_i=0
# fn_str = "lr%.1f_mu%.1f_Lambda%.2f_tr%d" % (lr, mu, Lambda, trial_i)
# #f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
# #sys.stdout = f
# optim = HessAware_Gauss_Spherical_DC(4096, population_size=40, lr=lr, mu=mu, Lambda=Lambda, Hupdate_freq=201,
#             rankweight=True, nat_grad=True, maximize=True, max_norm=300)
# experiment = ExperimentEvolve_DC(unit, max_step=100, optimizer=optim)
# experiment.run(init_code=np.random.randn(1, 4096))
# param_str = "lr=%.1f, mu=%.4f, Lambda=%.2f." % (optim.lr, optim.mu, optim.Lambda)
# fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
# fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
# fig2 = experiment.visualize_best(show=False, title_str=param_str)
# fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
# fig3 = experiment.visualize_exp(show=False, title_str=param_str)
# fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
# fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
# fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
# plt.show(block=False)
# time.sleep(5)
# plt.close('all')unit = ('caffe-net', 'fc8', 1)
# # savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# expdir = join(savedir, "%s_%s_%d_Gauss_DC_Sph" % unit)
# os.makedirs(expdir, exist_ok=True)
# lr = 3; mu = 0.002; Lambda=1;trial_i=0
# fn_str = "lr%.1f_mu%.1f_Lambda%.2f_tr%d" % (lr, mu, Lambda, trial_i)
# #f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
# #sys.stdout = f
# optim = HessAware_Gauss_Spherical_DC(4096, population_size=40, lr=lr, mu=mu, Lambda=Lambda, Hupdate_freq=201,
#             rankweight=True, nat_grad=True, maximize=True, max_norm=300)
# experiment = ExperimentEvolve_DC(unit, max_step=100, optimizer=optim)
# experiment.run(init_code=np.random.randn(1, 4096))
# param_str = "lr=%.1f, mu=%.4f, Lambda=%.2f." % (optim.lr, optim.mu, optim.Lambda)
# fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
# fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
# fig2 = experiment.visualize_best(show=False, title_str=param_str)
# fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
# fig3 = experiment.visualize_exp(show=False, title_str=param_str)
# fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
# fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
# fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
# plt.show(block=False)
# time.sleep(5)
# plt.close('all')
#%%
unit = ('caffe-net', 'fc8', 1)
savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
# savedir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
expdir = join(savedir, "%s_%s_%d_Gauss_DC_Hybrid" % unit)
os.makedirs(expdir, exist_ok=True)
# lr = 3; mu = 0.002;
lr = 0.1; mu = 1
Lambda=1; trial_i=0
fn_str = "lr%.1f_mu%.1f_Lambda%.2f_tr%d" % (lr, mu, Lambda, trial_i)
#f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#sys.stdout = f
optim = HessAware_Gauss_Hybrid_DC(4096, population_size=40, lr=lr, mu=mu, lr_sph=2, mu_sph=0.005,Lambda=Lambda, Hupdate_freq=201,
            rankweight=True, nat_grad=True, maximize=True, max_norm=300)
experiment = ExperimentEvolve_DC(unit, max_step=100, optimizer=optim)
experiment.run(init_code=np.random.randn(1, 4096))
param_str = "lr=%.1f, mu=%.4f, Lambda=%.2f." % (optim.lr, optim.mu, optim.Lambda)
fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
fig2 = experiment.visualize_best(show=False, title_str=param_str)
fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
fig3 = experiment.visualize_exp(show=False, title_str=param_str)
fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
plt.show(block=False)
time.sleep(5)
plt.close('all')
#%% ADAM DC
#savedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning"
#unit = ('caffe-net', 'fc8', 1)
#expdir = join(savedir, "%s_%s_%d_ADAM_DC" % unit)
#os.makedirs(expdir, exist_ok=True)
#lr=2; mu=1; nu=0.9; trial_i=0
#fn_str = "lr%.1f_mu%.1f_nu%.2f_tr%d" % (lr, mu, nu, trial_i)
#f = open(join(expdir, 'output_%s.txt' % fn_str), 'w')
#sys.stdout = f
#optim = HessAware_ADAM_DC(4096, population_size=40, lr=lr, mu=mu, nu=nu, maximize=True, max_norm=300)
#experiment = ExperimentEvolve_DC(unit, max_step=70, optimizer=optim)
#experiment.run(init_code=np.zeros((1, 4096)))
#param_str = "lr=%.1f, mu=%.1f, nu=%.1f." % (optim.lr, optim.mu, optim.nu)
#fig1 = experiment.visualize_trajectory(show=False, title_str=param_str)
#fig1.savefig(join(expdir, "score_traj_%s.png" % fn_str))
#fig2 = experiment.visualize_best(show=False, title_str=param_str)
#fig2.savefig(join(expdir, "Best_Img_%s.png" % fn_str))
#fig3 = experiment.visualize_exp(show=False, title_str=param_str)
#fig3.savefig(join(expdir, "Evol_Exp_%s.png" % fn_str))
#fig4 = experiment.visualize_codenorm(show=False, title_str=param_str)
#fig4.savefig(join(expdir, "norm_traj_%s.png" % fn_str))
