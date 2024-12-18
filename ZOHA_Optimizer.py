"""
This code is translated from matlab code in `CMAES_optimizer_matlab`.
Code and parameters are battle tested from monkey experiments and insilico experiments! 
"""
import numpy as np
from numpy.linalg import norm, svd
from scipy.linalg import orth
from numpy.random import randn
from numpy import sqrt, zeros, abs, floor, log, log2, eye, exp, linspace, logspace, log10, mean, std, sin
from ZO_HessAware_Optimizers import renormalize, rankweight, ExpMap
# following functions are borrowed from matlab.
def ang_dist(V1, V2):
    nV1 = V1 / norm(V1, axis=1, keepdims=True)
    nV2 = V2 / norm(V2, axis=1, keepdims=True)
    cosang = nV1 @ nV2.T
    ang = np.real(np.arccos(cosang))
    return ang

def SLERP(V1, V2, samp_t): # So called SLERP
    nV1 = V1 / norm(V1, axis=1, keepdims=True)
    nV2 = V2 / norm(V2, axis=1, keepdims=True)
    cosang = nV1 @ nV2.T
    ang = np.real(np.arccos(cosang))
    if isinstance(samp_t, np.ndarray):
        lingrid = samp_t.reshape(-1, 1)
        Vecs = (sin((1-lingrid) * ang) @ V1 + sin(lingrid * ang) @ V2) / sin(ang)
    elif np.isscalar(samp_t):
        Vecs = (sin((1 - samp_t) * ang) * V1 + sin(samp_t * ang) * V2) / sin(ang)
    else:
        raise
    return Vecs
#%%
class ZOHA_Sphere_lr_euclid:
    def __init__(self, space_dimen, population_size=40, select_size=20, lr=1.5, \
                 maximize=True, rankweight=True, rankbasis=False, sphere_norm=300):
        self.dimen = space_dimen   # dimension of input space
        self.B = population_size   # population batch size
        self.select_cutoff = select_size
        self.sphere_norm = sphere_norm
        self.lr = lr  # learning rate (step size) of moving along gradient

        self.tang_codes = zeros((self.B, self.dimen))
        self.grad = zeros((1, self.dimen))  # estimated gradient
        self.innerU = zeros((self.B, self.dimen))  # inner random vectors with covariance matrix Id
        self.outerV = zeros ((self.B, self.dimen))  # outer random vectors with covariance matrix H^{-1}, equals innerU @ H^{-1/2}
        self.xcur = zeros((1, self.dimen)) # current base point
        self.xnew = zeros((1, self.dimen)) # new base point

        self.istep = -1  # step counter
        # self.counteval = 0
        self.maximize = maximize # maximize / minimize the function
        self.rankweight = rankweight# Switch between using raw score as weight VS use rank weight as score
        self.rankbasis = rankbasis # Ranking basis or rank weights only
        # opts # object to store options for the future need to examine or tune

    def lr_schedule(self, n_gen=100, mode="inv", lim=(50, 7.33) ,):
        # note this normalize to the expected norm of a N dimensional Gaussian
        if mode == "inv":
            self.mulist = 15 + 1 / (0.0017 * np.arange(1, n_gen +1) + 0.0146);
            # self.opts.mu_init = self.mulist[0]
            # self.opts.mu_final = self.mulist[-1]
            self.mulist = self.mulist / 180 * np.pi / sqrt(self.dimen)
            self.mu_init = self.mulist[0]; self.mu_final = self.mulist[-1]
        else:
            self.mu_init = lim[0]
            self.mu_final = lim[1]
            if mode == "lin":
                self.mulist = linspace(self.mu_init, self.mu_final, n_gen) / 180 * np.pi / sqrt(self.dimen)
            elif mode == "exp":
                self.mulist = logspace(log10(self.mu_init), log10(self.mu_final), n_gen) / 180 * np.pi / sqrt(self.dimen)

    def step_simple(self, scores, codes):
        N = self.dimen;
        print('Gen %d max score %.3f, mean %.3f, std %.3f\n ' %(self.istep, max(scores), mean(scores), std(scores) ))
        if self.istep == -1:
        # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            print('First generation')
            self.xcur = codes[0:1, :]
            if not self.rankweight: # use the score difference as weight
                # B normalizer should go here larger cohort of codes gives more estimates
                weights = (scores - scores[0]) / self.B # / self.mu
            else:  # use a function of rank as weight, not really gradient.
                if not self.maximize: # note for weighted recombination, the maximization flag is here.
                    code_rank = scores.argsort().argsort() # find rank of ascending order
                else:
                    code_rank = (-scores).argsort().argsort() # find rank of descending order
                # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more.
                raw_weights = rankweight(len(code_rank))
                weights = raw_weights[code_rank] # map the rank to the corresponding weight of recombination
                # Consider the basis in our rank! but the weight will be wasted as we don't use it.

            w_mean = weights[np.newaxis,:] @ codes # mean in the euclidean space
            self.xnew = w_mean / norm(w_mean) * self.sphere_norm # project it back to shell.
        else:
            self.xcur = codes[0:1, :]
            if not self.rankweight: # use the score difference as weight
                # B normalizer should go here larger cohort of codes gives more estimates
                weights = (scores - scores[0]) / self.B; # / self.mu
            else:  # use a function of rank as weight, not really gradient.
                if not self.rankbasis: # if false, then exclude the first basis vector from rank (thus it receive no weights.)
                    rankedscore = scores[1:]
                else:
                    rankedscore = scores
                if not self.maximize: # note for weighted recombination, the maximization flag is here.
                    code_rank = rankedscore.argsort().argsort() # find rank of ascending order
                else:
                    code_rank = (-rankedscore).argsort().argsort() # find rank of descending order
                # Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more.
                raw_weights = rankweight(len(code_rank), mu=self.select_cutoff)
                weights = raw_weights[code_rank] # map the rank to the corresponding weight of recombination
                # Consider the basis in our rank! but the weight will be wasted as we don't use it.
                if not self.rankbasis:
                    weights = np.append(0, weights) # the weight of the basis vector will do nothing! as the deviation will be nothing
            # estimate gradient from the codes and scores
            # assume weights is a row vector
            w_mean = weights[np.newaxis,:] @ codes # mean in the euclidean space
            w_mean = w_mean / norm(w_mean) * self.sphere_norm # rescale, project it back to shell.
            self.xnew = SLERP(self.xcur, w_mean, self.lr) # use lr to spherical extrapolate
            print("Step size %.3f, multip learning rate %.3f, " % (ang_dist(self.xcur, self.xnew), ang_dist(self.xcur, self.xnew) * self.lr));
            ang_basis_to_samp = ang_dist(codes, self.xnew)
            print("New basis ang to last samples mean %.3f(%.3f), min %.3f" % (mean(ang_basis_to_samp), std(ang_basis_to_samp), min(ang_basis_to_samp)));

        # Generate new sample by sampling from Gaussian distribution
        self.tang_codes = zeros((self.B, N))  # Tangent vectors of exploration
        self.innerU = randn(self.B, N)  # Isotropic gaussian distributions
        self.outerV = self.innerU # H^{-1/2}U, more transform could be applied here!
        self.outerV = self.outerV - (self.outerV @ self.xnew.T) @ self.xnew / norm(self.xnew) ** 2 # orthogonal projection to xnew's tangent plane.
        mu = self.mulist[self.istep + 1] if self.istep < len(self.mulist) - 1 else self.mulist[-1]
        new_samples = zeros((self.B + 1, N))
        new_samples[0, :] = self.xnew
        self.tang_codes = mu * self.outerV # m + sig * Normal(0,C)
        new_samples[1:, :] = ExpMap(self.xnew, self.tang_codes)
        print("Current Exploration %.1f deg" % (mu * sqrt(self.dimen - 1) / np.pi * 180))
        # new_ids = [];
        # for k in range(new_samples.shape[0]):
        #     new_ids = [new_ids, sprintf("gen%03d_%06d", self.istep+1, self.counteval)];
        #     self.counteval = self.counteval + 1;
        self.istep = self.istep + 1
        new_samples = renormalize(new_samples, self.sphere_norm)
        return new_samples

class ZOHA_Sphere_lr_euclid_ReducDim(ZOHA_Sphere_lr_euclid):
    def __init__(self, code_len, space_dimen, population_size=40, select_size=20, lr=1.5, \
                 maximize=True, rankweight=True, rankbasis=False, sphere_norm=300):
        super().__init__(space_dimen=space_dimen, population_size=population_size, select_size=select_size, lr=lr, \
                 maximize=maximize, rankweight=rankweight, rankbasis=rankbasis, sphere_norm=sphere_norm)
        self.code_len = code_len
        self.get_basis("rand")
        # self.opts.subspac_d = self.dimen

    def get_basis(self, basis="rand"):
        if type(basis) is str and basis == "rand":
            raw_basis = np.random.randn(self.code_len, self.dimen) # code length by reduced dimension
            orthobasis, _ = np.linalg.qr(raw_basis)  # orthonormalization by QR decomposition
            self.basis = orthobasis.T   # reduced dimension by code length
        elif type(basis) is np.ndarray:
            if basis.shape == (self.code_len, self.dimen):
                self.basis = basis.T
            elif basis.shape == (self.dimen, self.code_len):
                self.basis = basis
            else:
                raise RuntimeError("basis shape incorrect")
        else:
            raise RuntimeError("basis option unrecognized")
        return self.basis

    def step_simple(self, scores, codes):
        codes_proj = codes @ self.basis.T  # isometric transform by the basis set
        new_codes_proj = super().step_simple(scores, codes_proj)
        return new_codes_proj @ self.basis # isometric inverse transform by the basis set

#%%
# optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
# optimizer.lr_schedule(n_gen=100)
# codes = np.random.randn(40, 4096)
# scores = np.random.randn(40)
# optimizer.step_simple(scores, codes)
#%%
if __name__ == "__main__":
    from insilico_Exp import ExperimentEvolve
    optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
    optimizer.lr_schedule(n_gen=100, mode="exp")
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
    #%%
    optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, 50, population_size=40, select_size=20)
    optimizer.lr_schedule(n_gen=50, mode="exp")
    optimizer.get_basis("rand")
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
