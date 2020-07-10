import numpy as np
from numpy.linalg import norm
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

#%%
# optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
# optimizer.lr_schedule(n_gen=100)
# codes = np.random.randn(40, 4096)
# scores = np.random.randn(40)
# optimizer.step_simple(scores, codes)
#%%
from insilico_Exp import ExperimentEvolve
# optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
# optimizer.lr_schedule(n_gen=100, mode="exp")
# Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=100, backend="torch", optimizer=optimizer, GAN="fc6")
# # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
# # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
# Exp.run()
#%%
mode_dict = ["inv","lin","exp"]
def optim_result(param):
    pop_size = int(param[0,0])
    select_rate = param[0,1]
    select_size = int(pop_size * select_rate)
    lr = param[0,2]
    mu_init = param[0,3]
    mu_rate = param[0,4]
    mode = int(param[0,5])
    n_gen = 4000 // pop_size # 4000 is the total evaluation budget
    optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=pop_size, select_size=select_size, lr=lr)
    optimizer.lr_schedule(n_gen=n_gen, lim=(mu_init, mu_rate*mu_init), mode=mode_dict[mode])
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
    return np.percentile(Exp.scores_all, 99.5)

#%%
def optim_result_short(param):
    pop_size = int(param[0,0])
    select_rate = param[0,1]
    select_size = int(pop_size * select_rate)
    lr = param[0,2]
    mu_init = param[0,3]
    mu_rate = param[0,4]
    mode = int(param[0,5])
    n_gen = 2000 // pop_size # 4000 is the total evaluation budget
    optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=pop_size, select_size=select_size, lr=lr)
    optimizer.lr_schedule(n_gen=n_gen, lim=(mu_init, mu_rate*mu_init), mode=mode_dict[mode])
    Exp = ExperimentEvolve(("alexnet", "fc8", 1), max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
    # note using torch and batch processing evolution is much faster ~ 27sec 100 gens
    # Exp = ExperimentEvolve(("caffe-net", "fc8", 1), max_step=50, backend="caffe", optimizer=optimizer, GAN="fc6")
    Exp.run()
    return np.percentile(Exp.scores_all, 99.5)
#%%
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
mixed_domain =[{'name': 'pop_size', 'type': 'continuous', 'domain': (20, 50),'dimensionality': 1},
                {'name': 'select_rate', 'type': 'continuous', 'domain': (0.1, 0.9),'dimensionality': 1},
                {'name': 'lr', 'type': 'continuous', 'domain': (0.5, 2),'dimensionality': 1},
               {'name': 'mu_init', 'type': 'continuous', 'domain': (40, 80),'dimensionality': 1},
               {'name': 'mu_rate', 'type': 'continuous', 'domain': (0.15, 0.9),'dimensionality': 1},
               {'name': 'mode', 'type': 'categorical', 'domain': (0,1,2),'dimensionality': 1},]
#                {'name': 'var3', 'type': 'discrete', 'domain': (3,8,10),'dimensionality': 1},]
#                {'name': 'var4', 'type': 'categorical', 'domain': (0,1,2),'dimensionality': 1},
#                {'name': 'var5', 'type': 'continuous', 'domain': (-1,2),'dimensionality': 1}]
myBopt = BayesianOptimization(f=optim_result,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 10,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='EI',        # Expected Improvement
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=True)
#%% 8 hour 20 min 19 sec 811 iterations
%%time
max_iter = 900       ## maximum number of iterations
max_time = 30000      ## maximum allowed time
eps      = 1e-3     ## tolerance, max distance between consicutive evaluations.
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=0, verbosity=True)

#%%
from os.path import join
savedir = "E:\OneDrive - Washington University in St. Louis\Optimizer_Tuning\BO_autotune";
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_tune.npz"), X=myBopt.X, Y=myBopt.Y, Y_best=myBopt.Y_best, domain=mixed_domain, mode_dict=mode_dict)
#%%
import pandas as pd
scores_tab = pd.DataFrame(np.append(myBopt.X,-myBopt.Y,1), columns=["pop_size","select_ratio","lr","mu_init","mu_rate","mode","scores"])
scores_tab.to_csv(join(savedir, "ZOHA_Sphere_lr_euclid_BO_tune.csv"))
#%%
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_obj.npz"), BOobj=myBopt) # not successful since there some local functions in BO object.
#%%
%%time
myBopt_short = BayesianOptimization(f=optim_result_short,                     # Objective function
                             domain=mixed_domain,          # Box-constraints of the problem
                             initial_design_numdata = 20,   # Number data initial design
                             initial_design_type="random",
                             acquisition_optimizer_type='lbfgs',
                             acquisition_type='LCB',        # Expected Improvement # "LCB" seems to create more variablity than "EI"
                             exact_feval = False,         # True evaluations, no sample noise
                             maximize=True)
#%%  8 hour 20 min 19 sec 811 iterations
max_iter = 400       ## maximum number of iterations
max_time = 10000      ## maximum allowed time
eps      = 1e-2     ## tolerance, max distance between consicutive evaluations.
myBopt_short.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=True)
#
np.savez(join(savedir, "ZOHA_Sphere_lr_euclid_BO_short_tune.npz"), X=myBopt_short.X, Y=myBopt_short.Y, Y_best=myBopt_short.Y_best, domain=mixed_domain, mode_dict=mode_dict)
scores_short_tab = pd.DataFrame(np.append(myBopt_short.X,-myBopt_short.Y,1), columns=["pop_size","select_ratio","lr","mu_init","mu_rate","mode","scores"])
scores_short_tab.to_csv(join(savedir, "ZOHA_Sphere_lr_euclid_BO_short_tune.csv"))