from Generator import Generator
from DynamicParameter import DynamicParameter
import utils
import numpy as np
import os
from shutil import copyfile, rmtree
from time import time


class Optimizer:
    def __init__(self, recorddir, random_seed=None, thread=None):
        if thread is not None:
            assert isinstance(thread, int), 'thread must be an integer'
        assert os.path.isdir(recorddir)
        assert isinstance(random_seed, int) or random_seed is None, 'random_seed must be an integer or None'

        self._generator = None
        self._istep = 0  # step num of Optimizer start from 0, at the end of each step _istep is 1...maxstep
        self._dynparams = {}

        self._curr_samples = None  # array of codes
        self._curr_images = None  # list of image arrays
        self._curr_sample_idc = None  # range object
        self._curr_sample_ids = None  # list
        self._next_sample_idx = 0  # scalar

        self._best_code = None
        self._best_score = None

        self._thread = thread
        if thread is not None:
            recorddir = os.path.join(recorddir, 'thread%02d' % self._thread)
            if not os.path.isdir(recorddir):
                os.mkdir(recorddir)
        self._recorddir = recorddir

        self._random_generator = np.random.RandomState()
        if random_seed is not None:
            if self._thread is not None:
                print('random seed set to %d for optimizer (thread %d)' % (random_seed, self._thread))
            else:
                print('random seed set to %d for optimizer' % random_seed)
            self._random_seed = random_seed
            self._random_generator = np.random.RandomState(seed=self._random_seed)

    def load_generator(self):
        self._generator = Generator()
        self._prepare_images()

    def _prepare_images(self):
        ''' Use generator to generate image from each code in `_curr_samples`
        These images will finally be used in `scorer`
        '''
        if self._generator is None:
            raise RuntimeError('generator not loaded. please run optimizer.load_generator() first')

        curr_images = []
        for sample in self._curr_samples:
            im_arr = self._generator.visualize(sample)
            curr_images.append(im_arr)
        self._curr_images = curr_images

    def step(self, scores):
        '''Take in score for each sample and generate a next generation of samples.'''
        raise NotImplementedError

    def save_current_state(self, image_size=None):
        utils.write_images(self._curr_images, self._curr_sample_ids, self._recorddir, image_size)
        utils.write_codes(self._curr_samples, self._curr_sample_ids, self._recorddir)

    def save_current_codes(self):
        utils.write_codes(self._curr_samples, self._curr_sample_ids, self._recorddir)

    @property
    def current_images(self):  # Wrapper of `_curr_images`
        if self._curr_images is None:
            raise RuntimeError('Current images have not been initialized. Is generator loaded?')
        return self._curr_images

    @property
    def current_images_copy(self):
        return list(np.array(self._curr_images).copy())

    @property
    def current_image_ids(self):
        return self._curr_sample_ids

    @property
    def curr_image_idc(self):
        return self._curr_sample_idc

    @property
    def nsamples(self):
        return len(self._curr_samples)

    @property
    def dynamic_parameters(self):
        return self._dynparams


def mutate(population, genealogy, mutation_size, mutation_rate, random_generator):
    do_mutate = random_generator.random_sample(population.shape) < mutation_rate
    population_new = population.copy()
    population_new[do_mutate] += random_generator.normal(loc=0, scale=mutation_size, size=np.sum(do_mutate))
    genealogy_new = ['%s+mut' % gen for gen in genealogy]
    return population_new, genealogy_new


def mate(population, genealogy, fitness, new_size, random_generator, skew=0.5):
    """
    fitness > 0
    """
    # clean data
    assert len(population) == len(genealogy)
    assert len(population) == len(fitness)
    if np.max(fitness) == 0:
        fitness[np.argmax(fitness)] = 0.001
    if np.min(fitness) <= 0:
        fitness[fitness <= 0] = np.min(fitness[fitness > 0])

    fitness_bins = np.cumsum(fitness)
    fitness_bins /= fitness_bins[-1]
    parent1s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    parent2s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    new_samples = np.empty((new_size, population.shape[1]))
    new_genealogy = []
    for i in range(new_size):
        parentage = random_generator.random_sample(population.shape[1]) < skew
        new_samples[i, parentage] = population[parent1s[i]][parentage]
        new_samples[i, ~parentage] = population[parent2s[i]][~parentage]
        new_genealogy.append('%s+%s' % (genealogy[parent1s[i]], genealogy[parent2s[i]]))
    return new_samples, new_genealogy


class Genetic(Optimizer):
    def __init__(self, population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
                 parental_skew=0.5, n_conserve=0, random_seed=None, thread=None):
        super(Genetic, self).__init__(recorddir, random_seed, thread)

        # various parameters
        self._popsize = int(population_size)
        self._mut_rate = float(mutation_rate)
        self._mut_size = float(mutation_size)
        self._kT_mul = float(kT_multiplier)
        self._kT = None  # deprecated; will be overwritten
        self._n_conserve = int(n_conserve)
        assert (self._n_conserve < self._popsize)
        self._parental_skew = float(parental_skew)

        # initialize dynamic parameters & their types
        self._dynparams['mutation_rate'] = \
            DynamicParameter('d', self._mut_rate, 'probability that each gene will mutate at each step')
        self._dynparams['mutation_size'] = \
            DynamicParameter('d', self._mut_size, 'stdev of the stochastic size of mutation')
        self._dynparams['kT_multiplier'] = \
            DynamicParameter('d', self._kT_mul, 'used to calculate kT; kT = kT_multiplier * stdev of scores')
        self._dynparams['n_conserve'] = \
            DynamicParameter('i', self._n_conserve, 'number of best individuals kept unmutated in each step')
        self._dynparams['parental_skew'] = \
            DynamicParameter('d', self._parental_skew, 'amount inherited from one parent; 1 means no recombination')
        self._dynparams['population_size'] = \
            DynamicParameter('i', self._popsize, 'size of population')

        # initialize samples & indices
        self._init_population = self._random_generator.normal(loc=0, scale=1, size=(self._popsize, 4096))
        self._init_population_dir = None
        self._init_population_fns = None
        self._curr_samples = self._init_population.copy()  # curr_samples is current population of codes
        self._genealogy = ['standard_normal'] * self._popsize
        self._curr_sample_idc = range(self._popsize)
        self._next_sample_idx = self._popsize
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]

        # # reset random seed to ignore any calls during init
        # if random_seed is not None:
        #     random_generator.seed(random_seed)

    def load_init_population(self, initcodedir, size):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        assert size <= self._popsize, 'size %d too big for population of size %d' % (size, self._popsize)
        # load codes
        init_population, genealogy = utils.load_codes2(initcodedir, size)
        # fill the rest of population if size==len(codes) < population size
        if len(init_population) < self._popsize:
            remainder_size = self._popsize - len(init_population)
            remainder_pop, remainder_genealogy = mate(
                init_population, genealogy,  # self._curr_sample_ids[:size],
                np.ones(len(init_population)), remainder_size,
                self._random_generator, self._parental_skew
            )
            remainder_pop, remainder_genealogy = mutate(
                remainder_pop, remainder_genealogy, self._mut_size, self._mut_rate, self._random_generator
            )
            init_population = np.concatenate((init_population, remainder_pop))
            genealogy = genealogy + remainder_genealogy
        # apply
        self._init_population = init_population
        self._init_population_dir = initcodedir
        self._init_population_fns = genealogy  # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        self._genealogy = ['[init]%s' % g for g in genealogy]
        # no update for idc, idx, ids because popsize unchanged
        try:
            self._prepare_images()
        except RuntimeError:  # this means generator not loaded; on load, images will be prepared
            pass

    def save_init_population(self):
        '''Record experimental parameter: initial population
        in the directory "[:recorddir]/init_population" '''
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None), \
            'please load init population first by calling load_init_population();' + \
            'if init is not loaded from file, it can be found in experiment backup_images folder after experiment runs'
        recorddir = os.path.join(self._recorddir, 'init_population')
        try:
            os.mkdir(recorddir)
        except OSError as e:
            if e.errno == 17:
                # ADDED Sep.17, To let user delete the directory if existing during the system running.
                chs = input("Dir %s exist input y to delete the dir and write on it, n to exit" % recorddir)
                if chs is 'y':
                    print("Directory %s all removed." % recorddir)
                    rmtree(recorddir)
                    os.mkdir(recorddir)
                else:
                    raise OSError('trying to save init population but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(recorddir, fn))

    def step(self, scores, no_image=False):
        # clean variables
        assert len(scores) == len(self._curr_samples), \
            'number of scores (%d) != population size (%d)' % (len(scores), len(self._curr_samples))
        new_size = self._popsize  # this may != len(curr_samples) if it has been dynamically updated
        new_samples = np.empty((new_size, self._curr_samples.shape[1]))
        # instead of chaining the genealogy, alias it at every step
        curr_genealogy = np.array(self._curr_sample_ids, dtype=str)
        new_genealogy = [''] * new_size  # np array not used because str len will be limited by len at init

        # deal with nan scores:
        nan_mask = np.isnan(scores)
        n_nans = int(np.sum(nan_mask))
        valid_mask = ~nan_mask
        n_valid = int(np.sum(valid_mask))
        if n_nans > 0:
            print('optimizer: missing %d scores for samples %s' % (
            n_nans, str(np.array(self._curr_sample_idc)[nan_mask])))
            if n_nans > new_size:
                print('Warning: n_nans > new population_size because population_size has just been changed AND ' +
                      'too many images failed to score. This will lead to arbitrary loss of some nan score images.')
            if n_nans > new_size - self._n_conserve:
                print('Warning: n_nans > new population_size - self._n_conserve. ' +
                      'IFF population_size has just been changed, ' +
                      'this will lead to aribitrary loss of some/all nan score images.')
            # carry over images with no scores
            thres_n_nans = min(n_nans, new_size)
            new_samples[-thres_n_nans:] = self._curr_samples[nan_mask][-thres_n_nans:]
            new_genealogy[-thres_n_nans:] = curr_genealogy[nan_mask][-thres_n_nans:]

        # if some images have scores
        if n_valid > 0:
            valid_scores = scores[valid_mask]
            self._kT = max((np.std(valid_scores) * self._kT_mul, 1e-8))  # prevents underflow kT = 0
            print('kT: %f' % self._kT)
            sort_order = np.argsort(valid_scores)[::-1]  # sort from high to low
            valid_scores = valid_scores[sort_order]
            # Note: if new_size is smalled than n_valid, low ranking images will be lost
            thres_n_valid = min(n_valid, new_size)
            new_samples[:thres_n_valid] = self._curr_samples[valid_mask][sort_order][:thres_n_valid]
            new_genealogy[:thres_n_valid] = curr_genealogy[valid_mask][sort_order][:thres_n_valid]

            # if need to generate new samples
            if n_nans < new_size - self._n_conserve:
                fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)
                # skips first n_conserve samples
                n_mate = new_size - self._n_conserve - n_nans
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    mate(
                        new_samples[:thres_n_valid], new_genealogy[:thres_n_valid],
                        fitness, n_mate, self._random_generator, self._parental_skew
                    )
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    mutate(
                        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid],
                        self._mut_size, self._mut_rate, self._random_generator
                    )

            # if any score turned out to be best
            if self._best_score is None or self._best_score < valid_scores[0]:
                self._best_score = valid_scores[0]
                self._best_code = new_samples[0].copy()

        self._istep += 1
        self._curr_samples = new_samples
        self._genealogy = new_genealogy
        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + new_size)  # cumulative id .
        self._next_sample_idx += new_size
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        if not no_image:
            self._prepare_images()

    def step_simple(self, scores, codes):
        """ Taking scores and codes from outside to return new codes,
        without generating images
        Used in cases when the images are better handled in outer objects like Experiment object

        Discard the nan handling part!
        Discard the genealogy recording part
        """
        assert len(scores) == len(codes), \
            'number of scores (%d) != population size (%d)' % (len(scores), len(codes))
        new_size = self._popsize  # this may != len(curr_samples) if it has been dynamically updated
        new_samples = np.empty((new_size, codes.shape[1]))
        # instead of chaining the genealogy, alias it at every step
        curr_genealogy = np.array(self._curr_sample_ids, dtype=str)
        new_genealogy = [''] * new_size  # np array not used because str len will be limited by len at init

        # deal with nan scores:
        nan_mask = np.isnan(scores)
        n_nans = int(np.sum(nan_mask))
        valid_mask = ~nan_mask
        n_valid = int(np.sum(valid_mask))
        assert n_nans == 0  # discard the part dealing with nans
        # if some images have scores
        valid_scores = scores[valid_mask]
        self._kT = max((np.std(valid_scores) * self._kT_mul, 1e-8))  # prevents underflow kT = 0
        print('kT: %f' % self._kT)
        sort_order = np.argsort(valid_scores)[::-1]  # sort from high to low
        valid_scores = valid_scores[sort_order]
        # Note: if new_size is smalled than n_valid, low ranking images will be lost
        thres_n_valid = min(n_valid, new_size)
        new_samples[:thres_n_valid] = codes[valid_mask][sort_order][:thres_n_valid]
        new_genealogy[:thres_n_valid] = curr_genealogy[valid_mask][sort_order][:thres_n_valid]

        fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)
        # skips first n_conserve samples
        n_mate = new_size - self._n_conserve - n_nans
        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
            mate(
                new_samples[:thres_n_valid], new_genealogy[:thres_n_valid],
                fitness, n_mate, self._random_generator, self._parental_skew
            )
        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
            mutate(
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid],
                self._mut_size, self._mut_rate, self._random_generator
            )

        self._istep += 1
        self._genealogy = new_genealogy
        self._curr_samples = new_samples
        self._genealogy = new_genealogy
        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + new_size)  # cumulative id .
        self._next_sample_idx += new_size
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        return new_samples

    # def step_with_immigration(self, scores, immigrants, immigrant_scores):
    #     assert len(immigrants.shape) == 2, 'population is not batch sized (dim != 2)'
    #     self._curr_samples = np.concatenate((self._curr_samples, immigrants))
    #     scores = np.concatenate((scores, immigrant_scores))
    #     self.step(scores)

    def add_immigrants(self, codedir, size, ignore_conserve=False):
        if not ignore_conserve:
            assert size <= len(self._curr_samples) - self._n_conserve, \
                'size of immigrantion should be <= size of unconserved population because ignore_conserve is False'
        else:
            assert size < len(self._curr_samples), 'size of immigrantion should be < size of population'
            if size > len(self._curr_samples) - self._n_conserve:
                print('Warning: some conserved codes are being overwritten')

        immigrants, immigrant_codefns = utils.load_codes2(codedir, size)
        n_immi = len(immigrants)
        n_conserve = len(self._curr_samples) - n_immi
        self._curr_samples = np.concatenate((self._curr_samples[:n_conserve], immigrants))
        self._genealogy = self._genealogy[:n_conserve] + ['[immi]%s' % fn for fn in immigrant_codefns]
        next_sample_idx = self._curr_sample_idc[n_conserve] + n_immi
        self._curr_sample_idc = range(self._curr_sample_idc[0], next_sample_idx)
        self._next_sample_idx = next_sample_idx
        if self._thread is None:
            new_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        else:
            new_sample_ids = ['thread%02d_gen%03d_%06d' %
                              (self._thread, self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        self._curr_sample_ids = self._curr_sample_ids[:n_conserve] + new_sample_ids
        self._prepare_images()

    def update_dynamic_parameters(self):
        if self._dynparams['mutation_rate'].value != self._mut_rate:
            self._mut_rate = self._dynparams['mutation_rate'].value
            print('updated mutation_rate to %f at step %d' % (self._mut_rate, self._istep))
        if self._dynparams['mutation_size'].value != self._mut_size:
            self._mut_size = self._dynparams['mutation_size'].value
            print('updated mutation_size to %f at step %d' % (self._mut_size, self._istep))
        if self._dynparams['kT_multiplier'].value != self._kT_mul:
            self._kT_mul = self._dynparams['kT_multiplier'].value
            print('updated kT_multiplier to %.2f at step %d' % (self._kT_mul, self._istep))
        if self._dynparams['parental_skew'].value != self._parental_skew:
            self._parental_skew = self._dynparams['parental_skew'].value
            print('updated parental_skew to %.2f at step %d' % (self._parental_skew, self._istep))
        if self._dynparams['population_size'].value != self._popsize or \
                self._dynparams['n_conserve'].value != self._n_conserve:
            n_conserve = self._dynparams['n_conserve'].value
            popsize = self._dynparams['population_size'].value
            if popsize < n_conserve:  # both newest
                if popsize == self._popsize:  # if popsize hasn't changed
                    self._dynparams['n_conserve'].set_value(self._n_conserve)
                    print('rejected n_conserve update: new n_conserve > old population_size')
                else:  # popsize has changed
                    self._dynparams['population_size'].set_value(self._popsize)
                    print('rejected population_size update: new population_size < new/old n_conserve')
                    if n_conserve <= self._popsize:
                        self._n_conserve = n_conserve
                        print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))
                    else:
                        self._dynparams['n_conserve'].set_value(self._n_conserve)
                        print('rejected n_conserve update: new n_conserve > old population_size')
            else:
                if self._popsize != popsize:
                    self._popsize = popsize
                    print('updated population_size to %d at step %d' % (self._popsize, self._istep))
                if self._n_conserve != n_conserve:
                    self._n_conserve = n_conserve
                    print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))

    def save_current_genealogy(self):
        savefpath = os.path.join(self._recorddir, 'genealogy_gen%03d.npz' % self._istep)
        save_kwargs = {'image_ids': np.array(self._curr_sample_ids, dtype=str),
                       'genealogy': np.array(self._genealogy, dtype=str)}
        utils.savez(savefpath, save_kwargs)

    @property
    def generation(self):
        '''Return current step number'''
        return self._istep


from numpy.linalg import norm
from numpy.random import randn
from numpy import exp, floor, log, log2, sqrt, zeros, eye, ones, diag
from numpy import exp, sqrt, real, eye, zeros, dot, cos, sin


# from numpy.linalg import norm

class CMAES(Optimizer):
    # functions to be added
    #         load_init_population(initcodedir, size=population_size)
    #         save_init_population()
    #         step()
    # Note this is a single step version of CMAES
    def __init__(self, recorddir, space_dimen, init_sigma=None, population_size=None, maximize=True,
                 random_seed=None, thread=None):
        super(CMAES, self).__init__(recorddir, random_seed, thread)
        # --------------------  Initialization --------------------------------
        # assert len(initX) == N or initX.size == N
        # xmean = np.array(initX)
        # xmean.shape = (-1, 1)

        N = space_dimen
        self.space_dimen = space_dimen
        # Overall control parameter
        self.maximize = maximize  # if the program is to maximize or to minimize

        # Strategy parameter setting: Selection
        if population_size is None:
            self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
            # the relation between dimension and population size.
        else:
            self.lambda_ = population_size  # use custom specified population size
        mu = self.lambda_ / 2  # number of parents/points for recombination
        #  Select half the population size as parents
        weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
        self.mu = int(floor(mu))
        self.weights = weights / sum(weights)  # normalize recombination weights array
        mueff = self.weights.sum() ** 2 / sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.weights.shape = (1, -1)  # Add the 1st dim to the weights mat
        self.mueff = mueff  # add to class variable
        self.sigma = init_sigma  # Note by default, sigma is None here.
        print("Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\nInitial sigma: %.3f"
              % (self.space_dimen, self.lambda_, self.mu, self.sigma))

        # Strategy parameter settiself.weightsng: Adaptation
        self.cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C
        self.cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control
        self.c1 = 2 / ((N + 1.3) ** 2 + mueff)  # learning rate for rank-one update of C
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # and for rank-mu update
        self.damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + self.cs  # damping for sigma
        # usually close to 1
        print("cc=%.3f, cs=%.3f, c1=%.3f, cmu=%.3f, damps=%.3f" % (self.cc, self.cs, self.c1, self.cmu, self.damps))

        self.xmean = zeros((1, N))
        self.xold = zeros((1, N))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, N))
        self.ps = zeros((1, N))  # evolution paths for C and sigma
        self.B = eye(N)  # B defines the coordinate system
        self.D = ones(N)  # diagonal D defines the scaling
        self.C = self.B * diag(self.D ** 2) * self.B.T  # covariance matrix C
        self.invsqrtC = self.B * diag(1 / self.D) * self.B.T  # C^-1/2
        # self.D.shape = (-1, 1)

        self.eigeneval = 0  # track update of B and D
        self.counteval = 0
        self.chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (
                21 * N ** 2))  # expectation of ||N(0,I)|| == norm(randn(N,1)) in 1/N expansion formula

    def step(self, scores):
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to conform with other code, this part is transposed.
        N = self.space_dimen
        lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        cc, cs, c1, cmu, damps = self.cc, self.cs, self.c1, self.cmu, self.damps
        sigma, C, B, D, invsqrtC, ps, pc, = self.sigma, self.C, self.B, self.D, self.invsqrtC, self.ps, self.pc,
        # set short name for everything

        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort(scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        # scores = scores[code_sort_index]  # Ascending order. minimization

        if self._istep == 0:
            # if without initialization, the first xmean is evaluated from weighted average all the natural images
            self.xmean = self.weights @ self._curr_samples[code_sort_index[0:mu], :]
        else:
            self.xold = self.xmean
            self.xmean = self.weights @ self._curr_samples[code_sort_index[0:mu],
                                        :]  # Weighted recombination, new mean value

            # Cumulation: Update evolution paths
            norm_step_len = (self.xmean - self.xold) / sigma
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (norm_step_len @ invsqrtC)
            hsig = norm(ps) / chiN / sqrt(1 - (1 - cs) ** (2 * self.counteval / lambda_)) < (1.4 + 2 / (N + 1))
            pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * norm_step_len

            # Adapt covariance matrix C
            # x_tmp = (1 / sigma) * (self._curr_samples[code_sort_index[0:mu], :] - self.xold)
            #
            # C = ((1 - c1 - cmu) * C  # regard old matrix
            #      + c1 * (pc.T @ pc  # plus rank one update
            #              + (1 - hsig) * cc * (2 - cc) * C)  # minor correction if hsig==0
            #      + cmu * x_tmp.T @ diag(self.weights.flat) @ x_tmp)  # plus rank mu update

            # Adapt step size sigma
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1))
            # self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(ps) / self.chiN - 1))
            print("sigma: %.2f" % sigma)

            # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
            # if self.counteval - self.eigeneval > lambda_ / (c1 + cmu) / N / 10:  # to achieve O(N^2)
            #     # FIXME: Seems every time after this decomposition , the score went down!
            #     t1 = time()
            #     self.eigeneval = self.counteval
            #     C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            #     [D, B] = np.linalg.eig(C)  # eigen decomposition, B==normalized eigenvectors
            #     print("Spectrum Range:%.2f, %.2f" % (D.min(), D.max()))
            #     D = sqrt(D)  # D is a vector of standard deviations now
            #     invsqrtC = B @ diag(1 / D) @ B.T
            #     t2 = time()
            #     print("Cov Matrix Eigenvalue Decomposition (linalg) time cost: %.2f s" % (t2-t1))
        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.lambda_, N))
        new_ids = []
        for k in range(self.lambda_):
            new_samples[k:k + 1, :] = self.xmean + sigma * ((D * randn(1, N)) @ B.T)  # m + sig * Normal(0,C)
            # Clever way to generate multivariate gaussian!!
            # Stretch the guassian hyperspher with D and transform the
            # ellipsoid by B mat linear transform between coordinates
            new_ids.append("gen%03d_%06d" % (self._istep, self.counteval))
            # assign id to newly generated images. These will be used as file names at 2nd round
            self.counteval += 1

        self.sigma, self.C, self.B, self.D, self.invsqrtC, self.ps, self.pc, = sigma, C, B, D, invsqrtC, ps, pc,

        self._istep += 1
        self._curr_samples = new_samples
        self._curr_sample_ids = new_ids
        self._prepare_images()

    def save_optimizer_state(self):
        # if needed, a save Optimizer status function. Automatic save the optimization parameters at 1st step.
        if self._istep == 1:
            optim_setting = {"space_dimen": self.space_dimen, "population_size": self.lambda_,
                             "select_num": self.mu, "weights": self.weights,
                             "cc": self.cc, "cs": self.cs, "c1": self.c1, "cmu": self.cmu, "damps": self.damps}
            utils.savez(os.path.join(self._recorddir, "optimizer_setting.npz"), optim_setting)
        utils.savez(os.path.join(self._recorddir, "optimizer_state_block%03d.npz" % self._istep),
                    {"sigma": self.sigma, "C": self.C, "D": self.D, "ps": self.ps, "pc": self.pc})

    def load_init_population(self, initcodedir, size):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        assert size <= self.lambda_, 'size %d too big for population of size %d' % (size, self.lambda_)
        # load codes
        init_population, genealogy = utils.load_codes2(initcodedir, size)  # find `size` # of images in the target dir.
        # if needed can be control the number
        # apply
        self._init_population = init_population
        self._init_population_dir = initcodedir
        self._init_population_fns = genealogy  # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        self._curr_sample_ids = genealogy.copy()
        # self._curr_samples = self._curr_samples.T
        # no update for idc, idx, ids because popsize unchanged
        try:
            self._prepare_images()
        except RuntimeError:  # this means generator not loaded; on load, images will be prepared
            pass

    def save_init_population(self):
        '''Record experimental parameter: initial population
        in the directory "[:recorddir]/init_population" '''
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None), \
            'please load init population first by calling load_init_population();' + \
            'if init is not loaded from file, it can be found in experiment backup_images folder after experiment runs'
        recorddir = os.path.join(self._recorddir, 'init_population')
        try:
            os.mkdir(recorddir)
        except OSError as e:
            if e.errno == 17:
                # ADDED Sep.17, To let user delete the directory if existing during the system running.
                chs = input("Dir %s exist input y to delete the dir and write on it, n to exit" % recorddir)
                if chs is 'y':
                    print("Directory %s all removed." % recorddir)
                    rmtree(recorddir)
                    os.mkdir(recorddir)
                else:
                    raise OSError('trying to save init population but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(recorddir, fn))


class CholeskyCMAES(Optimizer):
    # functions to be added
    #         load_init_population(initcodedir, size=population_size)
    #         save_init_population()
    #         step()
    """ Note this is a variant of CMAES Cholesky suitable for high dimensional optimization"""

    def __init__(self, recorddir, space_dimen, init_sigma=None, init_code=None, population_size=None, Aupdate_freq=None,
                 maximize=True, random_seed=None, thread=None, optim_params={}):
        super(CholeskyCMAES, self).__init__(recorddir, random_seed, thread)
        # --------------------  Initialization --------------------------------
        # assert len(initX) == N or initX.size == N
        # xmean = np.array(initX)
        # xmean.shape = (-1, 1)

        N = space_dimen
        self.space_dimen = space_dimen
        # Overall control parameter
        self.maximize = maximize  # if the program is to maximize or to minimize

        # Strategy parameter setting: Selection
        if population_size is None:
            self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
            # the relation between dimension and population size.
        else:
            self.lambda_ = population_size  # use custom specified population size
        mu = self.lambda_ / 2  # number of parents/points for recombination
        #  Select half the population size as parents
        weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
        self.mu = int(floor(mu))
        self.weights = weights / sum(weights)  # normalize recombination weights array
        mueff = self.weights.sum() ** 2 / sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.weights.shape = (1, -1)  # Add the 1st dim 1 to the weights mat
        self.mueff = mueff  # add to class variable
        self.sigma = init_sigma  # Note by default, sigma is None here.
        print("Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\nInitial sigma: %.3f"
              % (self.space_dimen, self.lambda_, self.mu, self.sigma))

        # Strategy parameter settiself.weightsng: Adaptation
        # self.cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C
        # self.cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control
        # self.c1 = 2 / ((N + 1.3) ** 2 + mueff)  # learning rate for rank-one update of C
        # self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))  # and for rank-mu update
        # self.damps = 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + self.cs  # damping for sigma
        self.cc = 4 / (N + 4)  # defaultly  0.0009756
        self.cs = sqrt(mueff) / (sqrt(mueff) + sqrt(N))  # 0.0499
        self.c1 = 2 / (N + sqrt(2)) ** 2  # 1.1912701410022985e-07
        if "cc" in optim_params.keys():  # if there is outside value for these parameter, overwrite them
            self.cc = optim_params["cc"]
        if "cs" in optim_params.keys():
            self.cs = optim_params["cs"]
        if "c1" in optim_params.keys():
            self.c1 = optim_params["c1"]
        self.damps = 1 + self.cs + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1)  # damping for sigma usually  close to 1

        print("cc=%.3f, cs=%.3f, c1=%.3f damps=%.3f" % (self.cc, self.cs, self.c1, self.damps))
        if init_code is not None:
            self.init_x = np.asarray(init_code)
            self.init_x.shape = (1, N)
        else:
            self.init_x = None  # FIXED Nov. 1st
        self.xmean = zeros((1, N))
        self.xold = zeros((1, N))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, N))
        self.ps = zeros((1, N))  # evolution paths for C and sigma
        self.A = eye(N, N)  # covariant matrix is represent by the factors A * A '=C
        self.Ainv = eye(N, N)

        self.eigeneval = 0  # track update of B and D
        self.counteval = 0
        if Aupdate_freq is None:
            self.update_crit = self.lambda_ / self.c1 / N / 10
        else:
            self.update_crit = Aupdate_freq * self.lambda_
        self.chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        # expectation of ||N(0,I)|| == norm(randn(N,1)) in 1/N expansion formula

    def step(self, scores, no_image=False):
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to conform with other code, this part is transposed.

        # set short name for everything to simplify equations
        N = self.space_dimen
        lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        cc, cs, c1, damps = self.cc, self.cs, self.c1, self.damps
        sigma, A, Ainv, ps, pc, = self.sigma, self.A, self.Ainv, self.ps, self.pc,

        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort(scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        # scores = scores[code_sort_index]  # Ascending order. minimization

        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            if self.init_x is None:
                self.xmean = self.weights @ self._curr_samples[code_sort_index[0:mu], :]
            else:
                self.xmean = self.init_x
        else:
            self.xold = self.xmean
            self.xmean = self.weights @ self._curr_samples[code_sort_index[0:mu],
                                        :]  # Weighted recombination, new mean value

            # Cumulation statistics through steps: Update evolution paths
            randzw = self.weights @ self.randz[code_sort_index[0:mu], :]
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * randzw
            pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * randzw @ A

            # Adapt step size sigma
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1))
            # self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(ps) / self.chiN - 1))
            print("sigma: %.2f" % sigma)

            # Update A and Ainv with search path
            if self.counteval - self.eigeneval > self.update_crit:  # to achieve O(N ^ 2) do decomposition less frequently
                self.eigeneval = self.counteval
                t1 = time()
                v = pc @ Ainv
                normv = v @ v.T
                # Directly update the A Ainv instead of C itself
                A = sqrt(1 - c1) * A + sqrt(1 - c1) / normv * (
                            sqrt(1 + normv * c1 / (1 - c1)) - 1) * v @ pc.T  # FIXME, dimension error
                Ainv = 1 / sqrt(1 - c1) * Ainv - 1 / sqrt(1 - c1) / normv * (
                            1 - 1 / sqrt(1 + normv * c1 / (1 - c1))) * Ainv @ v.T @ v
                t2 = time()
                print("A, Ainv update! Time cost: %.2f s" % (t2 - t1))
            # if self.counteval - self.eigeneval > lambda_ / (c1 + cmu) / N / 10:  # to achieve O(N^2)
            #     # FIXME: Seems every time after this decomposition , the score went down!
            #     t1 = time()
            #     self.eigeneval = self.counteval
            #     C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            #     [D, B] = np.linalg.eig(C)  # eigen decomposition, B==normalized eigenvectors
            #     print("Spectrum Range:%.2f, %.2f" % (D.min(), D.max()))
            #     D = sqrt(D)  # D is a vector of standard deviations now
            #     invsqrtC = B @ diag(1 / D) @ B.T
            #     t2 = time()
            #     print("Cov Matrix Eigenvalue Decomposition (linalg) time cost: %.2f s" % (t2-t1))

        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.lambda_, N))
        new_ids = []
        self.randz = randn(self.lambda_, N)  # save the random number for generating the code.
        for k in range(self.lambda_):
            new_samples[k:k + 1, :] = self.xmean + sigma * (self.randz[k, :] @ A)  # m + sig * Normal(0,C)
            # Clever way to generate multivariate gaussian!!
            # Stretch the guassian hyperspher with D and transform the
            # ellipsoid by B mat linear transform between coordinates
            if self._thread is None:
                new_ids.append("gen%03d_%06d" % (self._istep, self.counteval))
            else:
                new_ids.append('thread%02d_gen%03d_%06d' %
                               (self._thread, self._istep, self.counteval))
            # FIXME A little inconsistent with the naming at line 173/175/305/307 esp. for gen000 code
            # assign id to newly generated images. These will be used as file names at 2nd round
            self.counteval += 1

        self.sigma, self.A, self.Ainv, self.ps, self.pc = sigma, A, Ainv, ps, pc,
        self._istep += 1
        self._curr_samples = new_samples
        self._curr_sample_ids = new_ids
        if not no_image:
            self._prepare_images()
        return new_samples

    def step_simple(self, scores, codes):
        """ Taking scores and codes to return new codes, without generating images
        Used in cases when the images are better handled in outer objects like Experiment object
        """
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to confirm with other code, this part is transposed.

        # set short name for everything to simplify equations
        N = self.space_dimen
        lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        cc, cs, c1, damps = self.cc, self.cs, self.c1, self.damps
        sigma, A, Ainv, ps, pc, = self.sigma, self.A, self.Ainv, self.ps, self.pc,

        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort(scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        # scores = scores[code_sort_index]  # Ascending order. minimization

        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            if self.init_x is None:
                self.xmean = self.weights @ codes[code_sort_index[0:mu], :]
            else:
                self.xmean = self.init_x
        else:
            self.xold = self.xmean
            self.xmean = self.weights @ codes[code_sort_index[0:mu], :]  # Weighted recombination, new mean value

            # Cumulation statistics through steps: Update evolution paths
            randzw = self.weights @ self.randz[code_sort_index[0:mu], :]
            ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * randzw
            pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * randzw @ A

            # Adapt step size sigma
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1))
            # self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(ps) / self.chiN - 1))
            print("sigma: %.2f" % sigma)

            # Update A and Ainv with search path
            if self.counteval - self.eigeneval > self.update_crit:  # to achieve O(N ^ 2) do decomposition less frequently
                self.eigeneval = self.counteval
                t1 = time()
                v = pc @ Ainv
                normv = v @ v.T
                # Directly update the A Ainv instead of C itself
                A = sqrt(1 - c1) * A + sqrt(1 - c1) / normv * (
                            sqrt(1 + normv * c1 / (1 - c1)) - 1) * v @ pc.T  # FIXME, dimension error
                Ainv = 1 / sqrt(1 - c1) * Ainv - 1 / sqrt(1 - c1) / normv * (
                            1 - 1 / sqrt(1 + normv * c1 / (1 - c1))) * Ainv @ v.T @ v
                t2 = time()
                print("A, Ainv update! Time cost: %.2f s" % (t2 - t1))

        # Generate new sample by sampling from Gaussian distribution
        new_samples = zeros((self.lambda_, N))
        self.randz = randn(self.lambda_, N)  # save the random number for generating the code.
        for k in range(self.lambda_):
            new_samples[k:k + 1, :] = self.xmean + sigma * (self.randz[k, :] @ A)  # m + sig * Normal(0,C)
            # Clever way to generate multivariate gaussian!!
            # Stretch the guassian hyperspher with D and transform the
            # ellipsoid by B mat linear transform between coordinates
            # FIXME A little inconsistent with the naming at line 173/175/305/307 esp. for gen000 code
            # assign id to newly generated images. These will be used as file names at 2nd round
            self.counteval += 1

        self.sigma, self.A, self.Ainv, self.ps, self.pc = sigma, A, Ainv, ps, pc,
        self._istep += 1
        return new_samples

    def save_optimizer_state(self):
        """a save Optimizer status function.
        Automatic save the optimization parameters at 1st step.save the changing parameters if not"""
        if self._istep == 1:
            optim_setting = {"space_dimen": self.space_dimen, "population_size": self.lambda_,
                             "select_num": self.mu, "weights": self.weights,
                             "cc": self.cc, "cs": self.cs, "c1": self.c1, "damps": self.damps, "init_x": self.init_x}
            utils.savez(os.path.join(self._recorddir, "optimizer_setting.npz"), optim_setting)
        utils.savez(os.path.join(self._recorddir, "optimizer_state_block%03d.npz" % self._istep),
                    {"sigma": self.sigma, "A": self.A, "Ainv": self.Ainv, "ps": self.ps, "pc": self.pc})

    def load_init_population(self, initcodedir, size):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        assert size <= self.lambda_, 'size %d too big for population of size %d' % (size, self.lambda_)
        # load codes
        init_population, genealogy = utils.load_codes2(initcodedir, size)  # find `size` # of images in the target dir.
        # if needed can be control the number
        # apply
        self._init_population = init_population
        self._init_population_dir = initcodedir
        self._init_population_fns = genealogy  # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in range(size)]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in range(size)]
        # Note: FIXED on Nov. 14th in consistant with former version of code.
        # self._curr_sample_ids = genealogy.copy()  # FIXED: @Nov.14 keep the nomenclatuere the same
        # no update for idc, idx, ids because popsize unchanged
        try:
            self._prepare_images()
        except RuntimeError:  # this means generator not loaded; on load, images will be prepared
            pass

    def save_init_population(self):
        '''Record experimental parameter: initial population
        in the directory "[:recorddir]/init_population" '''
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None), \
            'please load init population first by calling load_init_population();' + \
            'if init is not loaded from file, it can be found in experiment backup_images folder after experiment runs'
        recorddir = os.path.join(self._recorddir, 'init_population')
        try:
            os.mkdir(recorddir)
        except OSError as e:
            if e.errno == 17:
                # ADDED Sep.17, To let user delete the directory if existing during the system running.
                chs = input("Dir %s exist input y to delete the dir and write on it, n to exit" % recorddir)
                if chs is 'y':
                    print("Directory %s all removed." % recorddir)
                    rmtree(recorddir)
                    os.mkdir(recorddir)
                else:
                    raise OSError('trying to save init population but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(recorddir, fn))


class CholeskyCMAES_Sphere(CholeskyCMAES):
    # functions to be added
    #         load_init_population(initcodedir, size=population_size)
    #         save_init_population()
    #         step()
    """ Note this is a variant of CMAES Cholesky suitable for high dimensional optimization"""

    def __init__(self, recorddir, space_dimen, init_sigma=None, init_code=None, population_size=None, Aupdate_freq=None,
                 sphere_norm=300,
                 maximize=True, random_seed=None, thread=None, optim_params={}):
        super(CholeskyCMAES_Sphere, self).__init__(recorddir, space_dimen, init_sigma=init_sigma, init_code=init_code,
                                                   population_size=population_size, Aupdate_freq=Aupdate_freq,
                                                   maximize=maximize, random_seed=random_seed, thread=thread,
                                                   optim_params=optim_params)
        N = space_dimen
        self.space_dimen = space_dimen
        # Overall control parameter
        self.maximize = maximize  # if the program is to maximize or to minimize

        # Strategy parameter setting: Selection
        if population_size is None:
            self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
            # the relation between dimension and population size.
        else:
            self.lambda_ = population_size  # use custom specified population size
        mu = self.lambda_ / 2  # number of parents/points for recombination
        #  Select half the population size as parents
        weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
        self.mu = int(floor(mu))
        self.weights = weights / sum(weights)  # normalize recombination weights array
        mueff = self.weights.sum() ** 2 / sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.weights.shape = (1, -1)  # Add the 1st dim 1 to the weights mat
        self.mueff = mueff  # add to class variable
        self.sigma = init_sigma  # Note by default, sigma is None here.
        print("Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\nInitial sigma: %.3f"
              % (self.space_dimen, self.lambda_, self.mu, self.sigma))

        # Strategy parameter settiself.weightsng: Adaptation
        self.cc = 4 / (N + 4)  # defaultly  0.0009756
        self.cs = sqrt(mueff) / (sqrt(mueff) + sqrt(N))  # 0.0499
        self.c1 = 2 / (N + sqrt(2)) ** 2  # 1.1912701410022985e-07
        if "cc" in optim_params.keys():  # if there is outside value for these parameter, overwrite them
            self.cc = optim_params["cc"]
        if "cs" in optim_params.keys():
            self.cs = optim_params["cs"]
        if "c1" in optim_params.keys():
            self.c1 = optim_params["c1"]
        self.damps = 1 + self.cs + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1)  # damping for sigma usually  close to 1
        self.MAXSGM = 0.5
        if "MAXSGM" in optim_params.keys():
            self.MAXSGM = optim_params["MAXSGM"]
        self.sphere_norm = sphere_norm
        if "sphere_norm" in optim_params.keys():
            self.sphere_norm = optim_params["sphere_norm"]
        print("cc=%.3f, cs=%.3f, c1=%.3f damps=%.3f" % (self.cc, self.cs, self.c1, self.damps))
        print("Maximum Sigma %.2f" % self.MAXSGM)

        if init_code is not None:
            self.init_x = np.asarray(init_code)
            self.init_x.shape = (1, N)
        else:
            self.init_x = None  # FIXED Nov. 1st
        self.xmean = zeros((1, N))
        self.xold = zeros((1, N))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, N))
        self.ps = zeros((1, N))  # evolution paths for C and sigma
        self.A = eye(N, N)  # covariant matrix is represent by the factors A * A '=C
        self.Ainv = eye(N, N)
        self.tang_codes = zeros((self.lambda_, N))

        self.eigeneval = 0  # track update of B and D
        self.counteval = 0
        if Aupdate_freq is None:
            self.update_crit = self.lambda_ / self.c1 / N / 10
        else:
            self.update_crit = Aupdate_freq * self.lambda_
        self.chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    def step(self, scores, no_image=False):
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to conform with other code, this part is transposed.

        # set short name for everything to simplify equations
        N = self.space_dimen
        lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        cc, cs, c1, damps = self.cc, self.cs, self.c1, self.damps
        sigma, A, Ainv, ps, pc, = self.sigma, self.A, self.Ainv, self.ps, self.pc,

        # obj.codes = codes;

        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort(scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        sorted_score = scores[code_sort_index]
        # print(sorted_score)
        if self._istep == 0:
            print('First generation\n')
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            if self.init_x is None:
                if mu <= len(scores):
                    self.xmean = self.weights @ self._curr_samples[code_sort_index[0:mu], :]
                else:  # if ever the obj.mu (selected population size) larger than the initial population size (init_population is 1 ?)
                    tmpmu = max(floor(len(scores) / 2), 1)  # make sure at least one element!
                    self.xmean = self.weights[:tmpmu] @ self._curr_samples[code_sort_index[:tmpmu], :] / sum(
                        self.weights[:tmpmu])
                self.xmean = self.xmean / norm(self.xmean)
            else:
                self.xmean = self.init_x

        else:  # if not first step

            # print('Not First generation\n')
            self.xold = self.xmean
            xold = self.xold
            # Weighted recombination, move the mean value
            # mean_tangent = self.weights * self.tang_codes(code_sort_index(1:self.mu), :);
            # self.xmean = ExpMap(self.xmean, mean_tangent); # Map the mean tangent onto sphere
            # Do spherical mean in embedding space, not in tangent
            # vector space!
            #                 self.xmean = self.weights * self.codes(code_sort_index(1:self.mu), :);
            #                 self.xmean = self.xmean / norm(self.xmean); # Spherical Mean
            # [vtan_old, vtan_new] = InvExpMap(xold, self.xmean); # Tangent vector of the arc between old and new mean

            vtan_old = self.weights @ self.tang_codes[code_sort_index[0:mu],
                                      :]  # mean in Tangent Space for tang_codes in last generation
            xmean = ExpMap(xold, vtan_old)  # Project to new tangent space
            self.xmean = xmean

            print(norm(vtan_old))
            vtan_new = VecTransport(xold, xmean, vtan_old);  # Transport this vector to new spot
            uni_vtan_old = vtan_old / norm(vtan_old);
            uni_vtan_new = vtan_new / norm(vtan_new);  # uniform the tangent vector
            ps_transp = VecTransport(xold, xmean, ps);  # transport the path vector from old to new mean
            pc_transp = VecTransport(xold, xmean, pc);
            # Cumulation: Update evolution paths
            # In the sphereical case we have to transport the vector to
            # the new center
            # FIXME, pc explode problem????

            # randzw = weights * randz(code_sort_index(1:mu), :); # Note, use the randz saved from last generation
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * randzw;
            # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * randzw * A;
            pc = (1 - cc) * pc_transp + sqrt(
                cc * (2 - cc) * mueff) * vtan_new / sigma;  # do the update in the new tangent space
            # Transport the A and Ainv to the new tangent space
            A = A + A @ uni_vtan_old.T @ (uni_vtan_new - uni_vtan_old) + A @ xold.T @ (
                        xmean - xold)  # transport the A mapping from old to new mean
            Ainv = Ainv + (uni_vtan_new - uni_vtan_old).T @ uni_vtan_old @ Ainv + (xmean - xold).T @ xold @ Ainv
            # Use the new Ainv to transform ps to z space
            ps = (1 - cs) * ps_transp + sqrt(cs * (2 - cs) * mueff) * vtan_new @ Ainv / sigma
            # Adapt step size sigma.
            # Decide whether to grow sigma or shrink sigma by comparing
            # the cumulated path length norm(ps) with expectation chiN
            # chiN = RW_Step_Size_Sphere(lambda, sigma, N);
            sigma = min(self.MAXSGM, sigma * exp((cs / damps) * (norm(real(ps)) * sqrt(N) / chiN - 1)))
            if sigma == self.MAXSGM:
                print("Reach upper limit for sigma! ")
            print("Step %d, sigma: %0.2e, Scores\n" % (self._istep, sigma))
            # disp(A * Ainv)
            # Update the A and Ainv mapping
            if self.counteval - self.eigeneval > self.update_crit:  # to achieve O(N ^ 2)
                self.eigeneval = self.counteval
                t1 = time()
                v = pc @ Ainv
                normv = v @ v.T
                A = sqrt(1 - c1) * A + sqrt(1 - c1) / normv * (
                            sqrt(1 + normv * c1 / (1 - c1)) - 1) * v.T @ pc  # FIXED! dimension error
                Ainv = 1 / sqrt(1 - c1) * Ainv - 1 / sqrt(1 - c1) / normv * (
                            1 - 1 / sqrt(1 + normv * c1 / (1 - c1))) * Ainv @ v.T @ v
                print("A, Ainv update! Time cost: %.2f s" % (time() - t1))
                print("A Ainv Deiviation Norm from identity %.2E",
                      norm(eye(N) - self.A * self.Ainv))  # Deviation from being inverse to each other
        # of first step

        # Generate new sample by sampling from Multivar Gaussian distribution
        self.tang_codes = zeros((self.lambda_, N))
        new_samples = zeros((self.lambda_, N))
        new_ids = []
        self.randz = randn(self.lambda_, N)  # save the random number for generating the code.
        # For optimization path update in the next generation.
        self.tang_codes = self.sigma * (self.randz @ self.A) / sqrt(N)  # sig * Normal(0,C)
        # Clever way to generate multivariate gaussian!!
        self.tang_codes = self.tang_codes - (self.tang_codes @ self.xmean.T) @ self.xmean
        # FIXME, wrap back problem
        # DO SOLVED, by limiting the sigma to small value?
        new_samples = ExpMap(self.xmean, self.tang_codes)
        # Exponential map the tang vector from center to the sphere
        for k in range(self.lambda_):
            if self._thread is None:
                new_ids.append("gen%03d_%06d" % (self._istep, self.counteval))
            else:
                new_ids.append('thread%02d_gen%03d_%06d' %
                               (self._thread, self._istep, self.counteval))
                # FIXME self.A little inconsistent with the naming at line 173/175/305/307 esp. for gen000 code
            # assign id to newly generated images. These will be used as file names at 2nd round
            self.counteval = self.counteval + 1

        # self._istep = self._istep + 1

        self.sigma, self.A, self.Ainv, self.ps, self.pc = sigma, A, Ainv, ps, pc,
        self._istep += 1
        self._curr_samples = new_samples * self.sphere_norm
        self._curr_sample_ids = new_ids
        if not no_image:
            self._prepare_images()


def ExpMap(x, tang_vec):
    '''Assume x is (1, N)'''
    # EPS = 1E-3;
    # assert(abs(norm(x)-1) < EPS)
    # assert(sum(x * tang_vec') < EPS)
    angle_dist = sqrt((tang_vec ** 2).sum(axis=1))  # vectorized
    angle_dist = angle_dist[:, np.newaxis]
    uni_tang_vec = tang_vec / angle_dist
    # x = repmat(x, size(tang_vec, 1), 1); # vectorized
    y = cos(angle_dist) @ x[:] + sin(angle_dist) * uni_tang_vec
    return y


def VecTransport(xold, xnew, v):
    xold = xold / norm(xold)
    xnew = xnew / norm(xnew)
    x_symm_axis = xold + xnew
    v_transport = v - 2 * v @ x_symm_axis.T / norm(
        x_symm_axis) ** 2 * x_symm_axis  # Equation for vector parallel transport along geodesic
    # Don't use dot in numpy, it will have wierd behavior if the array is not single dimensional
    return v_transport
#
# class FDGD(Optimizer):
#     def __init__(self, nsamples, mutation_size, learning_rate, antithetic=True, init_code=None):
#         self._nsamples = int(nsamples)
#         self._mut_size = float(mutation_size)
#         self._lr = float(learning_rate)
#         self._antithetic = antithetic
#         self.parameters = {'mutation_size': mutation_size, 'learning_rate': learning_rate,
#                            'nsamples': nsamples, 'antithetic': antithetic}
#
#         if init_code is not None:
#             self._init_code = init_code.copy().reshape(4096)
#         else:
#             # self.init_code = np.random.normal(loc=0, scale=1, size=(4096))
#             self._init_code = np.zeros(shape=(4096,))
#         self._curr = self._init_code.copy()
#         self._best_code = self._curr.copy()
#         self._best_score = None
#
#         self._pos_isteps = None
#         self._norm_isteps = None
#
#         self._prepare_next_samples()
#
#     def _prepare_next_samples(self):
#         self._pos_isteps = np.random.normal(loc=0, scale=self._mut_size, size=(self._nsamples, len(self._curr)))
#         self._norm_isteps = np.linalg.norm(self._pos_isteps, axis=1)
#
#         pos_samples = self._pos_isteps + self._curr
#         if self._antithetic:
#             neg_samples = -self._pos_isteps + self._curr
#             self._curr_samples = np.concatenate((pos_samples, neg_samples))
#         else:
#             self._curr_samples = np.concatenate((pos_samples, (self._curr.copy(),)))
#
#         self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + len(self._curr_samples))
#         self._next_sample_idx += len(self._curr_samples)
#
#         curr_images = []
#         for sample in self._curr_samples:
#             im_arr = self._generator.visualize(sample)
#             curr_images.append(im_arr)
#         self._curr_images = curr_images
#
#     def step(self, scores):
#         """
#         Use scores for current samples to update samples
#         :param scores: array or list of scalar scores, one for each current sample, in order
#         :param write:
#             if True, immediately writes after samples are prepared.
#             if False, user need to call .write_images(path)
#         :return: None
#         """
#         scores = np.array(scores)
#         assert len(scores) == len(self._curr_samples),\
#             'number of scores (%d) and number of samples (%d) are different' % (len(scores), len(self._curr_samples))
#
#         pos_scores = scores[:self._nsamples]
#         if self._antithetic:
#             neg_scores = scores[self._nsamples:]
#             dscore = (pos_scores - neg_scores) / 2.
#         else:
#             dscore = pos_scores - scores[-1]
#
#         grad = np.mean(dscore.reshape(-1, 1) * self._pos_isteps * (self._norm_isteps ** -2).reshape(-1, 1), axis=0)
#         self._curr += self._lr * grad
#
#         score_argmax = np.argsort(scores)[-1]
#         if self._best_score is None or self._best_score < scores[score_argmax]:
#             self._best_score = scores[score_argmax]
#             self._best_code = self._curr_samples[score_argmax]
#
#         self._prepare_next_samples()
