"""Supporting classes and experimental code for in-silico experiment"""
# Manifold_experiment
import torch_net_utils #net_utils
import net_utils
import utils
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES # newer CMAES api
from utils import load_GAN
from Generator import Generator
from time import time, sleep
import numpy as np
from Optimizer import Genetic, Optimizer  # CholeskyCMAES, Optimizer is the base class for these things
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from os.path import join
from sys import platform
#% Decide the result storage place based on the computer the code is running
if platform == "linux": # cluster
    recorddir = "/scratch/binxu/CNN_data/"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
        initcodedir = r"D:\Generator_DB_Windows\stimuli\texture006"  # Code & image folder to initialize the Genetic Algorithm
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        recorddir = r"E:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  ## Home_WorkStation Victoria
        recorddir = r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\Generator_DB_Windows\data\with_CNN"
# Basic properties for Optimizer.

init_sigma = 3
Aupdate_freq = 10
code_length = 4096
# from config import GANname, code_length
# print(GANname, " codelength %d"%code_length)
# generator = load_GAN(GANname)
# utils.generator = generator
# render = generator.render
# def render(codes, scale=255):
#     '''Render a list of codes to list of images'''
#     if type(codes) is list:
#         images = [generator.visualize(codes[i], scale) for i in range(len(codes))]
#     else:
#         images = [generator.visualize(codes[i, :], scale) for i in range(codes.shape[0])]
#     return images
#%% Simplified in silico experiment modules
class CNNmodel:
    """ Basic CNN scorer
    Demo:
        CNN = CNNmodel('caffe-net')  #
        CNN.select_unit( ('caffe-net', 'conv1', 5, 10, 10) )
        scores = CNN.score(imgs)
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv1' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name):
        self._classifier = net_utils.load(model_name)
        self._transformer = net_utils.get_transformer(self._classifier, scale=1)
        self.artiphys = False

    def select_unit(self, unit_tuple):
        self._classifier_name = str(unit_tuple[0])
        self._net_layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(unit_tuple[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(unit_tuple) == 5:
            self._net_unit_x = int(unit_tuple[3])
            self._net_unit_y = int(unit_tuple[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layername in record_layers:  # will be arranged in a dict of lists
            self.recordings[layername] = []

    # def forward(self, imgs):
    #     return recordings

    def score(self, images):
        scores = np.zeros(len(images))
        for i, img in enumerate(images):
            # Note: now only support single repetition
            tim = self._transformer.preprocess('data', img)  # shape=(3, 227, 227) # assuming input scale is 0,1 output will be 0,255
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)  # propagate the image the target layer
            # record only the neuron intended
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]
            if self._net_unit_x is not None:
                # if `self._net_unit_x/y` (inside dimension) are provided, then use them to slice the output score
                score = score[self._net_unit_x, self._net_unit_y]
            scores[i] = score
            if self.artiphys:  # record the whole layer's activation
                for layername in self.record_layers:
                    score_full = self._classifier.blobs[layername].data[0, :]
                    # self._pattern_array.append(score_full)
                    self.recordings[layername].append(score_full.copy())
        if self.artiphys:
            return scores, self.recordings
        else:
            return scores
#%% A simple torch models! supporting caffe-net
from torch_net_utils import load_caffenet, visualize, preprocess
class CNNmodel_Torch:
    """ Basic CNN scorer
    Demo:
        CNN = CNNmodel('caffe-net')  #
        CNN.select_unit( ('caffe-net', 'conv1', 5, 10, 10) )
        scores = CNN.score(imgs)
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv1' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name):
        if model_name == "caffe-net":
            self._classifier = load_caffenet()
            self.preprocess = preprocess # finally implemented and tested!
        # self._transformer = net_utils.get_transformer(self._classifier, scale=1)
        self.artiphys = False

    # def preprocess(self, img):

    def select_unit(self, unit_tuple):
        self._classifier_name = str(unit_tuple[0])
        self._net_layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(unit_tuple[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(unit_tuple) == 5:
            self._net_unit_x = int(unit_tuple[3])
            self._net_unit_y = int(unit_tuple[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layername in record_layers:  # will be arranged in a dict of lists
            self.recordings[layername] = []

    # def forward(self, imgs):
    #     return recordings

    def score(self, images, with_grad=False):
        scores = np.zeros(len(images))
        for i, img in enumerate(images):
            # Note: now only support single repetition
            resz_out_img = self.preprocess(img, input_scale=255)  # shape=(3, 227, 227) # assuming input scale is 0,1 output will be 0,255
            blobs_CNN = self._classifier(resz_out_img)
            if self._net_unit_x is not None:
                score = blobs_CNN[self._net_layer][0, self._net_iunit, self._net_unit_x, self._net_unit_y]
            else:
                score = blobs_CNN[self._net_layer][0, self._net_iunit]
            scores[i] = score
            if self.artiphys:  # record the whole layer's activation
                for layername in self.record_layers:
                    score_full = blobs_CNN[layername][0, :]
                    # self._pattern_array.append(score_full)
                    self.recordings[layername].append(score_full.copy())
        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

#%% More general torch models!
import torch
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from torch_net_utils import layername_dict
from GAN_utils import upconvGAN
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

activation = {} # global variable is important for hook to work! it's an important channel for communication
def get_activation(name, unit=None):
    if unit is None:
        def hook(model, input, output):
            activation[name] = output.detach()
    else:
        def hook(model, input, output):
            if len(output.shape) == 4:
                activation[name] = output.detach()[:, unit[0], unit[1], unit[2]]
            elif len(output.shape) == 2:
                activation[name] = output.detach()[:, unit[0]]
    return hook

class TorchScorer:
    """ Torch CNN Scorer using hooks to fetch score from any layer in the net. Allows all models in torchvision zoo
    Demo:
        scorer = TorchScorer("vgg16")
        scorer.select_unit(("vgg16", "fc2", 10, 10, 10))
        scorer.score([np.random.rand(224, 224, 3), np.random.rand(227,227,3)])
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv2' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name):
        if model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
            self.layers = list(self.model.features) + list(self.model.classifier)
            self.layername = layername_dict[model_name]
            self.model.cuda().eval()
        elif model_name == "alexnet":
            self.model = models.alexnet(pretrained=True)
            self.layers = list(self.model.features) + list(self.model.classifier)
            self.layername = layername_dict[model_name]
            self.model.cuda().eval()
        elif model_name == "densenet121":
            self.model = models.densenet121(pretrained=True)
            self.layers = list(self.model.features) + [self.model.classifier]
            self.layername = layername_dict[model_name]
            self.model.cuda().eval()
        # self.preprocess = transforms.Compose([transforms.ToPILImage(),
        #                                       transforms.Resize(size=(224, 224)),
        #                                       transforms.ToTensor(),
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) # Imagenet normalization RGB
        self.RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).cuda()
        self.RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).cuda()
        self.artiphys = False

    def preprocess(self, img, input_scale=255):
        """preprocess single image array or a list (minibatch) of images"""
        # could be modified to support batch processing. Added batch @ July. 10, 2020
        # test and optimize the performance by permute the operators. Use CUDA acceleration from preprocessing
        if type(img) is list: # the following lines have been optimized for speed locally.
            img_tsr = torch.stack(tuple(torch.from_numpy(im) for im in img)).cuda().float().permute(0, 3, 1, 2) / input_scale
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        else:  # assume it's individual image
            img_tsr = transforms.ToTensor()(img / input_scale).float()
            img_tsr = self.normalize(img_tsr).unsqueeze(0)
            resz_out_img = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_img

    def set_unit(self, name, layer, unit=None):
        idx = self.layername.index(layer)
        handle = self.layers[idx].register_forward_hook(get_activation(name, unit))
        return handle

    def select_unit(self, unit_tuple):
        # self._classifier_name = str(unit_tuple[0])
        self.layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self.chan = int(unit_tuple[2])
        if len(unit_tuple) == 5:
            self.unit_x = int(unit_tuple[3])
            self.unit_y = int(unit_tuple[4])
        else:
            self.unit_x = None
            self.unit_y = None
        self.set_unit("score", self.layer, unit=(self.chan, self.unit_x, self.unit_y))

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layer in record_layers:  # will be arranged in a dict of lists
            self.set_unit(layer, layer, unit=None)
            self.recordings[layer] = []

    def score(self, images, with_grad=False, B=42):
        """Score in batch will accelerate processing greatly! """ # assume image is using 255 range
        scores = np.zeros(len(images))
        csr = 0  # if really want efficiency, we should use minibatch processing.
        imgn = len(images)
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(images[csr:csr_end], input_scale=255.0)
            # img_batch.append(resz_out_img)
            with torch.no_grad():
                # self.model(torch.cat(img_batch).cuda())
                self.model(img_batch)
            scores[csr:csr_end] = activation["score"].squeeze().cpu().numpy().squeeze()
            csr = csr_end
            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = activation[layer]
                    # self._pattern_array.append(score_full)
                    self.recordings[layer].append(score_full.cpu().numpy())
            # , input_scale=255 # shape=(3, 227, 227) # assuming input scale is 0,1 output will be 0,255

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores
#%%
# Compiled Experimental module!
# Currently available
# - Evolution
# - resize and evolution
# - evolution in a restricted linear subspace
# - tuning among major axis of the GAN model and rotated O(N) axis for GAN model

class ExperimentEvolve:
    """ Basic Evolution Experiments
    Default behavior is to use the current CMAES optimizer to optimize for 200 steps for the given unit.
    support Caffe or Torch Backend

    the render function should have such signature, input numpy array of B-by-code_length, output list of images.
        it also has a named parameter scale=255.0. which specify the range of pixel value of output.
    """
    def __init__(self, model_unit, max_step=200, backend="caffe", optimizer=None, GAN="fc6", verbose=False):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        if backend == "caffe":
            self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        elif backend == "torch":
            if model_unit[0] is 'caffe-net':
                self.CNNmodel = CNNmodel_Torch(model_unit[0])
            else: # alexnet, VGG, DENSE and anything else
                self.CNNmodel = TorchScorer(model_unit[0])
        else:
            raise NotImplementedError
        self.CNNmodel.select_unit(model_unit)
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = upconvGAN(name=GAN).cuda()
            self.render = self.G.render  # function that map a 2d array of code (samp_n by code len) to a list of images
            # self.G = Generator(name=GAN)
            # self.render = self.G.render
            if GAN == "fc8":
                code_length = 1000
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            code_length = 256  # 128
            # 128d Class Embedding code or 256d full code could be used.
        elif GAN == "BigBiGAN":
            from BigBiGAN import BigBiGAN_render
            self.render = BigBiGAN_render
            code_length = 120  # 120 d space for Unconditional generation in BigBiGAN
        else:
            raise NotImplementedError
        if optimizer is not None:  # Default optimizer is this
            self.optimizer = optimizer
        else:
            self.optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=init_sigma,
                init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq, maximize=True, random_seed=None,
                                           optim_params={})
            # CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
            #                                init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq)
            # assert issubclass(type(optimizer), Optimizer)
        self.max_steps = max_step
        self.verbose = verbose
        self.code_length = code_length

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        t00 = time()
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.random.randn(20, self.code_length)
                    # codes = np.zeros([1, code_length])
                    if type(self.optimizer) is Genetic:
                        # self.optimizer.load_init_population(initcodedir, )
                        codes, self.optimizer._genealogy = utils.load_codes2(initcodedir, self.optimizer._popsize)
                else:
                    codes = init_code
            print('>>> step %d' % self.istep)
            t0 = time()
            self.current_images = self.render(codes, scale=255.0)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            if self.verbose:
                print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), -np.sort(-synscores)))
            else:
                print("img scores: mean %.2f max %.2f min %.2f"%(np.nanmean(synscores), np.nanmax(synscores),
                                                                 np.nanmin(synscores)))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)
        t11 = time()
        print("Summary\nGenerations: %d, Image samples: %d, Best score: %.2f (spent %.2f sec)" % (self.istep, self.codes_all.shape[0], self.scores_all.max(), t11 - t00))
        
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
        img_select = self.render(select_code, scale=1.0)
        fig = utils.visualize_img_list(img_select, score_select, show=show, nrow=None, title_str=title_str)
        if show:
            fig.show()
        return fig

    def visualize_best(self, show=False, title_str=""):
        """ Just Visualize the best Images for the experiment
        """
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx + 1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code, scale=1.0)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select) + title_str, fontsize=14)
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

#%
# from ZO_HessAware_Optimizers import HessAware_Gauss_DC
class ExperimentEvolve_DC:
    """
    Default behavior is to use the current CMAES optimizer to optimize for 200 steps for the given unit.
    This Experimental Class is defined to test out the new Optimizers equipped with Descent Checking
    """
    def __init__(self, model_unit, max_step=200, optimizer=None, backend="caffe", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        if backend is "caffe":
            self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        elif backend is "torch":
            self.CNNmodel = CNNmodel_Torch(model_unit[0])
        else:
            raise NotImplementedError
        self.CNNmodel.select_unit(model_unit)
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = Generator(name=GAN)
            self.render = self.G.render
            if GAN == "fc8":
                self.code_length = 1000
            else:
                self.code_length = 4096
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128
            # 128d Class Embedding code or 256d full code could be used.
        else:
            raise NotImplementedError
        if optimizer is None:  # Default optimizer is this
            self.optimizer = HessAware_Gauss_DC(space_dimen=self.code_length, )    # , optim_params=optim_params
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
        x_img = self.render(init_code)
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
            new_imgs = self.render(new_codes)
            new_scores = self.CNNmodel.score(new_imgs)
            y_code = self.optimizer.compute_grad(new_scores)
            y_img = self.render(y_code)
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
                    Hess_imgs = self.render(Hess_codes)
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
        img_select = self.render(select_code, scale=1.0)
        fig = utils.visualize_img_list(img_select, score_select, show=show, nrow=None, title_str=title_str)
        if show:
            fig.show()
        return fig

    def visualize_best(self, show=False, title_str=""):
        """ Just Visualize the best Images for the experiment """
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx + 1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code, scale=1.0)
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
from cv2 import resize
import cv2

def resize_and_pad(img_list, size, offset, canvas_size=(227, 227)):
    '''Resize and Pad a list of images to list of images
    Note this function is assuming the image is in (0,1) scale so padding with 0.5 as gray background.
    '''
    resize_img = []
    padded_shape = canvas_size + (3,)
    for img in img_list:
        if img.shape == padded_shape:  # save some computation...
            resize_img.append(img.copy())
        else:
            pad_img = np.ones(padded_shape) * 127.5
            pad_img[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], :] = resize(img, size, cv2.INTER_AREA)
            resize_img.append(pad_img.copy())
    return resize_img

class ExperimentResizeEvolve:
    """Resize the evolved image before feeding into CNN and see how the evolution goes. """
    def __init__(self, model_unit, imgsize=(227, 227), corner=(0, 0),
                 max_step=200, savedir="", explabel="", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = Generator(name=GAN)
            self.render = self.G.render
            if GAN == "fc8":
                self.code_length = 1000
            else:
                self.code_length = 4096
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128 # 128d Class Embedding code or 256d full code could be used.
        else:
            raise NotImplementedError
        self.optimizer = CholeskyCMAES(self.code_length, population_size=None, init_sigma=init_sigma,
                init_code=np.zeros([1, self.code_length]), Aupdate_freq=Aupdate_freq, maximize=True, random_seed=None,
                                           optim_params={})
        # CholeskyCMAES(recorddir=recorddir, space_dimen=self.code_length, init_sigma=init_sigma,
        #                            init_code=np.zeros([1, self.code_length]),
        #                            Aupdate_freq=Aupdate_freq)  # , optim_params=optim_params
        self.max_steps = max_step
        self.corner = corner  # up left corner of the image
        self.imgsize = imgsize  # size of image
        self.savedir = savedir
        self.explabel = explabel

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, self.code_length])
                else:
                    codes = init_code
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = self.render(codes)  # note visualize to 0,1 scale
            self.current_images = resize_and_pad(self.current_images, self.imgsize, self.corner)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def visualize_exp(self, show=False):
        idx_list = []
        for geni in range(min(self.generations), max(self.generations)+1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = self.render(select_code, scale=1)
        fig = utils.visualize_img_list(img_select, score_select, show=show)
        fig.savefig(join(self.savedir, "Evolv_Img_Traj_%s.png" % (self.explabel)))
        return fig

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code)#, scale=1
        fig = plt.figure(figsize=[3, 3])
        plt.subplot(1,2,1)
        plt.imshow(img_select[0]/255)
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        plt.subplot(1, 2, 2)
        resize_select = resize_and_pad(img_select, self.imgsize, self.corner)
        plt.imshow(resize_select[0]/255)
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        fig.savefig(join(self.savedir, "Best_Img_%s.png" % (self.explabel)))
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice).astype("float64")
        MaxScore = np.zeros_like(gen_slice).astype("float64")
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        figh.savefig(join(self.savedir, "Evolv_Traj_%s.png" % (self.explabel)))
        return figh

class ExperimentManifold:
    def __init__(self, model_unit, max_step=100, imgsize=(227, 227), corner=(0, 0),
                 savedir="", explabel="", backend="caffe", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.pref_unit = model_unit
        if backend == "caffe":
            self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        elif backend == "torch":
            if model_unit[0] == 'caffe-net': # `is` won't work here!
                self.CNNmodel = CNNmodel_Torch(model_unit[0])
            else:  # VGG, DENSE and anything else
                self.CNNmodel = TorchScorer(model_unit[0])
        else:
            raise NotImplementedError
        self.CNNmodel.select_unit(model_unit)
        # Allow them to choose from multiple optimizers, substitute generator.visualize and render
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = Generator(name=GAN)
            self.render = self.G.render
            if GAN == "fc8":
                self.code_length = 1000
            else:
                self.code_length = 4096
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128 # 128d Class Embedding code or 256d full code could be used.
        else:
            raise NotImplementedError
        self.optimizer = CholeskyCMAES(self.code_length, population_size=None, init_sigma=init_sigma,
                                       init_code=np.zeros([1, self.code_length]), Aupdate_freq=Aupdate_freq,
                                       maximize=True, random_seed=None,
                                       optim_params={})
        # self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=self.code_length, init_sigma=init_sigma,
        #                                init_code=np.zeros([1, self.code_length]),
        #                                Aupdate_freq=Aupdate_freq)  # , optim_params=optim_params
        self.max_steps = max_step
        self.corner = corner  # up left corner of the image
        self.imgsize = imgsize  # size of image, allowing showing CNN resized image
        self.savedir = savedir
        self.explabel = explabel
        self.Perturb_vec = []

    def run(self, init_code=None):
        """Same as Resized Evolution experiment"""
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, self.code_length])
                else:
                    codes = init_code
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = self.render(codes)
            t1 = time()  # generate image from code
            self.current_images = resize_and_pad(self.current_images, self.imgsize, self.corner)  # Fixed Apr.13
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def load_traj(self, filename):
        data = np.load(join(self.savedir, filename))
        self.codes_all = data["codes_all"]
        self.scores_all = data["scores_all"]
        self.generations = data["generations"]

    def analyze_traj(self):
        '''Get the trajectory and the PCs and the structures of it'''
        final_gen_norms = np.linalg.norm(self.codes_all[self.generations == max(self.generations), :], axis=1)
        self.sphere_norm = final_gen_norms.mean()
        code_pca = PCA(n_components=50)
        PC_Proj_codes = code_pca.fit_transform(self.codes_all)
        self.PC_vectors = code_pca.components_
        if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
            # this is important or the images we show will land in the opposite side of the globe.
            inv_PC1 = True
            self.PC_vectors[0, :] = - self.PC_vectors[0, :]
            self.PC1_sign = -1
        else:
            inv_PC1 = False
            self.PC1_sign = 1
            pass

    def run_manifold(self, subspace_list, interval=9):
        '''Generate examples on manifold and run'''
        self.score_sum = []
        figsum = plt.figure(figsize=[16.7, 4])
        for spi, subspace in enumerate(subspace_list):
            if subspace == "RND":
                title = "Norm%dRND%dRND%d" % (self.sphere_norm, 0 + 1, 1 + 1)
                print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d)" % self.sphere_norm)
                rand_vec2 = np.random.randn(2, self.code_length)
                rand_vec2 = rand_vec2 - (rand_vec2 @ self.PC_vectors.T) @ self.PC_vectors
                rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
                rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
                rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
                vectors = np.concatenate((self.PC_vectors[0:1, :], rand_vec2), axis=0)
                self.Perturb_vec.append(vectors)
                img_list = []
                interv_n = int(90 / interval)
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ vectors
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        img = self.G.visualize(code_vec)
                        img_list.append(img.copy())
            else:
                PCi, PCj = subspace
                title = "Norm%dPC%dPC%d" % (self.sphere_norm, PCi + 1, PCj + 1)
                print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (PCi + 1, PCj + 1, self.sphere_norm, ))
                img_list = []
                interv_n = int(90 / interval)
                self.Perturb_vec.append(self.PC_vectors[[0, PCi, PCj], :])
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ self.PC_vectors[[0, PCi, PCj], :]
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        img = self.G.visualize(code_vec)
                        img_list.append(img.copy())
                        # plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (
                        # self.sphere_norm, interval * j, interval * k)), img)
            pad_img_list = resize_and_pad(img_list, self.imgsize, self.corner) # Show image as given size at given location
            scores = self.CNNmodel.score(pad_img_list)
            # fig = utils.visualize_img_list(img_list, scores=scores, ncol=2*interv_n+1, nrow=2*interv_n+1, )
            # subsample images for better visualization
            msk, idx_lin = subsample_mask(factor=2, orig_size=(21, 21))
            img_subsp_list = [img_list[i] for i in range(len(img_list)) if i in idx_lin]
            fig = utils.visualize_img_list(img_subsp_list, scores=scores[idx_lin], ncol=interv_n + 1, nrow=interv_n + 1, )
            fig.savefig(join(self.savedir, "%s_%s.png" % (title, self.explabel)))
            scores = np.array(scores).reshape((2*interv_n+1, 2*interv_n+1))
            self.score_sum.append(scores)
            ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
            im = ax.imshow(scores)
            plt.colorbar(im, ax=ax)
            ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_xticklabels([-90,45,0,45,90])
            ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_yticklabels([-90,45,0,45,90])
            ax.set_title(title+"_Hemisphere")
        figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
        figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.png" % (self.explabel, self.sphere_norm)))
        self.Perturb_vec = np.concatenate(tuple(self.Perturb_vec), axis=0)
        return self.score_sum, figsum

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code)#, scale=1
        fig = plt.figure(figsize=[3, 1.7])
        plt.subplot(1, 2, 1)
        plt.imshow(img_select[0]/255)
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        plt.subplot(1, 2, 2)
        resize_select = resize_and_pad(img_select, self.imgsize, self.corner)
        plt.imshow(resize_select[0]/255)
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        fig.savefig(join(self.savedir, "Best_Img_%s.png" % (self.explabel)))
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice).astype("float64")
        MaxScore = np.zeros_like(gen_slice).astype("float64")
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        figh.savefig(join(self.savedir, "Evolv_Traj_%s.png" % (self.explabel)))
        return figh
#%%
def subsample_mask(factor=2, orig_size=(21, 21)):
    """Generate a mask for subsampling grid of `orig_size`"""
    row, col = orig_size
    row_sub = slice(0, row, factor)  # range or list will not work! Don't try!
    col_sub = slice(0, col, factor)
    msk = np.zeros(orig_size, dtype=np.bool)
    msk[row_sub, :][:, col_sub] = True
    msk_lin = msk.flatten()
    idx_lin = msk_lin.nonzero()[0]
    return msk, idx_lin

#%%
import math
def make_orthonormal_matrix(n):
    """
    Makes a square matrix which is orthonormal by concatenating
    random Householder transformations
    Note: May not distribute uniformly in the O(n) manifold.
    Note: Naively using  ortho_group, special_ortho_group  in scipy will result in unbearable computing time! Not useful
    """
    A = np.identity(n)
    d = np.zeros(n)
    d[n-1] = np.random.choice([-1.0, 1.0])
    for k in range(n-2, -1, -1):
        # generate random Householder transformation
        x = np.random.randn(n-k)
        s = np.sqrt((x**2).sum()) # norm(x)
        sign = math.copysign(1.0, x[0])
        s *= sign
        d[k] = -sign
        x[0] += s
        beta = s * x[0]
        # apply the transformation
        y = np.dot(x,A[k:n,:]) / beta
        A[k:n, :] -= np.outer(x,y)
    # change sign of rows
    A *= d.reshape(n,1)
    return A

class ExperimentGANAxis:
    """ Tuning w.r.t. all the major axis in the GAN or the randomly generated O(n) frame set. """
    def __init__(self, model_unit, savedir="", explabel="", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.scores_all_rnd = []
        self.codes_all = []
        self.pref_unit = model_unit
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        self.savedir = savedir
        self.explabel = explabel
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = Generator(name=GAN)
            self.render = self.G.render
            if GAN == "fc8":
                self.code_length = 1000
            else:
                self.code_length = 4096
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128
        else:
            raise NotImplementedError

    def run_axis(self, Norm, orthomat=None):
        '''Generate examples on manifold and run'''
        self.score_sum = []
        figsum = plt.figure(figsize=[16.7, 8])

        BATCH_SIZE = 128
        BATCH_N = int(self.code_length / BATCH_SIZE)
        print("Test the tuning on all the axis in GAN space (Norm %d)"%Norm)
        code_mat = np.eye(self.code_length, self.code_length)
        scores_all = []
        scores_all_neg = []
        for bi in range(BATCH_N):
            img_list = []
            for j in range(BATCH_SIZE):
                img = self.G.visualize(Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all.extend(list(scores))
            img_list = []
            for j in range(BATCH_SIZE):
                img = self.G.visualize(- Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all_neg.extend(list(scores))
            print("Finished batch %02d/%02d"%( bi+1, BATCH_N))
        self.scores_all = np.array(scores_all + scores_all_neg)
        ax = figsum.add_subplot(2, 1, 1)
        ax.scatter(np.arange(self.code_length), scores_all, alpha=0.5)
        ax.scatter(np.arange(self.code_length), scores_all_neg, alpha=0.4)
        ax.plot(sorted(scores_all), color='orange')
        ax.plot(sorted(scores_all_neg), color='green')
        ax.set_xlim(-50, self.code_length)
        if orthomat is None:
            code_mat = make_orthonormal_matrix(self.code_length)# ortho_group.rvs(4096)
        else:
            code_mat = orthomat
        scores_all = []
        scores_all_neg = []
        print("Test the tuning on a random O(N) in GAN space (Norm %d)" % Norm)
        for bi in range(BATCH_N):
            img_list = []
            for j in range(BATCH_SIZE):
                img = self.G.visualize(Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all.extend(list(scores))
            img_list = []
            for j in range(BATCH_SIZE):
                img = self.G.visualize(- Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all_neg.extend(list(scores))
            print("Finished batch %02d/%02d"% (bi + 1, BATCH_N))
        self.scores_all_rnd = np.array(scores_all + scores_all_neg)
        ax = figsum.add_subplot(2, 1, 2)
        ax.scatter(np.arange(self.code_length), scores_all, alpha=0.5)
        ax.plot(sorted(scores_all), color='orange')
        ax.scatter(np.arange(self.code_length), scores_all_neg, alpha=0.4)
        ax.plot(sorted(scores_all_neg), color='green')
        ax.set_xlim(-50, self.code_length)
        # ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
        # im = ax.imshow(scores)
        # plt.colorbar(im, ax=ax)
        # ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_xticklabels([-90,45,0,45,90])
        # ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_yticklabels([-90,45,0,45,90])
        # ax.set_title(title+"_Hemisphere")
        # figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
        figsum.savefig(os.path.join(self.savedir, "Axis_summary_%s_norm%d.png" % (self.explabel, Norm)))
        return self.scores_all, self.scores_all_rnd, figsum
#%
class ExperimentRestrictEvolve:
    """Evolution in a restricted linear subspace with subspace_d """
    def __init__(self, subspace_d, model_unit, max_step=200, GAN="fc6"):
        self.sub_d = subspace_d
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)  # ('caffe-net', 'fc8', 1)
        self.optimizer = CholeskyCMAES(subspace_d, population_size=None, init_sigma=init_sigma,
                                       init_code=np.zeros([1, subspace_d]), Aupdate_freq=Aupdate_freq,
                                       maximize=True, random_seed=None,
                                       optim_params={})
        # self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=subspace_d, init_sigma=init_sigma,
        #                                init_code=np.zeros([1, subspace_d]),
        #                                Aupdate_freq=Aupdate_freq)  # , optim_params=optim_params
        self.max_steps = max_step
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            self.G = Generator(name=GAN)
            self.render = self.G.render
            if GAN == "fc8":
                self.code_length = 1000
            else:
                self.code_length = 4096
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128
            # 128d Class Embedding code or 256d full code could be used.
        else:
            raise NotImplementedError

    def get_basis(self): # TODO substitute this with np.linalg.qr
        self.basis = np.zeros([self.sub_d, self.code_length])
        for i in range(self.sub_d):
            tmp_code = np.random.randn(1, self.code_length)
            tmp_code = tmp_code - (tmp_code @ self.basis.T) @ self.basis
            self.basis[i, :] = tmp_code / np.linalg.norm(tmp_code)
        return self.basis

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.coords_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    coords = np.zeros([1, self.sub_d])
                else:
                    coords = init_code
            codes = coords @ self.basis
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = self.render(codes)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            coords_new = self.optimizer.step_simple(synscores, coords)
            t3 = time()  # use results to update optimizer
            self.coords_all.append(coords)
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            coords = coords_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.coords_all = np.concatenate(tuple(self.coords_all), axis=0)
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def visualize_exp(self, show=False):
        idx_list = []
        for geni in range(min(self.generations), max(self.generations)+1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = self.render(select_code)
        fig = utils.visualize_img_list(img_select, score_select, show=show)
        return fig

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
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
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        return figh

#%%
if __name__ == "__main__":
    # experiment = ExperimentEvolve()
    # experiment.run()
    #%%
    # subspace_d = 50
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, ('caffe-net', 'fc8', 1))
    #     experiment.get_basis()
    #     experiment.run()
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(os.path.join(recorddir, "Subspc%dScoreTrajTrial%03d" % (subspace_d, triali) + ".png"))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(os.path.join(recorddir, "Subspc%dEvolveTrial%03d"%(subspace_d, triali) + ".png"))
    # #%%
    # #%% Restricted evolution for the 5 examplar layerse
    # subspace_d = 50
    # unit = ('caffe-net', 'conv5', 5, 10, 10)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     plt.close("all")
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    # #%%
    # subspace_d = 50
    # unit = ('caffe-net', 'conv3', 5, 10, 10)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(0, 100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     plt.close("all")
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #
    # subspace_d = 50
    # unit = ('caffe-net', 'fc6', 1)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #
    # subspace_d = 50
    # unit = ('caffe-net', 'fc7', 1)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)

    #%% Baseline Full Evolution
    unit_arr = [('caffe-net', 'conv3', 5, 10, 10),
                ('caffe-net', 'conv5', 5, 10, 10),
                ('caffe-net', 'fc6', 1),
                ('caffe-net', 'fc7', 1),
                ('caffe-net', 'fc8', 1)]
    # unit = ('caffe-net', 'fc7', 1)
    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(20):
    #         experiment = ExperimentEvolve(unit, max_step=200)
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "FullBestImgTrial%03d.png" % (triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "FullScoreTrajTrial%03d.png" % (triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "EvolveTrial%03d.png" % (triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_trial%03d.npz" % (triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)

    #%%
    unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv3', 5, 10, 10),
                #('caffe-net', 'conv1', 5, 10, 10),
                ]
    subspace_d = 20
                #  [('caffe-net', 'conv3', 5, 10, 10),
                # ('caffe-net', 'conv5', 5, 10, 10),
                # ('caffe-net', 'fc6', 1),
                # ('caffe-net', 'fc7', 1),
                # ('caffe-net', 'fc8', 1)]
    # unit = ('caffe-net', 'fc7', 1)
    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(20):
    #         experiment = ExperimentEvolve(unit, max_step=200)
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "FullBestImgTrial%03d.png" % (triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "FullScoreTrajTrial%03d.png" % (triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "EvolveTrial%03d.png" % (triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_trial%03d.npz" % (triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)

    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_subspac%d" % (unit[0], unit[1], unit[2], subspace_d))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(100):
    #         experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #         experiment.get_basis()
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #%%

    unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv4', 5, 10, 10),]
                # ('caffe-net', 'conv3', 5, 10, 10),
                # ('caffe-net', 'conv5', 5, 10, 10),
                # ('caffe-net', 'fc6', 1),
                # ('caffe-net', 'fc7', 1),
                # ('caffe-net', 'fc8', 1),
                # ]
    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=100, savedir=savedir, explabel="chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")

    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold_25gen" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=25, savedir=savedir, explabel="step25_chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_step25_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")

    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold_50gen" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=50, savedir=savedir, explabel="step50_chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_step50_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")
#%%
    omat = np.load("ortho4096.npy")
    savedir = join(recorddir, "axis_data")
    unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv3', 5, 10, 10),
                ('caffe-net', 'conv4', 5, 10, 10),
                ('caffe-net', 'conv5', 5, 10, 10),
                ('caffe-net', 'fc6', 1),
                ('caffe-net', 'fc7', 1),
                ('caffe-net', 'fc8', 1),
                ]
    for unit in unit_arr:
        exp = ExperimentGANAxis(unit, savedir=savedir,
                                explabel="%s_%d" % (unit[1],unit[2]))
        exp.run_axis(350, orthomat=omat)
        np.savez(join(savedir, "axis_score_%s_%d" % (unit[1],unit[2])), scores_all=exp.scores_all, scores_all_rnd=exp.scores_all_rnd)


    #%%
    savedir = join(recorddir, "resize_data")
    os.makedirs(savedir, exist_ok=True)
    unit_arr = [
                ('caffe-net', 'conv5', 5, 10, 10),
                ('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv3', 5, 10, 10),
                ('caffe-net', 'conv4', 5, 10, 10),
                ('caffe-net', 'fc6', 1),
                ('caffe-net', 'fc7', 1),
                ('caffe-net', 'fc8', 1),
                ]
    for unit in unit_arr:
        exp = ExperimentResizeEvolve(unit, )
                                #explabel="%s_%d" % (unit[1],unit[2]))
        exp.run()
        exp.visualize_best()
        exp.visualize_trajectory()
        exp.visualize_exp()