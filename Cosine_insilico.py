

# fetch population activity

# objective
from os.path import join
from time import time
from skimage.io import imread
import matplotlib.pylab as plt
import numpy as np
import torch
from insilico_Exp_torch import ExperimentManifold, resize_and_pad_tsr
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
from insilico_Exp_torch import TorchScorer
from GAN_utils import upconvGAN
# GAN = "fc6"
# G = upconvGAN(name=GAN).cuda()
# render_tsr = G.visualize_batch_np  # this output tensor
# render = G.render

# code_length = 4096
# optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=init_sigma,
#                init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq,
#                 maximize=True, random_seed=None, optim_params={})
# #%%
# scorer = TorchScorer("alexnet")
init_sigma = 3.0
Aupdate_freq = 10
class ExperimentEvolPopulation:
    def __init__(self, scorer, max_step=100, imgsize=(227, 227), corner=(0, 0),
                 savedir="", explabel="", backend="torch", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        # self.pref_unit = model_unit
        self.backend = backend
        # if backend == "caffe":
        #     self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        # elif backend == "torch":
        #     if model_unit[0] == 'caffe-net': # `is` won't work here!
        #         self.CNNmodel = CNNmodel_Torch(model_unit[0])
        #     else:  # AlexNet, VGG, ResNet, DENSE and anything else
        #         self.CNNmodel = TorchScorer(model_unit[0])
        # else:
        #     raise NotImplementedError
        # self.CNNmodel.select_unit(model_unit)
        self.CNNmodel = scorer
        # Allow them to choose from multiple optimizers, substitute generator.visualize and render
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            from GAN_utils import upconvGAN
            self.G = upconvGAN(name=GAN).cuda()
            self.render_tsr = self.G.visualize_batch_np  # this output tensor
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
            self.current_images = self.render_tsr(codes)
            t1 = time()  # generate image from code
            self.current_images = resize_and_pad_tsr(self.current_images, self.imgsize, self.corner)
            synscores = self.CNNmodel.score_tsr(self.current_images)
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

    def visualize_best(self, show=False, title_str=""):
        """ Just Visualize the best Images for the experiment """
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

activation = {} # global variable is important for hook to work! it's an important channel for communication
def get_activation(name, unit=None):
    if unit is None:
        def hook(model, input, output):
            activation[name] = output.detach()
    elif type(unit) == np.ndarray:
        def hook(model, input, output):
            if len(output.shape) == 4:
                activation[name] = output.detach().reshape(output.shape[0], -1)[:, unit]
            elif len(output.shape) == 2:
                activation[name] = output.detach()[:, unit]
    else:
        def hook(model, input, output):
            if len(output.shape) == 4:
                activation[name] = output.detach()[:, unit[0], unit[1], unit[2]]
            elif len(output.shape) == 2:
                activation[name] = output.detach()[:, unit[0]]
    return hook

class TorchScorer_Pop(TorchScorer):
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
        super().__init__(model_name)

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

    def select_rand_population(self, layer, N=1000):
        self.layer = layer
        randidx = np.random.randint(1E5, size=N)
        randidx = np.sort(randidx)
        self.set_unit("score", self.layer, unit=randidx)

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layer in record_layers:  # will be arranged in a dict of lists
            self.set_unit(layer, layer, unit=None)
            self.recordings[layer] = []

    def encode_target(self, image, input_scale=255):
        img_batch = self.preprocess(image, input_scale=input_scale)
        with torch.no_grad():
            # self.model(torch.cat(img_batch).cuda())
            self.model(img_batch)
        poprepr = activation["score"].squeeze().cpu().numpy()
        self.targrepr = poprepr.reshape(1, -1)

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

    def score_tsr(self, img_tsr, with_grad=False, B=42):
        """Score in batch will accelerate processing greatly! """
        # assume image is using 255 range
        scores = np.zeros(img_tsr.shape[0])
        csr = 0  # if really want efficiency, we should use minibatch processing.
        imgn = img_tsr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=1.0)
            # img_batch.append(resz_out_img)
            with torch.no_grad():
                # self.model(torch.cat(img_batch).cuda())
                self.model(img_batch)
            poprepr = activation["score"].squeeze().cpu().numpy()
            scores[csr:csr_end] = - np.mean((poprepr - self.targrepr)**2, axis=1)# activation["score"].squeeze().cpu().numpy().squeeze()
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
img = imread("RefCollection\\08-cat-cancer-sq.jpeg")
scorer = TorchScorer_Pop("vgg16")
scorer.select_rand_population("conv12", 1000) # relu seems to help?
scorer.encode_target([img])
Exp = ExperimentEvolPopulation(scorer, 100)
Exp.run()
Exp.visualize_best(True)
Exp.visualize_trajectory(True)
torch.cuda.empty_cache()
#%%
plt.hist(activation["score"].cpu().numpy().flatten(),30,density=True,alpha=0.4)
plt.hist(scorer.targrepr.flatten(),30,density=True,alpha=0.4)
plt.show()
#%%
# the target population is not the more the better....