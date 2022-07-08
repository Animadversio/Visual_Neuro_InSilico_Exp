import os
from os.path import join
from sys import platform
from time import time, sleep
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from GAN_utils import upconvGAN
from layer_hook_utils import layername_dict, register_hook_by_module_names, get_module_names, named_apply
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
from utils_old import visualize_img_list
# mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

activation = {}  # global variable is important for hook to work! it's an important channel for communication
def get_activation(name, unit=None, unitmask=None, ingraph=False):
    """Return a hook that record the unit activity into the entry in activation dict."""
    if unit is None and unitmask is None:  # if no unit is given, output the full tensor.
        def hook(model, input, output):
            activation[name] = output if ingraph else output.detach()

    elif unitmask is not None: # has a unit mask, which could be an index list or a tensor mask same shape of the 3 dimensions.
        def hook(model, input, output):
            out = output if ingraph else output.detach()
            Bsize = out.shape[0]
            activation[name] = out.view([Bsize, -1])[:, unitmask.reshape(-1)]

    else:
        def hook(model, input, output):
            out = output if ingraph else output.detach()
            if len(output.shape) == 4:
                activation[name] = out[:, unit[0], unit[1], unit[2]]
            elif len(output.shape) == 2:
                activation[name] = out[:, unit[0]]

    return hook


if platform == "linux": # cluster
    # torchhome = "/scratch/binxu/torch/checkpoints"  # CHPC
    scratchdir = os.environ["SCRATCH1"]
    torchhome = join(scratchdir, "torch/checkpoints")  # CHPC
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        torchhome = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        torchhome = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  ## Home_WorkStation Victoria
        torchhome = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] =='PONCELAB-OFF6':
        torchhome = r"C:\Users\ponce\Documents\torch"


class TorchScorer:
    """ Pure PyTorch CNN Scorer using hooks to fetch score from any layer in the net.
    Compatible with all models in torchvision zoo
    Demo:
        scorer = TorchScorer("vgg16")
        scorer.select_unit(("vgg16", "fc2", 10, 10, 10))
        scorer.score([np.random.rand(224, 224, 3), np.random.rand(227,227,3)])
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv2' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name, imgpix=227, rawlayername=True):
        self.imgpix = imgpix
        if isinstance(model_name, torch.nn.Module):
            self.model = model_name
            self.inputsize = (3, imgpix, imgpix)
            self.layername = None
        elif isinstance(model_name, str):
            if model_name == "vgg16":
                self.model = models.vgg16(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                # self.layername = layername_dict[model_name]
                self.layername = None if rawlayername else layername_dict["vgg16"]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "vgg16-face":
                self.model = models.vgg16(pretrained=False, num_classes=2622)
                self.model.load_state_dict(torch.load(join(torchhome, "vgg16_face.pt")))
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else layername_dict["vgg16"]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "alexnet":
                self.model = models.alexnet(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else layername_dict[model_name]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "densenet121":
                self.model = models.densenet121(pretrained=True)
                self.layers = list(self.model.features) + [self.model.classifier]
                self.layername = None if rawlayername else layername_dict[model_name]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "densenet169":
                self.model = models.densenet169(pretrained=True)
                self.layername = None
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "resnet101":
                self.model = models.resnet101(pretrained=True)
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            elif "resnet50" in model_name:
                if "resnet50-face" in model_name:  # resnet trained on vgg-face dataset.
                    self.model = models.resnet50(pretrained=False, num_classes=8631)
                    if model_name == "resnet50-face_ft":
                        self.model.load_state_dict(torch.load(join(torchhome, "resnet50_ft_weight.pt")))
                    elif model_name == "resnet50-face_scratch":
                        self.model.load_state_dict(torch.load(join(torchhome, "resnet50_scratch_weight.pt")))
                    else:
                        raise NotImplementedError("Feasible names are resnet50-face_scratch, resnet50-face_ft")
                else:
                    self.model = models.resnet50(pretrained=True)
                    if model_name in ["resnet50_linf_8", "resnet50_linf8"]:  # robust version of resnet50.
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_linf_8_pure.pt")))
                    elif model_name == "resnet50_linf_4":
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_linf_4_pure.pt")))
                    elif model_name == "resnet50_l2_3_0":
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_l2_3_0_pure.pt")))
                    else:
                        print("use the default resnet50 weights")
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            elif model_name == "cornet_s":
                from cornet import cornet_s
                Cnet = cornet_s(pretrained=True)
                self.model = Cnet.module
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            else:
                raise NotImplementedError("Cannot find the specified model %s"%model_name)
        else:
            raise NotImplementedError("model_name need to be either string or nn.Module")

        self.model.cuda().eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        # self.preprocess = transforms.Compose([transforms.ToPILImage(),
        #                                       transforms.Resize(size=(224, 224)),
        #                                       transforms.ToTensor(),
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])  # Imagenet normalization RGB
        self.RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).cuda()
        self.RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).cuda()
        self.hooks = []
        self.artiphys = False
        self.record_layers = []
        self.recordings = {}

        self.activation = {}

    def get_activation(self, name, unit=None, unitmask=None, ingraph=False):
        """
        :parameter
            name: key to retrieve the recorded activation `self.activation[name]`
            unit: a tuple of 3 element or single element. (chan, i, j ) or (chan)
            unitmask: used in population recording, it could be a binary mask of the same shape / length as the
                element number in feature tensor. Or it can be an array of integers.

            *Note*: when unit and unitmask are both None, this function will record the whole output feature tensor.

            ingraph: if True, then the recorded activation is still connected to input, so can pass grad.
                    if False then cannot
        :return
            hook:  Return a hook function that record the unit activity into the entry in activation dict of scorer.
        """
        if unit is None and unitmask is None:  # if no unit is given, output the full tensor. 
            def hook(model, input, output): 
                self.activation[name] = output if ingraph else output.detach()

        elif unitmask is not None:
            # has a unit mask, which could be an index list or a tensor mask same shape of the 3 dimensions.
            def hook(model, input, output): 
                out = output if ingraph else output.detach()
                Bsize = out.shape[0]
                self.activation[name] = out.view([Bsize, -1])[:, unitmask.reshape(-1)]

        else:
            def hook(model, input, output): 
                out = output if ingraph else output.detach()
                if len(output.shape) == 4: 
                    self.activation[name] = out[:, unit[0], unit[1], unit[2]]
                elif len(output.shape) == 2: 
                    self.activation[name] = out[:, unit[0]]

        return hook

    def set_unit(self, reckey, layer, unit=None, ingraph=False):
        if self.layername is not None:
            # if the network is a single stream feedforward structure, we can index it and use it to find the
            # activation
            idx = self.layername.index(layer)
            handle = self.layers[idx].register_forward_hook(self.get_activation(reckey, unit, ingraph=ingraph)) # we can get the layer by indexing
            self.hooks.append(handle)  # save the hooks in case we will remove it.
        else:
            # if not, we need to parse the architecture of the network.
            # indexing is not available, we need to register by recursively visit the layers and find match.
            handle, modulelist, moduletype = register_hook_by_module_names(layer, self.get_activation(reckey, unit, ingraph=ingraph),
                                self.model, self.inputsize, device="cuda")
            self.hooks.extend(handle)  # handle here is a list.
        return handle

    def set_units_by_mask(self, reckey, layer, unit_mask=None):
        if self.layername is not None:
            # if the network is a single stream feedforward structure, we can index it and use it to find the
            # activation
            idx = self.layername.index(layer)
            handle = self.layers[idx].register_forward_hook(self.get_activation(reckey, unitmask=unit_mask))
            # we can get the layer by indexing
            self.hooks.append(handle)  # save the hooks in case we will remove it.
        else:
            # if not, we need to parse the architecture of the network indexing is not available. 
            # we need to register by recursively visit the layers and find match.
            handle, modulelist, moduletype = register_hook_by_module_names(layer, 
                self.get_activation(reckey, unitmask=unit_mask), self.model, self.inputsize, device="cuda")
            self.hooks.extend(handle)  # handle here is a list.
        return handle
    
    def select_unit(self, unit_tuple, allow_grad=False):
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
        self.set_unit("score", self.layer, unit=(self.chan, self.unit_x, self.unit_y), ingraph=allow_grad)

    def set_recording(self, record_layers, allow_grad=False):
        """The function to select a scalar output from a NN"""
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers.extend(record_layers)
        for layer in record_layers:  # will be arranged in a dict of lists
            self.set_unit(layer, layer, unit=None, ingraph=allow_grad)
            self.recordings[layer] = []

    def set_popul_recording(self, record_layer, mask):
        self.artiphys = True
        self.record_layers.append(record_layer)
        h = self.set_units_by_mask(record_layer, record_layer, unit_mask=mask)
        self.recordings[record_layer] = []

    def preprocess(self, img, input_scale=255):
        """preprocess single image array or a list (minibatch) of images
        This includes Normalize using RGB mean and std and resize image to (227, 227)
        """
        # could be modified to support batch processing. Added batch @ July. 10, 2020
        # test and optimize the performance by permute the operators. Use CUDA acceleration from preprocessing
        if type(img) is list: # the following lines have been optimized for speed locally.
            img_tsr = torch.stack(tuple(torch.from_numpy(im) for im in img)).cuda().float().permute(0, 3, 1, 2) / input_scale
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is torch.Tensor:
            img_tsr = (img.cuda() / input_scale - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim == 4:
            img_tsr = torch.tensor(img / input_scale).float().permute(0,3,1,2).cuda()
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim in [2, 3]:  # assume it's individual image
            img_tsr = transforms.ToTensor()(img / input_scale).float()
            img_tsr = self.normalize(img_tsr).unsqueeze(0)
            resz_out_img = F.interpolate(img_tsr, (227, 227), mode='bilinear',
                                         align_corners=True)
            return resz_out_img
        else:
            raise ValueError

    def score(self, images, with_grad=False, B=42, input_scale=1.0):
        """Score in batch will accelerate processing greatly! """ # assume image is using 255 range
        scores = np.zeros(len(images))
        csr = 0  # if really want efficiency, we should use minibatch processing.
        imgn = len(images)
        for layer in self.recordings: 
            self.recordings[layer] = []
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(images[csr:csr_end], input_scale=input_scale)
            # img_batch.append(resz_out_img)
            with torch.no_grad():
                # self.model(torch.cat(img_batch).cuda())
                self.model(img_batch)
            if "score" in self.activation: # if score is not there set trace to zero. 
                scores[csr:csr_end] = self.activation["score"].squeeze().cpu().numpy().squeeze()

            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer] # temporory storage
                    self.recordings[layer].append(score_full.cpu().numpy()) # formulated storage
            
            csr = csr_end
        
        for layer in self.recordings: 
            self.recordings[layer] = np.concatenate(self.recordings[layer],axis=0)

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

    def score_tsr(self, img_tsr, with_grad=False, B=42, input_scale=1.0):
        """Score in batch will accelerate processing greatly!
        img_tsr is already torch.Tensor
        """
        # assume image is using 255 range
        imgn = img_tsr.shape[0]
        scores = np.zeros(img_tsr.shape[0])
        for layer in self.recordings: 
            self.recordings[layer] = []

        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=input_scale)
            with torch.no_grad():
                self.model(img_batch.cuda())
            if "score" in self.activation: # if score is not there set trace to zero.
                scores[csr:csr_end] = self.activation["score"].squeeze().cpu().numpy().squeeze()

            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer]
                    self.recordings[layer].append(score_full.cpu().numpy())

            csr = csr_end
        for layer in self.recordings:
            self.recordings[layer] = np.concatenate(self.recordings[layer],axis=0)

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

    def score_tsr_wgrad(self, img_tsr, B=10, input_scale=1.0):
        for layer in self.recordings:
            self.recordings[layer] = []
        imgn = img_tsr.shape[0]
        scores = torch.zeros(img_tsr.shape[0]).cuda()
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=input_scale)
            self.model(img_batch)
            if "score" in self.activation:  # if score is not there set trace to zero.
                scores[csr:csr_end] += self.activation["score"].squeeze()
            if self.artiphys:  # record the whole neurlayer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer]
                    # self._pattern_array.append(score_full)
                    self.recordings[layer].append(score_full) # .cpu().numpy()

            csr = csr_end

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

    def cleanup(self):
        print("Cleanuping...")
        for hook in self.hooks:
            hook.remove()
            self.hooks.remove(hook)
        self.hooks = []
        print("Cleanup hooks done.")


def visualize_trajectory(scores_all, generations, codes_arr=None, show=False, title_str=""):
    """ Visualize the Score Trajectory """
    gen_slice = np.arange(min(generations), max(generations) + 1)
    AvgScore = np.zeros(gen_slice.shape)
    MaxScore = np.zeros(gen_slice.shape)  # Bug fixed Oct. 14th, before there will be dtype casting problem
    for i, geni in enumerate(gen_slice):
        AvgScore[i] = np.mean(scores_all[generations == geni])
        MaxScore[i] = np.max(scores_all[generations == geni])
    figh, ax1 = plt.subplots()
    ax1.scatter(generations, scores_all, s=16, alpha=0.6, label="all score")
    ax1.plot(gen_slice, AvgScore, color='black', label="Average score")
    ax1.plot(gen_slice, MaxScore, color='red', label="Max score")
    ax1.set_xlabel("generation #")
    ax1.set_ylabel("CNN unit score")
    plt.legend()
    if codes_arr is not None:
        ax2 = ax1.twinx()
        if codes_arr.shape[1] == 256: # BigGAN
            nos_norm = np.linalg.norm(codes_arr[:, :128], axis=1)
            cls_norm = np.linalg.norm(codes_arr[:, 128:], axis=1)
            ax2.scatter(generations, nos_norm, s=5, color="orange", label="noise", alpha=0.2)
            ax2.scatter(generations, cls_norm, s=5, color="magenta", label="class", alpha=0.2)
        elif codes_arr.shape[1] == 4096: # FC6GAN
            norms_all = np.linalg.norm(codes_arr[:, :], axis=1)
            ax2.scatter(generations, norms_all, s=5, color="magenta", label="all", alpha=0.2)
        ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
        plt.legend()
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    if show:
        plt.show()
    return figh



# if platform == "linux":  # CHPC cluster
#     ckpt_dir = "/scratch/binxu.wang/torch"
# else:
#     if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
#         ckpt_dir = r"E:\Cluster_Backup\torch"
#     elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
#         ckpt_dir = r"E:\Cluster_Backup\torch"
#     # elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
#     #     ckpt_dir = r"E:\Cluster_Backup\torch"
#     # elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
#     #     ckpt_dir = r"E:\Cluster_Backup\torch"
#     elif os.environ['COMPUTERNAME'] == 'LAPTOP-U8TSR4RE':
#         ckpt_dir = r"D:\torch\checkpoints"
#     elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B': # Alfa rig monkey computer
#         ckpt_dir = r"C:\Users\Ponce lab\Documents\Python\torch_nets"
#     elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A': # Alfa rig monkey computer
#         ckpt_dir = r"C:\Users\Poncelab-ML2a\Documents\Python\torch_nets"
#     else:
#         ckpt_dir = r"C:\Users\ponce\Documents\torch"

def load_featnet(netname: str):
    """API to load common CNNs and their convolutional part.
    Default behvavior: load onto GPU, and set parameter gradient to false.
    """
    if netname == "alexnet":
        net = models.alexnet(True)
        net.requires_grad_(False)
        featnet = net.features.cuda().eval()
    elif netname == "vgg16":
        net = models.vgg16(pretrained=True)
        net.requires_grad_(False)
        featnet = net.features.cuda().eval()
    elif netname == "resnet50":
        net = models.resnet50(pretrained=True)
        net.requires_grad_(False)
        featnet = net.cuda().eval()
    elif netname == "resnet50_linf8":
        net = models.resnet50(pretrained=True)
        net.load_state_dict(torch.load(join(torchhome, "imagenet_linf_8_pure.pt")))
        net.requires_grad_(False)
        featnet = net.cuda().eval()
    else:
        raise NotImplementedError
    return featnet, net


init_sigma = 3
Aupdate_freq = 10
# from cv2 import resize
# import cv2
from skimage.transform import rescale, resize
def resize_and_pad(img_list, size, offset, canvas_size=(227, 227), scale=1.0):
    '''Resize and Pad a list of images to list of images
    Note this function is assuming the image is in (0,1) scale so padding with 0.5 as gray background.
    '''
    resize_img = []
    padded_shape = canvas_size + (3,)
    for img in img_list:
        if img.shape == padded_shape:  # save some computation...
            resize_img.append(img.copy())
        else:
            pad_img = np.ones(padded_shape) * 0.5 * scale
            pad_img[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], :] = resize(img, size, )#cv2.INTER_AREA)
            resize_img.append(pad_img.copy())
    return resize_img


def resize_and_pad_tsr(img_tsr, size, offset, canvas_size=(227, 227), scale=1.0):
    '''Resize and Pad a tensor of images to tensor of images (torch tensor)
    Note this function is assuming the image is in (0,1) scale so padding with 0.5 as gray background.
    '''
    assert img_tsr.ndim in [3, 4]
    if img_tsr.ndim == 3:
        img_tsr.unsqueeze_(0)
    imgn = img_tsr.shape[0]

    padded_shape = (imgn, 3,) + canvas_size
    pad_img = torch.ones(padded_shape) * 0.5 * scale
    pad_img.to(img_tsr.dtype)
    rsz_tsr = F.interpolate(img_tsr, size=size)
    pad_img[:, :, offset[0]:offset[0] + size[0], offset[1]:offset[1] + size[1]] = rsz_tsr
    return pad_img


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


class ExperimentManifold:
    def __init__(self, model_unit, max_step=100, imgsize=(227, 227), corner=(0, 0), optimizer=None,
                 savedir="", explabel="", backend="torch", GAN="fc6"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.pref_unit = model_unit
        self.backend = backend
        if backend == "caffe":
            from insilico_Exp import CNNmodel # really old version
            self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        elif backend == "torch":
            if model_unit[0] == 'caffe-net': # `is` won't work here!
                from insilico_Exp import CNNmodel_Torch
                self.CNNmodel = CNNmodel_Torch(model_unit[0])  # really old version
            else:  # AlexNet, VGG, ResNet, DENSE and anything else
                self.CNNmodel = TorchScorer(model_unit[0])
        else:
            raise NotImplementedError
        self.CNNmodel.select_unit(model_unit)
        # Allow them to choose from multiple optimizers, substitute generator.visualize and render
        if GAN == "fc6" or GAN == "fc7" or GAN == "fc8":
            from GAN_utils import upconvGAN
            self.G = upconvGAN(name=GAN).cuda()
            self.render_tsr = self.G.visualize_batch_np  # this output tensor
            self.render = self.G.render
            # self.G = Generator(name=GAN)
            # self.render = self.G.render
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
        if optimizer is None:
            self.optimizer = CholeskyCMAES(self.code_length, population_size=None, init_sigma=init_sigma,
                                       init_code=np.zeros([1, self.code_length]), Aupdate_freq=Aupdate_freq,
                                       maximize=True, random_seed=None, optim_params={})
        else:
            self.optimizer = optimizer

        self.max_steps = max_step
        self.corner = corner  # up left corner of the image
        self.imgsize = imgsize  # size of image, allowing showing CNN resized image
        self.savedir = savedir
        self.explabel = explabel
        self.Perturb_vec = []

    def re_init(self):
        self.optimizer = CholeskyCMAES(self.code_length, population_size=None, init_sigma=init_sigma,
                                       init_code=np.zeros([1, self.code_length]), Aupdate_freq=Aupdate_freq,
                                       maximize=True, random_seed=None, optim_params={})
        self.Perturb_vec = []
        self.CNNmodel.cleanup()

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
            # if self.backend == "caffe":
            #     self.current_images = self.render(codes)
            #     t1 = time()  # generate image from code
            #     self.current_images = resize_and_pad(self.current_images, self.imgsize, self.corner)  # Fixed Apr.13
            #     synscores = self.CNNmodel.score(self.current_images)
            #     t2 = time()  # score images
            # elif self.backend == "torch":
            self.current_images = self.render_tsr(codes)
            t1 = time()  # generate image from code
            self.current_images = resize_and_pad_tsr(self.current_images, self.imgsize, self.corner)
            # Fixed Jan.14 2021
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

    def save_last_gen(self, filename=""):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx + 1, :]
        lastgen_code = np.mean(self.codes_all[self.generations == max(self.generations), :], axis=0, keepdims=True)
        lastgen_score = np.mean(self.scores_all[self.generations == max(self.generations)], )
        np.savez(join(self.savedir, "Evolution_codes_%s.npz" % (self.explabel)),
                 best_code=select_code, best_score=self.scores_all[idx],
                 lastgen_codes=lastgen_code, lastgen_score=lastgen_score)
        print("Last generation and Best code saved.")

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

    def render_manifold(self, subspace_list, interval=18, ):
        '''Generate images on manifold '''
        T0 = time()
        figsum = plt.figure(figsize=[16.7, 4])
        img_tsr_list = []
        for spi, subspace in enumerate(subspace_list):
            code_list = []
            if subspace == "RND":
                title = "Norm%dRND%dRND%d" % (self.sphere_norm, 0 + 1, 1 + 1)
                print(
                    "Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d) " % self.sphere_norm)
                rand_vec2 = np.random.randn(2, self.code_length)
                rand_vec2 = rand_vec2 - (rand_vec2 @ self.PC_vectors.T) @ self.PC_vectors
                rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
                rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
                rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
                vectors = np.concatenate((self.PC_vectors[0:1, :], rand_vec2), axis=0)
                # self.Perturb_vec.append(vectors)
                interv_n = int(90 / interval)
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ vectors
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        code_list.append(code_vec)
            else:
                PCi, PCj = subspace
                title = "Norm%dPC%dPC%d" % (self.sphere_norm, PCi + 1, PCj + 1)
                print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (
                PCi + 1, PCj + 1, self.sphere_norm,))
                interv_n = int(90 / interval)
                # self.Perturb_vec.append(self.PC_vectors[[0, PCi, PCj], :])
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ self.PC_vectors[[0, PCi, PCj], :]
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        code_list.append(code_vec)
            print("Latent vectors ready, rendering. (%.3f sec passed)" % (time() - T0))
            code_arr = np.array(code_list)
            img_tsr = self.render_tsr(code_arr)
            img_tsr_list.append(img_tsr)
        return img_tsr_list

    def run_manifold(self, subspace_list, interval=9, print_manifold=True):
        '''Generate examples on manifold and run'''
        self.score_sum = []
        self.Perturb_vec = []
        T0 = time()
        figsum = plt.figure(figsize=[16.7, 4])
        for spi, subspace in enumerate(subspace_list):
            code_list = []
            if subspace == "RND":
                title = "Norm%dRND%dRND%d" % (self.sphere_norm, 0 + 1, 1 + 1)
                print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d) " % self.sphere_norm)
                rand_vec2 = np.random.randn(2, self.code_length)
                rand_vec2 = rand_vec2 - (rand_vec2 @ self.PC_vectors.T) @ self.PC_vectors
                rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
                rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
                rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
                vectors = np.concatenate((self.PC_vectors[0:1, :], rand_vec2), axis=0)
                self.Perturb_vec.append(vectors)
                # img_list = []
                interv_n = int(90 / interval)
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ vectors
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        code_list.append(code_vec)
                        # img = self.G.visualize(code_vec)
                        # img_list.append(img.copy())
            else:
                PCi, PCj = subspace
                title = "Norm%dPC%dPC%d" % (self.sphere_norm, PCi + 1, PCj + 1)
                print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (PCi + 1, PCj + 1, self.sphere_norm, ))
                # img_list = []
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
                        code_list.append(code_vec)
                        # img = self.G.visualize(code_vec)
                        # img_list.append(img.copy())
                        # plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (
                        # self.sphere_norm, interval * j, interval * k)), img)

            # pad_img_list = resize_and_pad(img_list, self.imgsize, self.corner) # Show image as given size at given location
            # scores = self.CNNmodel.score(pad_img_list)
            print("Latent vectors ready, rendering. (%.3f sec passed)"%(time()-T0))
            code_arr = np.array(code_list)
            img_tsr = self.render_tsr(code_arr)
            pad_img_tsr = resize_and_pad_tsr(img_tsr, self.imgsize, self.corner)  # Show image as given size at given location
            scores = self.CNNmodel.score_tsr(pad_img_tsr)
            img_arr = img_tsr.permute([0,2,3,1])
            print("Image and score ready! Figure printing (%.3f sec passed)"%(time()-T0))
            # fig = utils.visualize_img_list(img_list, scores=scores, ncol=2*interv_n+1, nrow=2*interv_n+1, )
            # subsample images for better visualization
            msk, idx_lin = subsample_mask(factor=2, orig_size=(21, 21))
            img_subsp_list = [img_arr[i] for i in range(len(img_arr)) if i in idx_lin]
            if print_manifold:
                fig = visualize_img_list(img_subsp_list, scores=scores[idx_lin], ncol=interv_n + 1, nrow=interv_n + 1, )
                fig.savefig(join(self.savedir, "%s_%s.png" % (title, self.explabel)))
                plt.close(fig)
            scores = np.array(scores).reshape((2*interv_n+1, 2*interv_n+1)) # Reshape score as heatmap.
            self.score_sum.append(scores)
            ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
            im = ax.imshow(scores)
            plt.colorbar(im, ax=ax)
            ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_xticklabels([-90,45,0,45,90])
            ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_yticklabels([-90,45,0,45,90])
            ax.set_title(title+"_Hemisphere")
        figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
        figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.png" % (self.explabel, self.sphere_norm)))
        # figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.pdf" % (self.explabel, self.sphere_norm)))
        self.Perturb_vec = np.concatenate(tuple(self.Perturb_vec), axis=0)
        np.save(join(self.savedir, "Manifold_score_%s" % (self.explabel)), self.score_sum)
        np.savez(join(self.savedir, "Manifold_set_%s.npz" % (self.explabel)),
                 Perturb_vec=self.Perturb_vec, imgsize=self.imgsize, corner=self.corner,
                 evol_score=self.scores_all, evol_gen=self.generations, sphere_norm=self.sphere_norm)
        return self.score_sum, figsum

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code, scale=1.0) #, scale=1
        fig = plt.figure(figsize=[3, 1.7])
        plt.subplot(1, 2, 1)
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        plt.subplot(1, 2, 2)
        resize_select = resize_and_pad(img_select, self.imgsize, self.corner, scale=1.0)
        plt.imshow(resize_select[0])
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


if __name__=="__main__":
    Exp = ExperimentManifold(("resnet101", ".layer3.Bottleneck22", 10, 7, 7), max_step=50, imgsize=(150, 150), corner=(30, 30),
        savedir="", explabel="", backend="torch", GAN="fc6")
    Exp.run()
    Exp.visualize_best(True)
    Exp.visualize_trajectory(True)
    Exp.analyze_traj()
    Exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
