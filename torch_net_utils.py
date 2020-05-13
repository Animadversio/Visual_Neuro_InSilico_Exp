''' Utilities to streamline the import of Caffenet in torch '''
import os
from os.path import join
import torch
import sys
import numpy as np
#%%
# Depend on 2 packages, you should clone from
# https://github.com/Animadversio/pytorch-receptive-field
# https://github.com/Animadversio/pytorch-caffe.git
from sys import platform
if platform == "linux":  # CHPC cluster
    homedir = os.path.expanduser('~')
    netsdir = os.path.join(homedir, 'Generate_DB/nets')
    sys.path.append("/home/binxu/pytorch-caffe")
    sys.path.append("/home/binxu/pytorch-receptive-field")
    sys.path.append("/home/binxu/PerceptualSimilarity") # should be added there!
    # ckpt_path = {"vgg16": "/scratch/binxu/torch/vgg16-397923af.pth"}
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        sys.path.append(r"D:\Github\pytorch-caffe")
        sys.path.append(r"D:\Github\pytorch-receptive-field")
        sys.path.append(r"D:\Github\PerceptualSimilarity")
        homedir = "D:/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
        sys.path.append(r"C:\Users\ponce\Documents\GitHub\pytorch-caffe")
        sys.path.append(r"C:\Users\ponce\Documents\GitHub\pytorch-receptive-field")
        homedir = r"C:\Users\ponce\Documents\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        sys.path.append(r"E:\Github_Projects\pytorch-caffe")
        sys.path.append(r"E:\Github_Projects\pytorch-receptive-field")
        sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
        homedir = "E:/Monkey_Data/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
        sys.path.append(r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\GitHub\pytorch-caffe")
        sys.path.append(r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\GitHub\pytorch-receptive-field")
        homedir = "C:/Users/zhanq/OneDrive - Washington University in St. Louis/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    else:
        sys.path.append("D:\Github\pytorch-caffe")
        homedir = os.path.expanduser('~')
        netsdir = os.path.join(homedir, 'Documents/nets')
from caffenet import *  # Pytorch-caffe converter
from torch_receptive_field import receptive_field, receptive_field_for_unit
#%% Prepare PyTorch version of the Caffe networks
def load_caffenet():
    protofile = join(netsdir, r"caffenet\caffenet.prototxt")  # 'resnet50/deploy.prototxt'
    weightfile = join(netsdir, r'caffenet\bvlc_reference_caffenet.caffemodel')  # 'resnet50/resnet50.caffemodel'
    save_path = join(netsdir, r"caffenet\caffenet_state_dict.pt")
    net = CaffeNet(protofile)
    print(net)
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        net.load_weights(weightfile)
        torch.save(net.state_dict(), save_path)
    net.eval()
    net.verbose = False
    net.requires_grad_(requires_grad=False)
    for param in net.parameters():
        param.requires_grad = False
    return net

def GAN_path(name):
    if name == "fc6":
        save_path = os.path.join(netsdir, r"upconv/fc6/generator_state_dict.pt")
        protofile = os.path.join(netsdir, r"upconv/fc6/generator.prototxt")  # 'resnet50/deploy.prototxt'
        weightfile = os.path.join(netsdir, r'upconv/fc6/generator.caffemodel')  # 'resnet50/resnet50.caffemodel'
    elif name == "fc7":
        save_path = os.path.join(netsdir, r"upconv/fc7/generator_state_dict.pt")
        protofile = os.path.join(netsdir, r"upconv/fc7/generator.prototxt")  # 'resnet50/deploy.prototxt'
        weightfile = os.path.join(netsdir, r'upconv/fc7/generator.caffemodel')  # 'resnet50/resnet50.caffemodel'
    elif name == "fc8":
        save_path = os.path.join(netsdir, r"upconv/fc8/generator_state_dict.pt")
        protofile = os.path.join(netsdir, r"upconv/fc8/generator.prototxt")  # 'resnet50/deploy.prototxt'
        weightfile = os.path.join(netsdir, r'upconv/fc8/generator.caffemodel')  # 'resnet50/resnet50.caffemodel'
    else:
        raise ValueError(name + 'not defined')
    return save_path, protofile, weightfile

def load_generator(GAN="fc6"):
    # netsdir = r"D:/Generator_DB_Windows/nets"
    save_path, protofile, weightfile = GAN_path(GAN)
    Generator = CaffeNet(protofile)
    print(Generator)
    if os.path.exists(save_path):
        Generator.load_state_dict(torch.load(save_path))
    else:
        Generator.load_weights(weightfile)
        torch.save(Generator.state_dict(), save_path) # Generator.
    Generator.eval()
    Generator.verbose = False
    Generator.requires_grad_(requires_grad=False)
    for param in Generator.parameters():
        param.requires_grad = False
    return Generator

#%%
BGR_mean = torch.tensor([104.0, 117.0, 123.0])
BGR_mean = torch.reshape(BGR_mean, (1, 3, 1, 1))
#%%
def visualize(G, code, mode="cuda"):
    """Do the De-caffe transform (Validated)
    works for a single code """
    code = code.reshape(-1, 4096).astype(np.float32)
    if mode == "cpu":
        blobs = G(torch.from_numpy(code))
    else:
        blobs = G(torch.from_numpy(code).cuda())
    out_img = blobs['deconv0']  # get raw output image from GAN
    if mode == "cpu":
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    else:
        clamp_out_img = torch.clamp(out_img + BGR_mean.cuda(), 0, 255)
    vis_img = clamp_out_img[:, [2, 1, 0], :, :].permute([2, 3, 1, 0]).squeeze() / 255
    if mode == "cpu":
        return vis_img
    if mode == "cuda":
        return vis_img.cpu()
#%%
import torch.nn.functional as F
def preprocess(img, input_scale=255):
    """"""
    # TODO, could be modified to support batch processing
    img_tsr = torch.from_numpy(img).float()  # assume img is aready at 255 uint8 scale.
    if input_scale != 255:
        img_tsr = img_tsr / input_scale * 255
    img_tsr = img_tsr.unsqueeze(0).permute([0, 3, 1, 2])
    resz_out_img = F.interpolate(img_tsr[:, [2, 1, 0], :, :] - BGR_mean, (227, 227), mode='bilinear',
                                 align_corners=True)
    return resz_out_img
# import net_utils
# detfmr = net_utils.get_detransformer(net_utils.load('generator'))
# tfmr = net_utils.get_transformer(net_utils.load('caffe-net'))
#%%
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
def get_layer_names(model):
    layername = []
    conv_cnt = 0
    fc_cnt = 0
    pool_cnt = 0
    do_cnt = 0
    for layer in list(model.features)+list(model.classifier):
        if isinstance(layer, nn.Conv2d):
            conv_cnt += 1
            layername.append("conv%d" % conv_cnt)
        elif isinstance(layer, nn.ReLU):
            name = layername[-1] + "_relu"
            layername.append(name)
        elif isinstance(layer, nn.MaxPool2d):
            pool_cnt += 1
            layername.append("pool%d"%pool_cnt)
        elif isinstance(layer, nn.Linear):
            fc_cnt += 1
            layername.append("fc%d" % fc_cnt)
        elif isinstance(layer, nn.Dropout):
            do_cnt += 1
            layername.append("dropout%d" % do_cnt)
        else:
            layername.append(layer.__repr__())
    return layername
#%% Utility code to fetch activation
# def get_activation(name, unit=None):
#     if unit is None:
#         def hook(model, input, output):
#             activation[name] = output.detach()
#     else:
#         def hook(model, input, output):
#             if len(output.shape) == 4:
#                 activation[name] = output.detach()[:, unit[0], unit[1], unit[2]]
#             elif len(output.shape) == 2:
#                 activation[name] = output.detach()[:, unit[0]]
#     return hook
#
# def set_unit(model, name, layer, unit=None):
#     idx = layername.index(layer)
#     layers = list(model.features) + list(model.classifier)
#     handle = layers[idx].register_forward_hook(get_activation(name, unit))
#     return handle
#
# layers = list(classifier.features) + list(classifier.classifier)
# layername = get_layer_names(classifier)
# set_unit(classifier, "score", "conv2", unit=(0,10,10))
# classifier = models.vgg16(pretrained=True)
# classifier.eval()
# score_hk = set_unit(classifier, "score", "fc1", unit=(None, 10, 10))
# img = torch.rand((2, 3, 224, 224))
# out = classifier(img)
# print(activation["score"])
# activation["score"].shape
#%%
layername_dict ={"vgg16":['conv1', 'conv1_relu',
                         'conv2', 'conv2_relu', 'pool1',
                         'conv3', 'conv3_relu',
                         'conv4', 'conv4_relu', 'pool2',
                         'conv5', 'conv5_relu',
                         'conv6', 'conv6_relu',
                         'conv7', 'conv7_relu', 'pool3',
                         'conv8', 'conv8_relu',
                         'conv9', 'conv9_relu',
                         'conv10', 'conv10_relu', 'pool4',
                         'conv11', 'conv11_relu',
                         'conv12', 'conv12_relu',
                         'conv13', 'conv13_relu', 'pool5',
                         'fc1', 'fc1_relu', 'dropout1',
                         'fc2', 'fc2_relu', 'dropout2',
                         'fc3'],
                 "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1']}

#%%  Processing script to get the layer name array from a torch model
# layers = list(densenet.features)+[densenet.classifier]
# layername = []
# conv_cnt = 0
# fc_cnt = 0
# pool_cnt = 0
# do_cnt = 0
# for layer in layers:
#     if isinstance(layer, nn.Conv2d):
#         conv_cnt += 1
#         layername.append("conv%d" % conv_cnt)
#     elif isinstance(layer, nn.ReLU):
#         name = layername[-1] + "_relu"
#         layername.append(name)
#     elif isinstance(layer, nn.MaxPool2d):
#         pool_cnt += 1
#         layername.append("pool%d"%pool_cnt)
#     elif isinstance(layer, nn.Linear):
#         fc_cnt += 1
#         layername.append("fc%d" % fc_cnt)
#     elif isinstance(layer, nn.Dropout):
#         do_cnt += 1
#         layername.append("dropout%d" % do_cnt)
#     else:
#         layername.append(layer.__repr__())