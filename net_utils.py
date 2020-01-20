import os
import numpy as np
import caffe
# Note this is the only code zone that depend on caffe.
# So if another High level platform of CNN is to be used, only change this file

caffe.set_mode_gpu()
from sys import platform
if platform == "linux":  # CHPC cluster
    homedir = os.path.expanduser('~')
    netsdir = os.path.join(homedir, 'Generate_DB/nets')
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        homedir = "D:/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        homedir = "D:/Monkey_Data/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    else:
        homedir = os.path.expanduser('~')
        netsdir = os.path.join(homedir, 'Documents/nets')

if not os.path.isdir(netsdir):
    raise OSError('path for nets is invalid: %s' % netsdir)

ilsvrc2012_mean = np.array((104.0, 117.0, 123.0))  # ImageNet Mean in BGR order
loaded_nets = {}


def get(net_name):
    '''Get the definition and weights for different NNs. From downloaded Caffenet File in `netsdir` '''
    if net_name == 'caffe-net':
        net_weights = os.path.join(netsdir, 'caffenet', 'bvlc_reference_caffenet.caffemodel')
        net_definition = os.path.join(netsdir, 'caffenet', 'caffenet.prototxt')
    elif net_name == 'places-CNN':
        net_weights = os.path.join(netsdir, 'placesCNN', 'places205CNN_iter_300000.caffemodel')
        net_definition = os.path.join(netsdir, 'placesCNN', 'places205CNN_deploy_updated.prototxt')
    elif net_name == 'google-net':
        net_weights = os.path.join(netsdir, 'googlenet', 'bvlc_googlenet.caffemodel')
        net_definition = os.path.join(netsdir, 'googlenet', 'bvlc_googlenet_updated.prototxt')
    elif net_name == 'resnet-50':
        net_weights = os.path.join(netsdir, 'resnet-50', 'ResNet-50-model.caffemodel')
        net_definition = os.path.join(netsdir, 'resnet-50', 'ResNet-50-deploy.prototxt')
    elif net_name == 'resnet-101':
        net_weights = os.path.join(netsdir, 'resnet-101', 'ResNet-101-model.caffemodel')
        net_definition = os.path.join(netsdir, 'resnet-101', 'ResNet-101-deploy.prototxt')
    elif net_name == 'resnet-152':
        net_weights = os.path.join(netsdir, 'resnet-152', 'ResNet-152-model.caffemodel')
        net_definition = os.path.join(netsdir, 'resnet-152', 'ResNet-152-deploy.prototxt')
    elif net_name == 'generator':
        net_weights = os.path.join(netsdir, 'upconv', 'fc6', 'generator.caffemodel')
        net_definition = os.path.join(netsdir, 'upconv', 'fc6', 'generator.prototxt')
    else:
        raise ValueError(net_name + 'not defined')

    return net_definition, net_weights


def load(net_name):
    ''' Load nets by name using `get` routine. Store the nets in dict `loaded_nets` for repeating usage.
    :param net_name:
    :return: net
    '''
    try:
        return loaded_nets[net_name]    # do not load the same net multiple times
    except KeyError:
        net_def, net_weights = get(net_name)
        if os.name == 'nt':
            net = caffe.Net(net_def, caffe.TEST)
            net.copy_from(net_weights)
        else: # UBUNTU systems' name is 'posix'
            net = caffe.Net(net_def, caffe.TEST, weights=net_weights)
        loaded_nets[net_name] = net
        return net


def get_detransformer(generator):
    '''Inverse-Transforming the input generator dimensions, formats, etc...'''
    detransformer = caffe.io.Transformer({'data': generator.blobs['deconv0'].data.shape})
    detransformer.set_transpose('data', (2, 0, 1))             # move color channels to outermost dimension
    detransformer.set_mean('data', ilsvrc2012_mean)            # subtract the dataset-mean value in each channel
    detransformer.set_raw_scale('data', 255)                   # rescale from [0, 1] to [0, 255]
    detransformer.set_channel_swap('data', (2, 1, 0))          # swap channels from RGB to BGR
    return detransformer


def get_transformer(classifier, scale=255):
    transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))             # move color channels to outermost dimension
    transformer.set_mean('data', ilsvrc2012_mean)            # subtract the dataset-mean value in each channel
    if scale != 1:
        transformer.set_raw_scale('data', scale)             # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))          # swap channels from RGB to BGR
    return transformer
