"""Test the consistency between the Caffe version of Caffe Net and Generator and Pytorch Translation of it. """

import matplotlib.pylab as plt
import numpy as np

from insilico_Exp import CNNmodel, CNNmodel_Torch
from imageio import imread
img = imread(r"D:\Github\CMAES_optimizer_matlab\fc8_02.jpg")
net = CNNmodel("caffe-net");
net_trc = CNNmodel_Torch("caffe-net");
#%%
unit = ('caffe-net', 'fc8', 5)
net.select_unit(unit)
net_trc.select_unit(unit)
caffe_act = net.score(img[np.newaxis,])
torch_act = net_trc.score(img[np.newaxis,])
print("For image , unit %s "%(unit,))
print("Caffenet run in Caffe scores %.3f "%(caffe_act, ))
print("Caffenet run in PyTorch scores %.3f "%(torch_act, ))
#%% Test that the
from utils import generator
from torch_net_utils import load_generator, visualize
G_torch = load_generator()  # Torch generative network
G_caffe = generator  # Caffe GAN
#%%
code = 5*np.random.randn(1, 4096)
img_caffe = G_caffe.visualize(code)  # uint8 255
img_torch = visualize(G_torch, code)  # scale 0, 1 by default
#%
plt.figure()
plt.subplot(121)
plt.imshow(img_caffe)
plt.axis("off")
plt.subplot(122)
plt.imshow(img_torch)
plt.axis("off")
plt.show()
