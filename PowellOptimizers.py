"""
Obsolete code to test if BOBYQA optimizer is useful for the GAN-CNN pipeline optimization 
It takes super long and doesn't converge......Need too much info to compute a Hessian. 
"""
from pybobyqa import solver
#% Prepare PyTorch version of the Caffe networks
import sys
sys.path.append("D:\Github\pytorch-caffe")
from caffenet import *  # Pytorch-caffe converter
import numpy as np
import torch
import torch.nn.functional as F
from torch_net_utils import load_caffenet, load_generator, visualize, BGR_mean
import matplotlib.pylab as plt
from time import time
#%%
class GAN_CNN_pipeline:
    """Concatenate GAN and CNN as a function to send into pybobya"""
    def __init__(self, unit=None):
        self.net = load_caffenet()
        self.G = load_generator()
        self.unit = unit

    def select_unit(self, unit):
        self.unit = unit

    def forward(self, feat):
        '''Assume feat has already been torchify, forward only one vector and get one score'''
        unit = self.unit
        blobs = self.G(feat)  # forward the feature vector through the GAN
        out_img = blobs['deconv0']  # get raw output image from GAN
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
        resz_out_img = F.interpolate(clamp_out_img - BGR_mean, (227, 227), mode='bilinear', align_corners=True)
        blobs_CNN = self.net(resz_out_img)
        if len(unit) == 5:
            score = blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
        elif len(unit) == 3:
            score = blobs_CNN[unit[1]][0, unit[2]]
        return score

    def visualize(self, codes):
        # img = visualize(self.G, code)
        code_num = codes.shape[0]
        assert codes.shape[1] == 4096
        imgs = [visualize(self.G, codes[i, :]) for i in range(code_num)]
        return imgs

    def score(self, codes, with_grad=False):
        """Validated that this score is approximately the same with caffe Model"""
        code_num = codes.shape[0]
        assert codes.shape[1] == 4096
        scores = []
        for i in range(code_num):
            # make sure the array has 2 dimensions
            feat = Variable(torch.from_numpy(np.float32(codes[i:i+1, :])), requires_grad=with_grad)
            score = self.forward(feat)
            scores.append(score)
        return np.array(scores)

    def optim_score(self, code):
        feat = torch.from_numpy(np.float32(code[np.newaxis, :])) #, requires_grad=False)
        score = self.forward(feat)
        return - np.array(score)


Scorer = GAN_CNN_pipeline(('caffe-net', 'fc8', 10))
#%%
t0 = time()
z0 = np.zeros((4096, ))
rst = solver.solve(Scorer.optim_score, z1, maxfun=20000, rhobeg=0.2, bounds=(-300 * np.ones_like(z0), 300 * np.ones_like(z0)), scaling_within_bounds=True)
print(rst)
t1 = time()
print("%.1f sec" % (t1-t0))
z1 = rst.x
img = Scorer.visualize(rst.x[np.newaxis, ])
plt.imshow(img[0])
plt.show()