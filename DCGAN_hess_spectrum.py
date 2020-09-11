import torch
from time import time
import sys
import lpips
from GAN_hessian_compute import hessian_compute
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

use_gpu = True if torch.cuda.is_available() else False
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)

class DCGAN_wrapper():  # nn.Module
    def __init__(self, DCGAN, ):
        self.DCGAN = DCGAN

    def visualize(self, code, scale=1):
        imgs = self.DCGAN(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = DCGAN_wrapper(model.avgG)

#%%
noise, _ = model.buildNoiseData(1)
model.avgG