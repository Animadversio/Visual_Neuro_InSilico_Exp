import torch
from GAN_utils import upconvGAN
G = upconvGAN()
paramnum = 0
for param in G.G.parameters():
    print(param.shape)
    paramnum += torch.prod(param.shape)
