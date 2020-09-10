import torch
use_gpu = True if torch.cuda.is_available() else False
# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-512',
#                        pretrained=True, useGPU=use_gpu)
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=use_gpu)
num_images = 1
noise, _ = model.buildNoiseData(num_images)
noise.requires_grad_(True)
# with torch.no_grad():
generated_images = model.test(noise)