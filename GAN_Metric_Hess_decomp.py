import torch
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, compute_hessian_eigenthings

#%% Prepare the Networks
import sys
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda() # this notation is incorrect in older pytorch

import torchvision as tv
VGG = tv.models.vgg16(pretrained=True)
layernames = lucent_layernames(VGG)
#%%
feat = torch.randn((4096), dtype=torch.float32).requires_grad_(False).cuda()
GHVP = GANHVPOperator(G, feat, model_squ)
GHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())

#%%
obj_f = FeatLinModel(VGG, layername, weight=)
activHVP = GANHVPOperator(G, feat, obj_f)
activHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())
#%%
# Set up a network
from collections import OrderedDict
class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()

def hook_model(model, layerrequest = None):
    features = OrderedDict()
    alllayer = layerrequest is None
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                cur_layername = "_".join(prefix + [name])
                if alllayer:
                    features[cur_layername] = ModuleHook(layer)
                elif not alllayer and cur_layername in layerrequest:
                    features[cur_layername] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        # if layer == "input":
        #     return image
        if layer == "labels":
            return list(features.values())[-1].features
        return features[layer].features

    return hook, features
#%%
VGG = tv.models.vgg16(pretrained=True).requires_grad_(False)
#%%
hooks, feat_dict = hook_model(VGG, layerrequest = ('features_20',))
layernames = list(feat_dict.keys())
print(layernames)
tmpimg = torch.randn(1, 3, 224, 224)
VGG.forward(tmpimg)
feat = hooks('features_20')
# for name, hk in feat_dict.items():
#     hk.close()
#%%
def feat_tsr_weight(layer, weight=None, ):
    """ Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel. """
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.unsqueeze(0)).mean()

    return inner
