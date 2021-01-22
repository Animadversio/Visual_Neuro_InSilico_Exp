from torch_net_utils import receptive_field, receptive_field_for_unit
from os.path import join
import torchvision, torch
from torchvision.models import densenet161
from collections import OrderedDict
import numpy as np
dnet = densenet161(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
# from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(dnet, input_size=(3, 227, 227))
#%%
# resnet101 = torchvision.models.resnet101(pretrained=True)
# rf_dict = receptive_field(vgg16.features, (3, 227, 227), device="cpu")
resnet101 = torchvision.models.resnet101(pretrained=True)  # using the pytorch alexnet as proxy for caffenet.
resnet_feat = torch.nn.Sequential(resnet101.conv1,
                               resnet101.bn1,
                               resnet101.relu,
                               resnet101.maxpool,
                               resnet101.layer1,
                               resnet101.layer2,
                               resnet101.layer3,
                               resnet101.layer4)
rf_dict = receptive_field(resnet_feat, (3, 227, 227), device="cpu")
#%%

torchvision.models.resnet101()
#%%
psumnum = 0
for p in resnet101.parameters():
    print(p.shape, np.prod(list(p.shape)))
    psumnum += np.prod(list(p.shape))
print(psumnum)
#%%
# from receptivefield.pytorch import PytorchReceptiveField
# from receptivefield.image import get_default_image
# import torch.nn as nn
# def model_fn() -> nn.Module:
#     model = resnet_feat#(disable_activations=True)
#     model.eval()
#     return model
#
# input_shape = [227, 227, 3]
# rf = PytorchReceptiveField(model_fn)
# rf_params = rf.compute(input_shape=input_shape)
# # plot receptive fields
# rf.plot_rf_grids(
#     custom_image=get_default_image(input_shape, name='cat'),
#     figsize=(20, 12),
#     layout=(1, 2))
#%%
def print_name(module):
    print(str(module.__class__).split(".")[-1].split("'")[0], module.name)

resnet_feat.apply(print_name);
