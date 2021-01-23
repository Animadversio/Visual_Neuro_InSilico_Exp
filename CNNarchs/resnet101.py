from os.path import join
import torchvision, torch
from torchvision.models import resnet101
from collections import OrderedDict
import numpy as np
resnet = resnet101(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(resnet, input_size=(3, 227, 227))
#%%
from torch_net_utils import receptive_field, receptive_field_for_unit
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
# def print_name(module):
#     print(str(module.__class__).split(".")[-1].split("'")[0], module.name)
#
# resnet_feat.apply(print_name);
#%%
".Linearfc"
#%%
psumnum = 0
for p in resnet101.parameters():
    print(p.shape, np.prod(list(p.shape)))
    psumnum += np.prod(list(p.shape))
print(psumnum)