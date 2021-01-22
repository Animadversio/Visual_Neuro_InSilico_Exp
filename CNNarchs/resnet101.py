from os.path import join
import torchvision, torch
from torchvision.models import resnet101
from collections import OrderedDict

resnet = resnet101(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(resnet, input_size=(3, 227, 227))
#%%
".Linearfc"