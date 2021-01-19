import torchvision, torch
from torchvision.models import AlexNet, alexnet
Anet = alexnet(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
modulenames, moduletypes, module_spec = get_module_names(Anet.features, input_size=(3, 227, 227), device="cpu")
#%%
from torch_net_utils import receptive_field, receptive_field_for_unit
#%%
rfdict = receptive_field(Anet.features, (3,227,227), device="cpu")
receptive_field_for_unit(rfdict, "2", (28, 28))
receptive_field_for_unit(rfdict, "5", (13, 13))
receptive_field_for_unit(rfdict, "8", (6, 6))
receptive_field_for_unit(rfdict, "10", (6, 6))
receptive_field_for_unit(rfdict, "12", (6, 6))

