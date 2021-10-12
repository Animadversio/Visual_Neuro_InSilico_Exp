import torchvision, torch
from torchvision.models import inception_v3
incpt = inception_v3(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names
modulenames, moduletypes, module_spec = get_module_names(incpt, input_size=(3, 299, 299), device="cpu")