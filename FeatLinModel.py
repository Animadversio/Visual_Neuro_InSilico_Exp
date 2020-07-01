#   Set up a network
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

def get_model_layers(model, getLayerRepr=False):
    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers

def FeatLinModel(VGG, layername='features_20', type="weight", weight=None, chan=0, pos=(10, 10)):
    """A factory of linear models on a pretrained CNN.
    It's a scorer of image. """
    layers_all = get_model_layers(VGG)
    if 'features' in layername:
        layeridx = layers_all.index(layername) - 1 + 1 # -1 for the "features" layer
        VGGfeat = VGG.features[:layeridx]
    else:
        VGGfeat = VGG
    hooks, feat_dict = hook_model(VGG, layerrequest=(layername,))
    layernames = list(feat_dict.keys())
    print(layernames)
    if type == "weight":
        def weight_objective(img, scaler=True):
            VGGfeat.forward(img.cuda())
            feat = hooks(layername)
            if scaler:
                return -(feat * weight.unsqueeze(0)).mean()
            else:
                batch = img.shape[0]
                return -(feat * weight.unsqueeze(0)).view(batch, -1).mean(axis=1)

        return weight_objective
    elif type == "neuron":
        def neuron_objective(img, scaler=True):
            VGGfeat.forward(img.cuda())
            feat = hooks(layername)
            if len(feat.shape) == 4:
                if scaler:
                    return -(feat[:, chan, pos[0], pos[1]]).mean()
                else:
                    batch = img.shape[0]
                    return -(feat[:, chan, pos[0], pos[1]]).view(batch, -1).mean(axis=1)
            elif len(feat.shape) == 2:
                if scaler:
                    return -(feat[:, chan]).mean()
                else:
                    batch = img.shape[0]
                    return -(feat[:, chan]).view(batch, -1).mean(axis=1)
        return neuron_objective
