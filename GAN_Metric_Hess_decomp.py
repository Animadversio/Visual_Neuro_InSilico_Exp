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
#%%
import torchvision as tv
VGG = tv.models.vgg16(pretrained=True)
#%%
feat = torch.randn((4096), dtype=torch.float32).requires_grad_(False).cuda()
GHVP = GANHVPOperator(G, feat, model_squ)
GHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())
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
#%%
def FeatLinModel(VGG, layername='features_20', type="weight", weight=None):
    """A factory of linear models on """
    layers_all = get_model_layers(VGG)
    if 'features' in layername:
        layeridx = layers_all.index(layername) - 1 + 1 # -1 for the "features" layer
        VGGfeat = VGG.features[:layeridx]
    else:
        VGGfeat = VGG
    hooks, feat_dict = hook_model(VGG, layerrequest = (layername,))
    layernames = list(feat_dict.keys())
    print(layernames)
    if type == "weight":
        def weight_objective(img):
            VGGfeat.forward(img.cuda())
            feat = hooks(layername)
            return -(feat * weight.unsqueeze(0)).mean()

        return weight_objective
    elif type == "neuron":
        def neuron_objective(img):
            VGGfeat.forward(img.cuda())
            feat = hooks(layername)
            return -(feat[:, 10, 15, 15]).mean()

        return neuron_objective



# for name, hk in feat_dict.items():
#     hk.close()
#%%
VGG = tv.models.vgg16(pretrained=True).requires_grad_(False)
#%%
weight = torch.randn(512,32,32).cuda()
objective = FeatLinModel(VGG, layername='features_19', type="weight", weight=weight)
activHVP = GANHVPOperator(G, 5*feat, objective, activation=True)
#%%
activHVP.apply(5*torch.randn((4096)).requires_grad_(False).cuda())
#%%

#%%
feat = torch.randn(4096).cuda()
feat.requires_grad_(True)
objective = FeatLinModel(VGG, layername='features_4', type="neuron", weight=None)
act = objective(G.visualize(feat))
#%%
from hessian import hessian
# activHVP = GANHVPOperator(G, 5*feat, objective, activation=True)
H = hessian(act, feat)
#%%
alexnet = tv.models.alexnet(pretrained=True).cuda()
for param in alexnet.parameters():
    param.requires_grad_(False)
#%%
feat = torch.randn(4096).cuda()
feat.requires_grad_(True)
#%%
weight = torch.randn(192, 31, 31).cuda()
objective = FeatLinModel(alexnet, layername='features_4', type="weight", weight=weight)
act = objective(G.visualize(feat))
#%%
gradient = torch.autograd.grad(act, feat, retain_graph=True, create_graph=True,)
torch.autograd.grad(gradient[0], feat, retain_graph=True, only_inputs=True, grad_outputs=10*torch.ones(4096).cuda())
#%%
import numpy as np
feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(True)
img = G.visualize(feat)
fc8 = alexnet.forward(img)
act = - fc8[0, 1]
H = hessian(act, feat, create_graph=False)
#%%
import numpy as np
feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(True)
img = G.visualize(feat)
act = - img.mean()
# fc8 = alexnet.forward(img)
# act = - fc8[0, 1]
# H = hessian(act, feat, create_graph=False)
#%%
gradient = torch.autograd.grad(act, feat, retain_graph=True, create_graph=True,)
torch.autograd.grad(gradient[0], feat, retain_graph=True, only_inputs=True, grad_outputs=10*torch.ones(4096).cuda())
#%%
H = hessian(act, feat, create_graph=False)

#%%
x = torch.tensor([1.0,2])
x.requires_grad_(True)
A = torch.tensor([[2.0,3],[3,1]])
y = x.view(1,-1)@A@x.view(-1,1)
x_grad = torch.autograd.grad(y, x, retain_graph=True, create_graph=True)
torch.autograd.grad(x_grad, x, retain_graph=True, only_inputs=True)
#%%
import torch.nn.functional as F
feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(True)
img = G.visualize(feat)
# imgstd = (img**1).mean()
resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
obj = alexnet.features[:10](resz_img)[0,:,6,6].sum() #esz_img.std()
ftgrad = torch.autograd.grad(obj, feat, retain_graph=True, create_graph=True, only_inputs=True)
torch.autograd.grad(1 * ftgrad[0], feat, retain_graph=True, only_inputs=True, grad_outputs=torch.randn(4096).cuda(), )
# torch.autograd.grad(ftgrad, img, retain_graph=True, only_inputs=True, grad_outputs=torch.randn(4096).cuda(), )
#%% Approximate Forward Differencing
"""
So here is the conclusion, as the Perceptual loss take a squared difference when comparing 
feature tensros, the dependency of loss on image is more than power 1, and the derivative 
of it is not independent of image. However if the 
"""
feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(False)
vect = torch.tensor(np.random.randn(4096)).float().cuda()
vect = vect / vect.norm()
vect.requires_grad_(False)
#%% Through this I can show that the HVP is converging

hvp_col = []
for eps in [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]:
    perturb_vecs = feat.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).cuda() * vect.detach()
    perturb_vecs.requires_grad_(True)
    img = G.visualize(perturb_vecs)
    resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
    obj = alexnet.features[:10](resz_img)[:,:,6,6].sum() #esz_img.std()
    ftgrad_both = torch.autograd.grad(obj, perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)
    hvp = (ftgrad_both[0][0, :] - ftgrad_both[0][1, :]) / (2 * eps)
    hvp_col.append(hvp)

    # img = G.visualize(feat - eps * vect)
    # resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
    # obj = alexnet.features[:10](resz_img)[0, :, 6, 6].sum() #esz_img.std()
    # ftgrad_neg = torch.autograd.grad(obj, vect, retain_graph=False, create_graph=False, only_inputs=True)
    # hvp = (ftgrad_pos[0] - ftgrad_neg[0]) / eps / 2
    #
def torch_corr(vec1, vec2):
    return torch.mean((vec1 - vec1.mean()) * (vec2 - vec2.mean())) / vec1.std(unbiased=False) / vec2.std(unbiased=False)
#%%
for i in range(len(hvp_col)):
    print("correlation %.4f mse %.1E" % (torch_corr(hvp_col[i], hvp_col[-4]).item(),
                                         F.mse_loss(hvp_col[i], hvp_col[-4]).item()))
#%%
class GANForwardHVPOperator(Operator):
    def __init__(
            self,
            model,
            code,
            objective,
            use_gpu=True,
            # activation=False,
    ):
        if use_gpu:
            device = "cuda"
            self.device = device
        if hasattr(model,"parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(objective,"parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.model = model
        self.objective = objective
        self.code = code.clone().requires_grad_(False).float().to(device) # torch.float32
        self.img_ref = self.model.visualize(self.code)
        resz_img = F.interpolate(self.img_ref, (224, 224), mode='bilinear', align_corners=True)
        activ = self.objective(resz_img)
        self.size = self.code.numel()

    def select_code(self, code):
        self.code = code.clone().requires_grad_(False).float().to(self.device)  # torch.float32
        self.img_ref = self.model.visualize(self.code + self.perturb_vec)
        activ = self.objective(self.img_ref)
        gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=False, retain_graph=False)[0]
        self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=1E-3):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm().item()
        eps = EPS / vecnorm
        self.zero_grad()
        # take the second gradient
        perturb_vecs = self.code.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).to(self.device) * vec.detach()
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(perturb_vecs)
        resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
        activs = self.objective(resz_img)
        # obj = alexnet.features[:10](resz_img)[:, :, 6, 6].sum()  # esz_img.std()
        ftgrad_both = torch.autograd.grad(activs, perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)
        hessian_vec_prod = (ftgrad_both[0][0, :] - ftgrad_both[0][1, :]) / (2 * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            self.gradient, self.perturb_vec, grad_outputs=vec, only_inputs=True, retain_graph=True
        )
        hessian_vec_prod = grad_grad[0].view(-1)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass
#%%
feat = torch.randn(4096).cuda()
weight = torch.randn(256, 13, 13).cuda()
objective = FeatLinModel(alexnet, layername='features_10', type="weight", weight=weight)
activHVP = GANForwardHVPOperator(G, 5*feat, objective,)
#%%
activHVP.apply(1*torch.randn((4096)).requires_grad_(False).cuda())