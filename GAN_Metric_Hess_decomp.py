"""Demo code for computing Neuron's tuning w.r.t """
import torch
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from lanczos_generalized import lanczos_generalized
from GAN_hvp_operator import GANHVPOperator, compute_hessian_eigenthings

#%% Prepare the Networks
import sys
sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
sys.path.append(r"D:\Github\PerceptualSimilarity")
import models
model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
model_squ.requires_grad_(False).cuda()

from GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False).cuda() # this notation is incorrect in older pytorch

#%% Set up hook and the linear network based on the CNN
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
#%
def FeatLinModel(VGG, layername='features_20', type="weight", weight=None, chan=0, pos=(10, 10)):
    """A factory of linear models on """
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
            if scaler:
                return -(feat[:, chan, pos[0], pos[1]]).mean()
            else:
                batch = img.shape[0]
                return -(feat[:, chan, pos[0], pos[1]]).view(batch, -1).mean(axis=1)

        return neuron_objective


# for name, hk in feat_dict.items():
#     hk.close()
#%%
#%%
import torchvision as tv
# VGG = tv.models.vgg16(pretrained=True)
alexnet = tv.models.alexnet(pretrained=True).cuda()
for param in alexnet.parameters():
    param.requires_grad_(False)
#%% This is not working.... The local 2nd order derivative is 0
feat = torch.randn((4096), dtype=torch.float32).requires_grad_(False).cuda()
GHVP = GANHVPOperator(G, feat, model_squ)
GHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())
#%%
weight = torch.randn(512,32,32).cuda()
objective = FeatLinModel(VGG, layername='features_19', type="weight", weight=weight)
activHVP = GANHVPOperator(G, 5*feat, objective, activation=True)
#%
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
A = torch.tensor([[2.0, 3], [3, 1]])
y = x.view(1, -1)@A@x.view(-1, 1)
x_grad = torch.autograd.grad(y, x, retain_graph=True, create_graph=True)
torch.autograd.grad(x_grad, x, retain_graph=True, only_inputs=True)
#%%
import numpy as np
from time import time
from imageio import imwrite
from build_montages import build_montages
import matplotlib.pylab as plt
from os.path import join
import torch.nn.functional as F
#%%
feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(True)
img = G.visualize(feat)
resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
obj = alexnet.features[:10](resz_img)[0, :, 6, 6].mean().pow(2)  # esz_img.std()
ftgrad = torch.autograd.grad(obj, feat, retain_graph=True, create_graph=True, only_inputs=True)
torch.autograd.grad(1 * ftgrad[0], feat, retain_graph=True, only_inputs=True, grad_outputs=torch.randn(4096).cuda(), )
# torch.autograd.grad(ftgrad, img, retain_graph=True, only_inputs=True, grad_outputs=torch.randn(4096).cuda(), )
#%% Approximate Forward Differencing
"""
So here is the conclusion, as the Perceptual loss take a squared difference when comparing 
feature tensros, the dependency of loss on image is more than power 1, and the derivative 
of it is not independent of image. However if the 
"""
def torch_corr(vec1, vec2):
    return torch.mean((vec1 - vec1.mean()) * (vec2 - vec2.mean())) / vec1.std(unbiased=False) / vec2.std(unbiased=False)

feat = torch.tensor(np.random.randn(4096)).float().cuda()
feat.requires_grad_(False)
vect = torch.tensor(np.random.randn(4096)).float().cuda()
vect = vect / vect.norm()
vect.requires_grad_(False)
#%% Through this I can show that the HVP is converging
#   Forward differencing method. One Free parameter is the "eps" i.e. the norm of perturbation to apply on the central
#   vector. Too small norm of this will make the
hvp_col = []
for eps in [50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ]:
    perturb_vecs = 5*feat.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).cuda() * vect.detach()
    perturb_vecs.requires_grad_(True)
    img = G.visualize(perturb_vecs)
    resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
    obj = alexnet.features[:10](resz_img)[:, :, 6, 6].mean()   # esz_img.std()
    ftgrad_both = torch.autograd.grad(obj, perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)
    hvp = (ftgrad_both[0][0, :] - ftgrad_both[0][1, :]) / (2 * eps)
    hvp_col.append(hvp)
    print(hvp)
    # img = G.visualize(feat - eps * vect)
    # resz_img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=True)
    # obj = alexnet.features[:10](resz_img)[0, :, 6, 6].sum() #esz_img.std()
    # ftgrad_neg = torch.autograd.grad(obj, vect, retain_graph=False, create_graph=False, only_inputs=True)
    # hvp = (ftgrad_pos[0] - ftgrad_neg[0]) / eps / 2
#%
for i in range(len(hvp_col)):
    print("correlation %.4f mse %.1E" % (torch_corr(hvp_col[i], hvp_col[1]).item(),
                                         F.mse_loss(hvp_col[i], hvp_col[1]).item()))
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
hvp_arr = torch.cat(tuple(hvp.unsqueeze(0) for hvp in hvp_col), dim=0)
corrmat = np.corrcoef(hvp_arr.cpu().numpy())
plt.matshow(corrmat, cmap=plt.cm.jet)
plt.yticks(range(12), labels=[50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ])
plt.xticks(range(12), labels=[50, 25, 10, 5, 1, 5E-1, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, ])
plt.ylim(top = -0.5, bottom=11.5)
plt.xlim(left = -0.5, right=11.5)
plt.xlabel("Perturb Vector Length")
plt.suptitle("Correlation of HVP result\nusing different EPS in forward differencing")
plt.colorbar()
plt.savefig(join(savedir, "HVP_corr_oneTrial.jpg") )
plt.show()
#%%
class GANForwardHVPOperator(Operator):
    def __init__(
            self,
            model,
            code,
            objective,
            preprocess=lambda img: F.interpolate(img, (224, 224), mode='bilinear', align_corners=True),
            use_gpu=True,
            EPS=1E-2,
            # activation=False,
    ):
        if use_gpu:
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        if hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(objective, "parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.model = model
        self.objective = objective
        self.preprocess = preprocess
        self.code = code.clone().requires_grad_(False).float().to(device)  # torch.float32
        self.img_ref = self.model.visualize(self.code)
        resz_img = self.preprocess(self.img_ref)  # F.interpolate(self.img_ref, (224, 224), mode='bilinear', align_corners=True)
        activ = self.objective(resz_img)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.code.norm() * self.EPS

    def select_code(self, code):
        self.code = code.clone().requires_grad_(False).float().to(self.device)  # torch.float32
        self.perturb_norm = self.code.norm() * self.EPS
        self.img_ref = self.model.visualize(self.code + self.perturb_vec)
        resz_img = self.preprocess(self.img_ref)
        activ = self.objective(resz_img)
        gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=False, retain_graph=False)[0]
        self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        EPS = self.EPS if EPS is None else EPS
        self.perturb_norm = self.code.norm() * EPS
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).to(self.device) * vec.detach()
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(perturb_vecs)
        resz_img = self.preprocess(img)
        activs = self.objective(resz_img)  # , scaler=True
        # obj = alexnet.features[:10](resz_img)[:, :, 6, 6].sum()  # esz_img.std()
        ftgrad_both = torch.autograd.grad(activs, perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[0, :] - ftgrad_both[1, :]) / (2 * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass
#%%
from torchvision.transforms import Normalize, Compose
RGB_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
RGB_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()
preprocess = Compose([lambda img: (F.interpolate(img, (224, 224), mode='bilinear', align_corners=True) - RGB_mean) / RGB_std])
# weight = torch.randn(256, 13, 13).cuda()
# objective = FeatLinModel(alexnet, layername='features_10', type="weight", weight=weight)
objective = FeatLinModel(alexnet, layername='features_10', type="neuron", chan=slice(None), pos=(10, 10))

feat = 5*torch.randn(4096).cuda()

activHVP = GANForwardHVPOperator(G, feat, objective, preprocess=preprocess)
activHVP.apply(1*torch.randn((4096)).requires_grad_(False).cuda())
#%%
import torch.optim as optim
feat = 5*torch.randn(4096).cuda()
feat.requires_grad_(True)
optimizer = optim.Adam([feat], lr=5e-2)
for step in range(100):
    optimizer.zero_grad()
    obj = objective(preprocess(G.visualize(feat)))
    obj.backward()
    optimizer.step()
    if np.mod((step + 1), 10) == 0:
        print("step %d: %.2f"%(step, obj.item()))
#%%
feat.requires_grad_(False)
activHVP = GANForwardHVPOperator(G, feat, objective, preprocess=preprocess)
activHVP.apply(1*torch.randn((4096)).requires_grad_(False).cuda())
#%%
t0 = time()
eigvals, eigvects = lanczos(activHVP, num_eigenthings=500, use_gpu=True)
print(time() - t0)  # 40 sec
#%
eigvals = eigvals[::-1]
eigvects = eigvects[::-1, :]
#%%
summary_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessDecomp_Method"
#%%
RND = np.random.randint(100)
ref_vect = (feat / feat.norm()).cpu().numpy()
save_indiv = False
save_row = False
vec_norm = feat.norm().item()
ang_step = 180 / 10
theta_arr_deg = ang_step * np.linspace(-5, 5, 21)# np.arange(-5, 6)
theta_arr = theta_arr_deg / 180 * np.pi
img_list_all = []
scores_col = []
eig_id_arr = [0, 1, 5, 10, 15, 20, 40, 60, 80,99,150,200,250,299,450]
for eig_id in eig_id_arr:#,600,799]:
    # eig_id = 0
    perturb_vect = eigvects[eig_id,:]  # PC_vectors[1,:]
    codes_arc = np.array([np.cos(theta_arr),
                          np.sin(theta_arr) ]).T @ np.array([ref_vect, perturb_vect])
    norms = np.linalg.norm(codes_arc, axis=1)
    codes_arc = codes_arc / norms[:, np.newaxis] * vec_norm
    imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
    scores = - objective(preprocess(imgs), scaler=False)
    scores_col.append(scores.cpu().numpy())
    npimgs = imgs.detach().cpu().permute([2, 3, 1, 0]).numpy()

    if save_indiv:
        for i in range(npimgs.shape[3]):
            angle = theta_arr_deg[i]
            imwrite(join(newimg_dir, "norm%d_eig%d_ang%d.jpg" % (vec_norm, eig_id, angle)), npimgs[:, :, :, i])

    img_list = [npimgs[:, :, :, i] for i in range(npimgs.shape[3])]
    img_list_all.extend(img_list)
    if save_row:
        mtg1 = build_montages(img_list, [256, 256], [len(theta_arr), 1])[0]
        imwrite(join(summary_dir, "norm%d_eig_%d.jpg" % (vec_norm, eig_id)), mtg1)
mtg_all = build_montages(img_list_all, [256, 256], [len(theta_arr), int(len(img_list_all) // len(theta_arr))])[0]
imwrite(join(summary_dir, "norm%d_eig_all_opt_%d.jpg" % (vec_norm, RND)), mtg_all)
#%
scores_col = np.array(scores_col)
plt.matshow(scores_col)
plt.axis('image')
plt.title("Neural Tuning Towards Different Eigen Vectors of Activation")
plt.xlabel("Angle")
plt.ylabel("Eigen Vector #")
eiglabel = ["%d %.3f"%(id,eig) for id, eig in zip(eig_id_arr, eigvals[eig_id_arr])]
plt.yticks(range(len(eig_id_arr)), eiglabel) # eig_id_arr
plt.ylim(top=-0.5, bottom=len(eig_id_arr) - 0.5)
plt.colorbar()
plt.savefig(join(summary_dir, "norm%d_score_mat_%02d.jpg" % (vec_norm, RND)) )
plt.show()


#%%
scores_col = []
for eig_id in [0,1,5,10,15,20,40,60,80,99,150,200,250,299,450,600,799]:
    # eig_id = 0
    perturb_vect = eigvects[eig_id,:] # PC_vectors[1,:]
    codes_arc = np.array([np.cos(theta_arr),
                          np.sin(theta_arr) ]).T @ np.array([ref_vect, perturb_vect])
    norms = np.linalg.norm(codes_arc, axis=1)
    codes_arc = codes_arc / norms[:,np.newaxis] * vec_norm
    imgs = G.visualize(torch.from_numpy(codes_arc).float().cuda())
    scores = objective(F.interpolate(imgs, (224, 224), mode='bilinear', align_corners=True), scaler=False)
    scores_col.append(scores.cpu().numpy())

scores_col = np.array(scores_col)
#%%
plt.matshow(scores_col)
plt.axis('image')
plt.colorbar()
plt.show()