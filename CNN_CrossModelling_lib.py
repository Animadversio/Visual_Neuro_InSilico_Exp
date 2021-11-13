import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from insilico_Exp_torch import TorchScorer
from GAN_utils import upconvGAN
from ZO_HessAware_Optimizers import CholeskyCMAES
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
import sys
sys.path.append(r"D:\Github\Visual_Neuron_Modelling")
from CorrFeatTsr_lib import Corr_Feat_Machine
from featvis_lib import load_featnet, rectify_tsr, tsr_posneg_factorize
from CorrFeatTsr_visualize_lib import CorrFeatScore
#%% Creat a model neuron basis
scorer = TorchScorer("resnet50")
#%% Construct the modelling target neuron
neurlayer = ".layer4.Bottleneck0"
chan = 10
module_names, module_types, module_spec = get_module_names(scorer.model, input_size=(3, 227, 227), device="cuda", show=False);
layer_key = [k for k, v in module_names.items() if v == neurlayer][0]
feat_outshape = module_spec[layer_key]['outshape']
assert len(feat_outshape) == 3  # fc neurlayer will fail
cent_pos = (feat_outshape[1]//2, feat_outshape[2]//2)
print("Center position on the feature map is (%d %d) of neurlayer %s (tensor shape %s)" % (*cent_pos, neurlayer, feat_outshape))

scorer.select_unit(("resnet50", neurlayer, 5, 6, 6), allow_grad=True)

#%% Modelling Network, approximating the other neuron's tuning function.
# Model Network
from featvis_lib import load_featnet
net, featnet = load_featnet("resnet50_linf8")
featFetcher = Corr_Feat_Machine()
# featFetcher.register_hooks(net, ["conv2_2", "conv3_3", "conv4_3", "conv5_3"], netname='resnet_robust', verbose=False)
# featFetcher.register_hooks(net, [".layer3.Bottleneck2", ".layer3.Bottleneck6", ".layer4.Bottleneck0"], netname='resnet50_robust', verbose=False)
featFetcher.register_hooks(net, ["layer2", "layer3", "layer4"], netname='resnet50_robust', verbose=False)
featFetcher.init_corr()
#%% Use random image sampled from G to stimulate target and modeler
for i in tqdm(range(50)):
    imgs = G.visualize(3 * torch.randn(40,4096).cuda())
    resp = scorer.score_tsr_wgrad(imgs, B=40)
    with torch.no_grad():
        featnet(imgs)
    del imgs
    featFetcher.update_corr(resp.cpu())

featFetcher.calc_corr()
#%%
apprx_layer = "layer2"
cctsr = featFetcher.cctsr[apprx_layer].numpy()
Ttsr = featFetcher.Ttsr[apprx_layer].numpy()
rect_mode = "Tthresh"; thresh = (None, 2)
bdr = 1; NF = 3
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(cctsr, rect_mode, thresh, Ttsr=Ttsr),
                     bdr=bdr, Nfactor=NF, show=True,)
#%%
apprx_layer = "layer2"
cctsr = featFetcher.cctsr[apprx_layer].numpy()
Ttsr = featFetcher.Ttsr[apprx_layer].numpy()
plt.imshow(np.nanmean(np.abs(cctsr), axis=0))
plt.colorbar()
plt.show()

#%%
from grad_RF_estim import grad_RF_estimate
grad_RF_estimate(model, target_layer, target_unit, input_size=(3,227,227), device="cuda", show=True, reps=200, batch=1)