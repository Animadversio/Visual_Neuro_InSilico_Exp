import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from insilico_Exp_torch import TorchScorer
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
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

scorer.select_unit(("resnet50", neurlayer, 25, 6, 6), allow_grad=True)

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
apprx_layer = "layer3"
cctsr = featFetcher.cctsr[apprx_layer].numpy()
Ttsr = featFetcher.Ttsr[apprx_layer].numpy()
stdstr = featFetcher.featStd[apprx_layer].numpy()
covtsr = cctsr * stdstr
rect_mode = "Tthresh"; thresh = (None, 3)
bdr = 1; NF = 2
rect_cctsr = rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr)
rect_cctsr = np.nan_to_num(rect_cctsr)
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rect_cctsr,
                     bdr=bdr, Nfactor=NF, show=True,)
#%% Visualize features of the factorized model
from featvis_lib import vis_featmap_corr, vis_featvec_wmaps
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, apprx_layer, netname="resnet50_linf8",
                     featnet=featnet, bdr=0, Bsize=2, figdir="", savestr="", imshow=False, score_mode="cosine")

#%%
apprx_layer = "layer3"
plt.imshow(np.nanmean(np.abs(rect_cctsr), axis=0))
plt.colorbar()
plt.show()

#%% Calculate gradient Amplitude RF for the recording unit.
gradAmpmap = grad_RF_estimate(scorer.model, neurlayer, (25, 6, 6), input_size=(3,256,256),
                 device="cuda", show=True, reps=100, batch=10)
Xlim, Ylim = gradmap2RF_square(gradAmpmap,)
#%%

finimgs_col_neur, mtg_col_neur, _ = vis_featvec_wmaps(ccfactor, Hmaps, scorer.model, G, neurlayer, netname="resnet50",
                     featnet=None, bdr=0, Bsize=2, figdir="", savestr="", imshow=False, score_mode="cosine")
#%%
batch = 5
langevin_eps = 1.0
code = 2 * torch.randn(batch, 4096).cuda()
code.requires_grad_(True)
optimizer = torch.optim.SGD([code], lr=0.025)
for i in range(100):
    code = code + torch.randn(batch, 4096).cuda() * langevin_eps
    optimizer.zero_grad()
    scores = scorer.score_tsr_wgrad(G.visualize(code))
    avg_score = scores.mean().detach().item()
    loss = (-scores.sum())
    loss.backward()
    optimizer.step()
    print("Score {}".format(avg_score))

#%%
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
with torch.no_grad():
    imgs = G.visualize(code).cpu()
ToPILImage()(make_grid(imgs.cpu())).show()
#%%
import cma
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0)
for i in tqdm(range(150)):
    z = es.ask()
    z_arr = np.array(z)
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score_tsr(imgs, B=40)
    es.tell(z, (-resp).tolist())
    print("Score {:.3f} +- {:.3f}".format(resp.mean(),resp.std()))
# Use CMAES is better than Adam or SGD in optimizing this scorer.
ToPILImage()(make_grid(imgs.cpu())).show()
#%%
net, featnet = load_featnet("resnet50_linf8")
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(net, ["layer2", "layer3", "layer4"], netname='resnet50_robust', verbose=False)

#%%
import cma
featFetcher.init_corr()
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0)
for i in tqdm(range(150)):
    z = es.ask()
    z_arr = np.array(z)
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score_tsr_wgrad(imgs, B=40)
    es.tell(z, (-resp).tolist())
    print("Score {:.3f} +- {:.3f}".format(resp.mean(),resp.std()))
    with torch.no_grad():
        featnet(imgs)
    del imgs
    featFetcher.update_corr(resp.cpu())

featFetcher.calc_corr()
# Use CMAES is better than Adam or SGD in optimizing this scorer.
imgs = G.visualize(torch.tensor(z_arr).float().cuda()).cpu()
ToPILImage()(make_grid(imgs)).show()
#%%
apprx_layer = "layer3"
cctsr = featFetcher.cctsr[apprx_layer].numpy()
Ttsr = featFetcher.Ttsr[apprx_layer].numpy()
stdstr = featFetcher.featStd[apprx_layer].numpy()
covtsr = cctsr * stdstr
rect_mode = "Tthresh"; thresh = (None, 5)
bdr = 1; NF = 3
rect_cctsr = rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr)
rect_cctsr = np.nan_to_num(rect_cctsr)
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rect_cctsr,
                     bdr=bdr, Nfactor=NF, show=True,)
#%%
def image_gradMap(scorer, img_tsr):
    img_tsr_wg = img_tsr
    img_tsr_wg.requires_grad_(True)
    scores = scorer.score_tsr_wgrad(img_tsr_wg)
    scores.sum().backward()
    imggradAmp = img_tsr_wg.grad
    return imggradAmp
#%%
imggradAmp = image_gradMap(scorer, imgs[0:6,:,:,:])
imggradmap = imggradAmp.abs().sum(dim=[0, 1]).cpu()
plt.figure()
plt.imshow(imggradmap)
plt.show()
