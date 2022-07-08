#%%

import torch
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stats_utils import saveallforms
from lpips import LPIPS
from insilico_Exp_torch import ExperimentManifold
from torch_utils import show_imgrid, save_imgrid
from NN_sparseness.sparse_invariance_lib import shorten_layername
from NN_sparseness.insilico_manif_configs import RN50_config
# from Manifold.manifold_exp_func_lib import run_manifold, run_evol


def compute_distmat(Dist, imgtsr, batch=40, space=False):
    Nimg = imgtsr.shape[0]
    distmat_col = []
    for i in range(0, Nimg, batch):
        distmat_row = []
        for j in range(0, Nimg, batch):
            distmat_part = Dist.forward_distmat(\
                imgtsr[i:i+batch].cuda(), imgtsr[j:j+batch].cuda()).cpu()
            distmat_row.append(distmat_part)
        distmat_row = torch.concat(distmat_row, dim=1)
        distmat_col.append(distmat_row)
    distmat_all = torch.concat(distmat_col, dim=0)
    if space is False:
        distmat_all= distmat_all.squeeze(2).squeeze(2).squeeze(2)
    return distmat_all


savedir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\Manif_imdiversity\insilico"
#%%
Dist = LPIPS(net='squeeze').cuda()
#%% Calcluate diversity of images on the Manifold image space. 
exp = ExperimentManifold(('resnet50_linf8', '.layer3.Bottleneck5', 5, 7, 7),
                         max_step=50, imgsize=(227, 227))
layername = ".layer4.Bottleneck2"
layername = ".layer1.Bottleneck1"
imdivers_list = []
for chan in range(50):
    model_unit = ("resnet50_linf8", layername, chan, *RN50_config[layername]["unit_pos"])
    exp.re_init() # exp.CNNmodel.cleanup()
    exp.CNNmodel.select_unit(model_unit, False)
    #%%
    exp.run()
    exp.analyze_traj()
    torch.cuda.empty_cache()
    #%%
    # exp.run_manifold([(1,2)], interval=18, print_manifold=False)
    imgtsr = exp.render_manifold([(1,2)], interval=18, )[0]
    imgtsr_uniq = imgtsr.reshape(11, 11, 3, 256, 256).transpose(0, 1)\
        .reshape(121, 3, 256, 256)[10:111]
    #%%
    with torch.no_grad():
        distmat_all = compute_distmat(Dist, imgtsr_uniq, batch=15, )
    #%%
    # # show_imgrid(imgtsr_uniq, )
    #%%
    msk = torch.tril(torch.ones_like(distmat_all) * torch.nan)
    distmat_msk = distmat_all + msk
    #%%
    meandist = distmat_msk.nanmean()
    imdivers_list.append(meandist)
    PC012 = exp.PC_vectors[:3,:]
    scores, _ = exp.run_manifold([(1, 2)], interval=18, print_manifold=False)
    np.savez(join(savedir, f"{model_unit[0]}_{shorten_layername(model_unit[1])}_chan{model_unit[2]}.npz"),
             scores=scores, PC012=PC012, sphere_norm=exp.sphere_norm,
             distmat_all=distmat_all, meandist=meandist)
#%%
torch.save(torch.tensor(imdivers_list),
           join(savedir, f"{model_unit[0]}_{shorten_layername(model_unit[1])}_img_diversity.pt"))
#%%
netname = "resnet50_linf8"
imdiv_dict = {}
for layername in [".layer1.Bottleneck1",
                  ".layer2.Bottleneck3",
                  ".layer3.Bottleneck4",
                  ".layer4.Bottleneck2"]:
    imgdiv = torch.load(join(savedir, f"{netname}_"
                  f"{shorten_layername(layername)}_img_diversity.pt"))
    print(f"mean img diversity {layername}: {imgdiv.mean():.4f} sem: {imgdiv.std()/np.sqrt(imgdiv.numel()):.4f}")
    imdiv_dict[layername] = imgdiv
# ".layer4.Bottleneck2"   tensor(0.4335) +- 0.0021
# ".layer3.Bottleneck4"   tensor(0.4205)
# ".layer2.Bottleneck3"   tensor(0.4162) +- 0.0028
#%% summary of im diversity
df = pd.DataFrame(imdiv_dict)
df.columns = ["layer1", "layer2", "layer3", "layer4"]
df.to_csv(join(savedir, f"{netname}_Manifold_im_diversity_summary.csv"))
fig, ax = plt.subplots(figsize=(3.5, 5))
sns.stripplot(data=df, jitter=True, linewidth=0.5, alpha=0.7)
sns.boxplot(data=df, linewidth=0.5, )
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.set(xlabel="layer", ylabel="mean LPIPS", title=f"Resnet50-robust Manifold space\n image diversity")
plt.tight_layout()
saveallforms(savedir, f"{netname}_img_diversity", fig)
plt.show()