"""
This folder devotes to test the hypothesis, whether the prorotypes matter
or it could be rotated arbitrarily.
"""
#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# from featvis_lib import load_featnet
from insilico_Exp_torch import load_featnet
from layer_hook_utils import featureFetcher
from GAN_utils import upconvGAN
from torchvision.transforms import ToTensor, Normalize, ToPILImage, Compose
from torch_utils import show_imgrid, save_imgrid
from os.path import join
import pickle as pkl
from scipy.stats import special_ortho_group, ortho_group

featnet, net = load_featnet("resnet50_linf8") #
G = upconvGAN().cuda().eval()
G.requires_grad_(False)
preprocess = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])  # Imagenet normalization RGB
#%%
#%%
featkey = ".Linearfc"
channum = 2048
fetcher = featureFetcher(featnet, input_size=(3, 256, 256))
# fetcher.record(".AdaptiveAvgPool2davgpool", ingraph=True)
fetcher.record(featkey, ingraph=True, return_input=True)
Mrot = special_ortho_group.rvs(dim=channum)
rotMat = torch.tensor(Mrot).float().cuda()
#%%
featkey = ".Linearfc"
channum = 1000
fetcher = featureFetcher(featnet, input_size=(3, 256, 256))
fetcher.record(featkey, ingraph=True, return_input=False)
Mrot = special_ortho_group.rvs(dim=channum)
rotMat = torch.tensor(Mrot).float().cuda()
#%%
rootdir = r"E:\OneDrive - Harvard University\CNN_rotated_prototypes"
# figdir = r"E:\OneDrive - Harvard University\CNN_rotated_prototypes\resnet50_linf8\Linearfc_input"
figdir = r"E:\OneDrive - Harvard University\CNN_rotated_prototypes\resnet50_linf8\Linearfc_output"
torch.save(rotMat, join(figdir, "rotMat.pt"))
for unit_i in range(channum):
    print("Pooling layer rotated unit {}".format(unit_i))
    zs = torch.randn(5, 4096).cuda()
    zs.requires_grad_(True)
    optimizer = optim.Adam([zs], lr=0.08)
    pbar = tqdm(range(100))
    for i in pbar:
        imgs = G.visualize(zs)
        net(imgs)
        # activation = fetcher[featkey][0] @ rotMat[:, unit_i]
        activation = fetcher[featkey] @ rotMat[:, unit_i]
        loss = - activation.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"{activation.mean().item():.3f}+-{activation.std().item():.3f}")
        # if i % 10 == 0:
        #     print(f"{i} {activation.mean().item():.3f}+-{activation.std().item():.3f}")
    scores, idxs = activation.detach().sort(descending=True)
    imgs = imgs[idxs, :, :, :]
    zs = zs[idxs, :]
    save_imgrid(imgs, join(figdir, f"rotate_unit_{unit_i}_proto.png"))
    torch.save({"zs":zs, "scores":scores}, join(figdir, f"rotate_unit_{unit_i}_data.pt"))
    # break
#%%
show_imgrid(imgs)

