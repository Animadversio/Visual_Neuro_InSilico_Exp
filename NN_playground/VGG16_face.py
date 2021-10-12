#%%
import matplotlib.pylab as plt
import torch, numpy as np
import torch.nn.functional as F
from sys import path
from os.path import join
from collections import OrderedDict

path.append(r"D:\Github\vgg-face.pytorch") # \models\vgg_face.py
import models.vgg_face as vggface
vgg_F = vggface.VGG_16()
vgg_F.load_weights(r"D:\Github\vgg-face.pytorch\pretrained\VGG_FACE.t7")
#%% Try to move the vgg model into the pretrained vgg models
from torchvision.models import vgg16
vgg_trc = vgg16(False, num_classes=2622)
#%%
oldstatedict = vgg_F.state_dict()
newstatedict = {}
for trcnm, vggnm in zip(vgg_trc.state_dict(), oldstatedict):
    newstatedict[trcnm] = oldstatedict[vggnm].detach().clone()

vgg_trc.load_state_dict(newstatedict)
#%%
torch.save(vgg_trc.state_dict(), join(r"E:\Cluster_Backup\torch", "vgg16_face.pt"))
#%%
vgg_trc = vgg16(False, num_classes=2622)
vgg_trc.load_state_dict(torch.load(join(r"E:\Cluster_Backup\torch", "vgg16_face.pt")))

im = plt.imread("D:/Github/vgg-face.pytorch/images/ak.png")
im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).float()
im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().view(1, 3, 1, 1)
im /= 255.0

vgg_trc.eval()
preds = F.softmax(vgg_trc(im), dim=1)