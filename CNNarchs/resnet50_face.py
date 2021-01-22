from os.path import join
import torchvision, torch
from torchvision.models import resnet50
from collections import OrderedDict

resnetF = resnet50(False, num_classes=8631)  # Face recognition Network's output is 8631 classes in VGG-Face Dataset.
#%%
ckptdir = r"E:\Cluster_Backup\torch"
import pickle as pkl
for pklname in ["resnet50_ft_weight.pkl", "resnet50_scratch_weight.pkl"]:
    data = pkl.load(open(join(ckptdir, pklname), mode="rb")) #
    state_dict = {k:torch.tensor(v) for k, v in data.items()}
    resnetF.load_state_dict(state_dict)
    torch.save(resnetF.state_dict(), join(ckptdir, pklname.split(".")[0]+".pt"))
#%%
resnetF.load_state_dict(torch.load(join(ckptdir, pklname.split(".")[0]+".pt")))
