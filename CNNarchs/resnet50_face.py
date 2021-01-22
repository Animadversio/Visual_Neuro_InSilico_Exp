from os.path import join
import torchvision, torch
from torchvision.models import resnet50
from collections import OrderedDict
ckptdir = r"E:\Cluster_Backup\torch"

resnetF = resnet50(False, num_classes=8631)  # Face recognition Network's output is 8631 classes in VGG-Face Dataset.
#%%
import pickle as pkl
for pklname in ["resnet50_ft_weight.pkl", "resnet50_scratch_weight.pkl"]:
    data = pkl.load(open(join(ckptdir, pklname), mode="rb")) #
    state_dict = {k:torch.tensor(v) for k, v in data.items()}
    resnetF.load_state_dict(state_dict)
    torch.save(resnetF.state_dict(), join(ckptdir, pklname.split(".")[0]+".pt"))
#%%
ptname = "resnet50_scratch_weight.pt" #["resnet50_ft_weight.pkl", "resnet50_scratch_weight.pkl"]
resnetF.load_state_dict(torch.load(join(ckptdir, ptname)))
#%%
from grad_RF_estim import gradmap2RF_square, grad_RF_estimate
resnetF.cuda().eval()
for param in resnetF.parameters():
    param.requires_grad_(False)
unit_list = [("resnet50-face_scratch", ".ReLUrelu", 5, 57, 57),
            ("resnet50-face_scratch", ".layer1.Bottleneck1", 5, 28, 28),
            ("resnet50-face_scratch", ".layer2.Bottleneck0", 5, 14, 14),
            ("resnet50-face_scratch", ".layer2.Bottleneck2", 5, 14, 14),
            ("resnet50-face_scratch", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50-face_scratch", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50-face_scratch", ".layer3.Bottleneck4", 5, 7, 7),
            ("resnet50-face_scratch", ".layer4.Bottleneck0", 5, 4, 4),
            ("resnet50-face_scratch", ".layer4.Bottleneck2", 5, 4, 4), ]


for unit in unit_list:
    print("Unit %s" % (unit,))
    gradAmpmap = grad_RF_estimate(resnetF, unit[1], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                  device="cuda", show=True, reps=40, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
        Xlim, Ylim, (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0]), (Xlim[0], Ylim[0])))
#%%
