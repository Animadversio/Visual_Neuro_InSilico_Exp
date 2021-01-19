from os.path import join
import torchvision, torch
from torchvision.models import resnet50
from collections import OrderedDict

resnetR = resnet50(False)
#%% Load in the pt
savedir = r"E:\Cluster_Backup\torch"
for ptname in ["imagenet_linf_8.pt", "imagenet_linf_4.pt", "imagenet_l2_3_0.pt"]:
    data = torch.load(join(savedir, ptname))
    # resnetR.load_state_dict(data["model"])
    #% Load in the state dict
    assert len(resnetR.state_dict()) == len([name for name in data["model"] if "module.model." in name])
    state_dict_pure = OrderedDict()
    for name in data["model"]:
        if "module.model." in name:
            i = name.find("module.model.")
            newname = name[i+13:]
            state_dict_pure[newname] = data["model"][name]
        else:
            continue
    resnetR.load_state_dict(state_dict_pure)
    torch.save(resnetR.state_dict(), join(savedir, ptname.split(".")[0]+"_pure.pt"))

#%%
with torch.no_grad():
    resnetR(torch.randn(1, 3, 227, 227))
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(resnetR, input_size=(3, 227, 227))
#%%
resnetR = resnet50(False)
ptname = "imagenet_linf_8.pt" #in ["imagenet_linf_8.pt", "imagenet_linf_4.pt", "imagenet_l2_3_0.pt"]
resnetR.load_state_dict(torch.load(join(savedir, ptname.split(".")[0]+"_pure.pt")))
#%%
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57),
("resnet50", ".layer1.Bottleneck1", 5, 28, 28),
("resnet50", ".layer2.Bottleneck0", 5, 14, 14),
("resnet50", ".layer2.Bottleneck2", 5, 14, 14),
("resnet50", ".layer3.Bottleneck0", 5, 7, 7),
("resnet50", ".layer3.Bottleneck2", 5, 7, 7),
("resnet50", ".layer3.Bottleneck4", 5, 7, 7),
("resnet50", ".layer4.Bottleneck0", 5, 4, 4),
("resnet50", ".layer4.Bottleneck2", 5, 4, 4), ]
resnetR.cuda().eval()
for param in resnetR.parameters():
    param.requires_grad_(False)

for unit in unit_list:
    print("Unit %s" % (unit,))
    gradAmpmap = grad_RF_estimate(resnetR, unit[1], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                  device="cuda", show=True, reps=40, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
        Xlim, Ylim, (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0]), (Xlim[0], Ylim[0])))

#%%
# units=("resnet50", ".ReLUrelu", 5, 57, 57); Xlim=(111, 118); Ylim=(111, 118); imgsize=(7, 7); corner=(111, 111); RFfit=True;
# units=("resnet50", ".layer1.Bottleneck1", 5, 28, 28); Xlim=(101, 124); Ylim=(101, 124); imgsize=(23, 23); corner=(101, 101); RFfit=True;
# units=("resnet50", ".layer2.Bottleneck0", 5, 14, 14); Xlim=(99, 128); Ylim=(99, 128); imgsize=(29, 29); corner=(99, 99); RFfit=True;
# units=("resnet50", ".layer2.Bottleneck2", 5, 14, 14); Xlim=(89, 138); Ylim=(90, 139); imgsize=(49, 49); corner=(89, 90); RFfit=True;
# units=("resnet50", ".layer3.Bottleneck0", 5, 7, 7); Xlim=(77, 152); Ylim=(78, 153); imgsize=(75, 75); corner=(77, 78); RFfit=True;
# units=("resnet50", ".layer3.Bottleneck2", 5, 7, 7); Xlim=(47, 184); Ylim=(47, 184); imgsize=(137, 137); corner=(47, 47); RFfit=True;
# units=("resnet50", ".layer3.Bottleneck4", 5, 7, 7); Xlim=(25, 210); Ylim=(27, 212); imgsize=(185, 185); corner=(25, 27); RFfit=True;
# units=("resnet50", ".layer4.Bottleneck0", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False;
# units=("resnet50", ".layer4.Bottleneck2", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False;
# units=("resnet50", ".Linearfc", 5); RFfit=False;