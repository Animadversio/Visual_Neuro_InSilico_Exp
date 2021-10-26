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
from layer_hook_utils import get_module_names
module_names, module_types, module_spec = get_module_names(resnetF, input_size=(3, 227, 227), device="cuda")
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

commandstr = {}
for unit in unit_list:
    print("Unit %s" % (unit,))
    module_id = unit[1]
    gradAmpmap = grad_RF_estimate(resnetF, module_id, (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                  device="cuda", show=True, reps=40, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    corner = (Xlim[0], Ylim[0])
    # print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
    #     Xlim, Ylim, imgsize, corner))
    if len(unit) == 5:
        commandstr[module_id] = "%s %d %d %d --imgsize %d %d --corner %d %d --RFfit"%\
                            (module_id, unit[2], unit[3], unit[4], *imgsize, *corner)
    elif len(unit) == 3:
        commandstr[module_id] = "%s %d " % \
                                (module_id, unit[2])
    print(commandstr[module_id])
#%%
unit_list = [("resnet50", ".ReLUrelu", 5, 57, 57),
            ("resnet50", ".layer1.Bottleneck1", 5, 28, 28),
            ("resnet50", ".layer2.Bottleneck0", 5, 14, 14),
            ("resnet50", ".layer2.Bottleneck2", 5, 14, 14),
            ("resnet50", ".layer3.Bottleneck0", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck2", 5, 7, 7),
            ("resnet50", ".layer3.Bottleneck4", 5, 7, 7),
            ("resnet50", ".layer4.Bottleneck0", 5, 4, 4),
            ("resnet50", ".layer4.Bottleneck2", 5, 4, 4),
            ("resnet50", ".Linearfc", 5, ), ]

commandstr = {'.ReLUrelu': '.ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit',
 '.layer1.Bottleneck1': '.layer1.Bottleneck1 5 28 28 --imgsize 27 27 --corner 99 99 --RFfit',
 '.layer2.Bottleneck0': '.layer2.Bottleneck0 5 14 14 --imgsize 39 39 --corner 95 94 --RFfit',
 '.layer2.Bottleneck2': '.layer2.Bottleneck2 5 14 14 --imgsize 67 67 --corner 79 81 --RFfit',
 '.layer3.Bottleneck0': '.layer3.Bottleneck0 5 7 7 --imgsize 81 81 --corner 71 76 --RFfit',
 '.layer3.Bottleneck2': '.layer3.Bottleneck2 5 7 7 --imgsize 129 129 --corner 49 51 --RFfit',
 '.layer3.Bottleneck4': '.layer3.Bottleneck4 5 7 7 --imgsize 177 177 --corner 27 30 --RFfit',
 '.layer4.Bottleneck0': '.layer4.Bottleneck0 5 4 4 --imgsize 186 186 --corner 41 39 --RFfit',
 '.layer4.Bottleneck2': '.layer4.Bottleneck2 5 4 4',
 '.Linearfc': '.Linearfc 5', }
# ResNet Face's RF seems different from ResNet Robust.
netname = "resnet50-face_scratch"
taskN = 0
batchN = 128
inv_map = {v: k for k,v in module_names.items()}
for unit in unit_list:
    # print(unit[1], module_spec[inv_map[unit[1]]]['outshape'])
    outshape = module_spec[inv_map[unit[1]]]['outshape']
    chanN = outshape[0]
    csr = 0
    while csr < chanN:
        csrend = min(chanN, csr + batchN)
        print("--units", netname, commandstr[unit[1]], "--chan_rng", csr, csrend)
        csr = csrend
        taskN += 1
print("num of task %d"%taskN)