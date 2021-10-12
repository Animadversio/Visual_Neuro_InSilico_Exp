
from os.path import join
import torchvision, torch
from torchvision.models import densenet161, densenet169
from collections import OrderedDict
import numpy as np
dnet = densenet169(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(dnet, input_size=(3, 227, 227))
#%%
# resnet101 = torchvision.models.resnet101(pretrained=True)
# rf_dict = receptive_field(vgg16.features, (3, 227, 227), device="cpu")

#%%
from grad_RF_estim import gradmap2RF_square, grad_RF_estimate
dnet.cuda().eval()
for param in dnet.parameters():
    param.requires_grad_(False)
unit_list = [("densenet169", ".features.ReLUrelu0", 5, 57, 57),
             ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28),
             ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28),
             ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14),
             ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14),
             ("densenet169", ".features._DenseBlockdenseblock3", 5, 7, 7),
             ("densenet169", ".features.transition3.Conv2dconv", 5, 7, 7),
             ("densenet169", ".features._DenseBlockdenseblock4", 5, 3, 3), 
             ("densenet169", ".Linearclassifier", 5), ]

commandstr = {}
for unit in unit_list[:]:
    print("Unit %s" % (unit,))
    module_id = unit[1]
    if len(unit) == 3:
        commandstr[module_id] = "%s %d " % (module_id, unit[2])
    elif len(unit) == 5:
        gradAmpmap = grad_RF_estimate(dnet, module_id, (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                      device="cuda", show=True, reps=40, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
        corner = (Xlim[0], Ylim[0])
        commandstr[module_id] = "%s %d %d %d --imgsize %d %d --corner %d %d --RFfit" % \
                                (module_id, unit[2], unit[3], unit[4], *imgsize, *corner)
        print("Xlim %s Ylim %s\nimgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))
#%% Newer version command line interface
commandstr = \
    {'.features.ReLUrelu0': '.features.ReLUrelu0 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit',
     '.features._DenseBlockdenseblock1': '.features._DenseBlockdenseblock1 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit',
     '.features.transition1.Conv2dconv': '.features.transition1.Conv2dconv 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit',
     '.features._DenseBlockdenseblock2': '.features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit',
     '.features.transition2.Conv2dconv': '.features.transition2.Conv2dconv 5 14 14 --imgsize 85 85 --corner 73 72 --RFfit',
     '.features._DenseBlockdenseblock3': '.features._DenseBlockdenseblock3 5 7 7 ',
     '.features.transition3.Conv2dconv': '.features.transition3.Conv2dconv 5 7 7 ',
     '.features._DenseBlockdenseblock4': '.features._DenseBlockdenseblock4 5 3 3 ',
     '.Linearclassifier': '.Linearclassifier 5 '}
netname = "densenet169"
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
#%% Older version command line interface
taskN = 0
for unit in unit_list:
    outshape = module_spec[inv_map[unit[1]]]['outshape']
    chanN = outshape[0]
    csr = 0
    print("--units", netname, commandstr[unit[1]], "--chan_rng", csr, min(75, chanN))
    # while csr < chanN:
    #     csrend = min(chanN, csr + batchN)
    #     print('units=("%s", ".layer1.Bottleneck1", 5, 28, 28); imgsize=(23, 23); corner=(101, 101); RFfit=True; '
    #           'chan_rng=(0, 75);', netname, commandstr[unit[1]], csr, csrend)
    #     csr = csrend
    taskN += 1
print("num of task %d" % taskN)
