from os.path import join
import torchvision, torch
from torchvision.models import vgg16
from collections import OrderedDict

vggnet = vgg16(True)
vggnet.cuda().eval()
for param in vggnet.parameters():
    param.requires_grad_(False)
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
module_names, module_types, module_spec = get_module_names(vggnet, input_size=(3, 227, 227))  # new version of layername
#%%
origname = layername_dict["vgg16"]
orig2std_map = OrderedDict()
std2orig_map = OrderedDict()
for i, name in enumerate(origname):
    if i < 31:
        orig2std_map[name] = module_names[str(i + 1)]
        std2orig_map[module_names[str(i + 1)]] = name
    else:
        orig2std_map[name] = module_names[str(i + 2)]
        std2orig_map[module_names[str(i + 2)]] = name


#%%
unit_list = [("vgg16", "conv1", 5, 112, 112),
            ("vgg16", "conv2", 5, 112, 112),
            ("vgg16", "conv3", 5, 56, 56),
            ("vgg16", "conv4", 5, 56, 56),
            ("vgg16", "conv5", 5, 28, 28),
            ("vgg16", "conv6", 5, 28, 28),
            ("vgg16", "conv7", 5, 28, 28),
            ("vgg16", "conv9", 5, 14, 14),
            ("vgg16", "conv10", 5, 14, 14),
            ("vgg16", "conv12", 5, 7, 7),
            ("vgg16", "conv13", 5, 7, 7),
            ("vgg16", "fc1", 1),
            ("vgg16", "fc2", 1),
            ("vgg16", "fc3", 1), ]

for unit in unit_list[:-3]:
    print("Unit %s" % (unit,))
    gradAmpmap = grad_RF_estimate(vggnet, orig2std_map[unit[1]], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                  device="cuda", show=True, reps=40, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    corner = (Xlim[0], Ylim[0])
    print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
        Xlim, Ylim, imgsize, corner))
    print(unit[1], unit[2], unit[3], unit[4], "--imgsize", imgsize[0], imgsize[1], "--corner", corner[0], corner[1],
          "--RFfit")
#%%
unit_list = [("vgg16", "conv2", 5, 112, 112),
            ("vgg16", "conv3", 5, 56, 56),
            ("vgg16", "conv4", 5, 56, 56),
            ("vgg16", "conv5", 5, 28, 28),
            ("vgg16", "conv6", 5, 28, 28),
            ("vgg16", "conv7", 5, 28, 28),
            ("vgg16", "conv9", 5, 14, 14),
            ("vgg16", "conv10", 5, 14, 14),
            ("vgg16", "conv12", 5, 7, 7),
            ("vgg16", "conv13", 5, 7, 7),
            ("vgg16", "fc1", 1),
            ("vgg16", "fc2", 1),
            ("vgg16", "fc3", 1), ]

commandstr = {"conv2": "conv2 5 112 112 --imgsize 5 5 --corner 110 110 --RFfit",
"conv3": "conv3 5 56 56 --imgsize 10 10 --corner 108 108 --RFfit",
"conv4": "conv4 5 56 56 --imgsize 14 14 --corner 106 106 --RFfit",
"conv5": "conv5 5 28 28 --imgsize 24 24 --corner 102 102 --RFfit",
"conv6": "conv6 5 28 28 --imgsize 31 31 --corner 99 98 --RFfit",
"conv7": "conv7 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit",
"conv9": "conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit",
"conv10": "conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit",
"conv12": "conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit",
"conv13": "conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit",
"fc1": "fc1 1",
"fc2": "fc2 1",
"fc3": "fc3 1", }

netname = "vgg16"
batchN = 128
taskN = 0
inv_map = {v: k for k, v in module_names.items()}
for unit in unit_list:
    # print(unit[1], module_spec[inv_map[unit[1]]]['outshape'])
    outshape = module_spec[inv_map[orig2std_map[unit[1]]]]['outshape']
    chanN = outshape[0]
    csr = 0
    while csr < chanN:
        csrend = min(chanN, csr + batchN)
        print("--units", netname, commandstr[unit[1]], "--chan_rng", csr, csrend)
        csr = csrend
        taskN += 1
print("num of task %d"%taskN)