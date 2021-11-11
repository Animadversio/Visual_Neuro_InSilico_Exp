from cornet import cornet_s, cornet_z, cornet_rt
import torch
Cnet = cornet_s(pretrained=True) # 408Mb
# cornet_z(True)  # 15.8Mb
# cornet_rt(True)  # 39.8Mb
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names
module_names, module_types, module_spec = get_module_names(Cnet.module, input_size=(3, 227, 227), device="cuda")
#%%
from grad_RF_estim import gradmap2RF_square, grad_RF_estimate
Cnet_m = Cnet.module
#%%
Cnet_m.cuda().eval()
for param in Cnet_m.parameters():
    param.requires_grad_(False)
unit_list = [("Cornet_s", ".V1.ReLUnonlin1", 5, 57, 57),
            ("Cornet_s", ".V1.ReLUnonlin2", 5, 28, 28),
            ("Cornet_s", ".V2.Conv2dconv_input", 5, 28, 28),
            ("Cornet_s", ".CORblock_SV2", 5, 14, 14),
            ("Cornet_s", ".V4.Conv2dconv_input", 5, 14, 14),
            ("Cornet_s", ".CORblock_SV4", 5, 7, 7),
            ("Cornet_s", ".IT.Conv2dconv_input", 5, 7, 7),
            ("Cornet_s", ".CORblock_SIT", 5, 3, 3),
            ("Cornet_s", ".decoder.Linearlinear", 5,), ]

commandstr = {}
for unit in unit_list:
    print("Unit %s" % (unit,))
    module_id = unit[1]
    if len(unit) == 3:
        commandstr[module_id] = "%s %d " % (module_id, unit[2])
    elif len(unit) == 5:
        gradAmpmap = grad_RF_estimate(Cnet_m, unit[1], (slice(None), unit[3], unit[4]), input_size=(3, 227, 227),
                                      device="cuda", show=True, reps=40, batch=1)
        Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
        imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
        corner = (Xlim[0], Ylim[0])
        commandstr[module_id] = "%s %d %d %d --imgsize %d %d --corner %d %d --RFfit" % \
                                (module_id, unit[2], unit[3], unit[4], *imgsize, *corner)
        print("Xlim %s Ylim %s\nimgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))

#%%
commandstr = {'.V1.ReLUnonlin1': '.V1.ReLUnonlin1 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit',
             '.V1.ReLUnonlin2': '.V1.ReLUnonlin2 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit',
             '.V2.Conv2dconv_input': '.V2.Conv2dconv_input 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit',
             '.CORblock_SV2': '.CORblock_SV2 5 14 14 --imgsize 42 42 --corner 92 93 --RFfit',
             '.V4.Conv2dconv_input': '.V4.Conv2dconv_input 5 14 14 --imgsize 43 43 --corner 91 91 --RFfit',
             '.CORblock_SV4': '.CORblock_SV4 5 7 7 --imgsize 144 144 --corner 42 43 --RFfit',
             '.IT.Conv2dconv_input': '.IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit',
             '.CORblock_SIT': '.CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit',
             '.decoder.Linearlinear': '.decoder.Linearlinear 5 '}
#%%
netname = "Cornet_s"
taskN = 0
batchN = 128
inv_map = {v: k for k, v in module_names.items()}
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