import torchvision, torch
from torchvision.models import AlexNet, alexnet
Anet = alexnet(True)
#%%
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
modulenames, moduletypes, module_spec = get_module_names(Anet, input_size=(3, 227, 227), device="cpu")
#%%
from torch_net_utils import receptive_field, receptive_field_for_unit
#%% Get receptive field
rfdict = receptive_field(Anet.features, (3,227,227), device="cpu")
receptive_field_for_unit(rfdict, "2", (28, 28))
receptive_field_for_unit(rfdict, "5", (13, 13))
receptive_field_for_unit(rfdict, "8", (6, 6))
receptive_field_for_unit(rfdict, "10", (6, 6))
receptive_field_for_unit(rfdict, "12", (6, 6))
#%%
unit_list = [("alexnet", "conv1_relu", 5, 28, 28),
            ("alexnet", "conv2_relu", 5, 13, 13),
            ("alexnet", "conv3_relu", 5, 6, 6),
            ("alexnet", "conv4_relu", 5, 6, 6),
            ("alexnet", "conv5_relu", 5, 6, 6),
            ("alexnet", "fc6", 5),
            ("alexnet", "fc7", 5),
            ("alexnet", "fc8", 5),]

commandstr = \
    {"conv1_relu": '--units alexnet conv1_relu 5 28 28 --corner 110 110  --imgsize  11 11 --RFfit ',
     "conv2_relu": '--units alexnet conv2_relu 5 13 13 --corner 86 86  --imgsize  51 51 --RFfit ',
     "conv3_relu": '--units alexnet conv3_relu 5 6 6 --corner 62 62  --imgsize  99 99 --RFfit ',
     "conv4_relu": '--units alexnet conv4_relu 5 6 6 --corner 46 46  --imgsize  131 131 --RFfit ',
     "conv5_relu": '--units alexnet conv5_relu 5 6 6 --corner 30 30  --imgsize  163 163 --RFfit',
     "fc6": '--units alexnet fc6 5',
     "fc7": '--units alexnet fc7 5',
     "fc8": '--units alexnet fc8 5', }
inv_map = \
    {"conv1_relu": '2',
     "conv2_relu": '5',
     "conv3_relu": '8',
     "conv4_relu": '10',
     "conv5_relu": '12',
     "fc6": '16',
     "fc7": '19',
     "fc8": '21', }
netname = "alexnet"
taskN = 0
batchN = 128
# inv_map = {v: k for k,v in enumerate(layername_dict['alexnet'])}
for unit in unit_list:
    # print(unit[1], module_spec[inv_map[unit[1]]]['outshape'])
    outshape = module_spec[inv_map[unit[1]]]['outshape']
    chanN = outshape[0]
    csr = 0
    while csr < chanN:
        csrend = min(chanN, csr + batchN)
        print(commandstr[unit[1]], "--chan_rng", csr, csrend)# "--units", netname,
        csr = csrend
        taskN += 1
print("num of task %d"%taskN)