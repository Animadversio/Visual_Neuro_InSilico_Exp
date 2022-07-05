"""
This is a collection of configurations for insilico_manifold experiments
    Used to load up images and data and see RFs.
"""


RN50_config = {
            '.ReLUrelu': {"layer": '.ReLUrelu', "unit_pos": [57, 57], "imgsize": [7, 7], "corner": [111, 111], "RFfit": True,},
            '.layer1.Bottleneck1': {"layer": '.layer1.Bottleneck1', "unit_pos": [28, 28], "imgsize": [23, 23], "corner": [101, 101], "RFfit": True,},
            '.layer2.Bottleneck0': {"layer": '.layer2.Bottleneck0', "unit_pos": [14, 14], "imgsize": [29, 29], "corner": [99, 99], "RFfit": True,},
            '.layer2.Bottleneck2': {"layer": '.layer2.Bottleneck2', "unit_pos": [14, 14], "imgsize": [49, 49], "corner": [89, 90], "RFfit": True,},
            '.layer3.Bottleneck0': {"layer": '.layer3.Bottleneck0', "unit_pos": [7, 7], "imgsize": [75, 75], "corner": [77, 78], "RFfit": True,},
            '.layer3.Bottleneck2': {"layer": '.layer3.Bottleneck2', "unit_pos": [7, 7], "imgsize": [137, 137], "corner": [47, 47], "RFfit": True,},
            '.layer3.Bottleneck4': {"layer": '.layer3.Bottleneck4', "unit_pos": [7, 7], "imgsize": [185, 185], "corner": [25, 27], "RFfit": True,},
            '.layer4.Bottleneck0': {"layer": '.layer4.Bottleneck0', "unit_pos": [4, 4], "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
            '.layer4.Bottleneck2': {"layer": '.layer4.Bottleneck2', "unit_pos": [4, 4], "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
            '.Linearfc': {"layer": '.Linearfc', "unit_pos": None, "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    }

VGG16_config = {
    "conv2": {"layer": "conv2", "unit_pos": (112, 112), "imgsize": (5, 5), "corner": (110, 110), "RFfit": True, },
    "conv3": {"layer": "conv3", "unit_pos": (56, 56), "imgsize": (10, 10), "corner": (108, 108), "RFfit": True, },
    "conv4": {"layer": "conv4", "unit_pos": (56, 56), "imgsize": (14, 14), "corner": (106, 106), "RFfit": True, },
    "conv5": {"layer": "conv5", "unit_pos": (28, 28), "imgsize": (24, 24), "corner": (102, 102), "RFfit": True, },
    "conv6": {"layer": "conv6", "unit_pos": (28, 28), "imgsize": (31, 31), "corner": (99, 98), "RFfit": True, },
    "conv7": {"layer": "conv7", "unit_pos": (28, 28), "imgsize": (37, 37), "corner": (95, 95), "RFfit": True, },
    "conv9": {"layer": "conv9", "unit_pos": (14, 14), "imgsize": (68, 68), "corner": (82, 82), "RFfit": True, },
    "conv10": {"layer": "conv10", "unit_pos": (14, 14), "imgsize": (82, 82), "corner": (75, 75), "RFfit": True, },
    "conv12": {"layer": "conv12", "unit_pos": (7, 7), "imgsize": (141, 141), "corner": (50, 49), "RFfit": True, },
    "conv13": {"layer": "conv13", "unit_pos": (7, 7), "imgsize": (169, 169), "corner": (36, 35), "RFfit": True, },
    "fc1": {"layer": "fc1", "unit_pos": None, "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    "fc2": {"layer": "fc2", "unit_pos": None, "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    "fc3": {"layer": "fc3", "unit_pos": None, "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    }

DenseNet169_config = {
    '.features.ReLUrelu0': {"layer": '.features.ReLUrelu0', "unit_pos": (57, 57), "imgsize": (7, 7), "corner": (111, 111), "RFfit": True, },
    '.features._DenseBlockdenseblock1': {"layer": '.features._DenseBlockdenseblock1', "unit_pos": (28, 28), "imgsize": (37, 37), "corner": (95, 95), "RFfit": True, },
    '.features.transition1.Conv2dconv': {"layer": '.features.transition1.Conv2dconv', "unit_pos": (28, 28), "imgsize": (37, 37), "corner": (95, 95), "RFfit": True, },
    '.features._DenseBlockdenseblock2': {"layer": '.features._DenseBlockdenseblock2', "unit_pos": (14, 14), "imgsize": (75, 75), "corner": (78, 78), "RFfit": True, },
    '.features.transition2.Conv2dconv': {"layer": '.features.transition2.Conv2dconv', "unit_pos": (14, 14), "imgsize": (85, 85), "corner": (73, 72), "RFfit": True, },
    '.features._DenseBlockdenseblock3': {"layer": '.features._DenseBlockdenseblock3', "unit_pos": (7, 7), "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    '.features.transition3.Conv2dconv': {"layer": '.features.transition3.Conv2dconv', "unit_pos": (7, 7), "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    '.features._DenseBlockdenseblock4': {"layer": '.features._DenseBlockdenseblock4', "unit_pos": (3, 3), "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
    '.Linearclassifier': {"layer": ".Linearclassifier", "unit_pos": None, "imgsize": [227, 227], "corner": [0, 0], "RFfit": False,},
}
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



def manifold_config(netname):
    if netname in ["resnet50", "resnet50_linf_8", "resnet50_linf8"]:
        return RN50_config
    elif netname in ["vgg16"]:
        return VGG16_config
    elif netname in ["densenet169"]:
        return DenseNet169_config
    else:
        raise ValueError("Unknown netname: {}".format(netname))

