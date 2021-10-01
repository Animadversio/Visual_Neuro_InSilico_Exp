import sys
import torch
from os.path import join
simclr_path = r"D:\DL_Projects\Vision\SimCLRv2-Pytorch"
sys.path.append(simclr_path)
from resnet import get_resnet, name_to_params
from NN_timescale_lib import *
#%%
pth_path = join(simclr_path,"r50_1x_sk0_ema.pth")
model, _ = get_resnet(*name_to_params(pth_path))
model.load_state_dict(torch.load(pth_path)['resnet'])

# ResNet self-supervised
scorer = TorchScorer("resnet50")
scorer.model = model.cuda()
module_names, module_types, module_spec = get_module_names(scorer.model,[3,227,227],"cuda")
targetnames = [".net.0.1.ReLU1",
               ".net.1.blocks.BottleneckBlock0",
               ".net.1.blocks.BottleneckBlock2",
               ".net.2.blocks.BottleneckBlock0",
               ".net.2.blocks.BottleneckBlock2",
               ".net.3.blocks.BottleneckBlock0",
               ".net.3.blocks.BottleneckBlock2",
               ".net.3.blocks.BottleneckBlock4",
               ".net.4.blocks.BottleneckBlock0",
               ".net.4.blocks.BottleneckBlock2",
               ]
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=2000, run_frames=50000, seglen=1000, batch=120, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_ssl_run", savenm="resnet_ssl_run")
#%% ResNet self-supervised
model, _ = get_resnet(*name_to_params(pth_path))
model.load_state_dict(torch.load(pth_path)['resnet'])
scorer = TorchScorer("resnet50")
scorer.model = model.cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=2000, run_frames=50000, seglen=1000, batch=120, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_ssl_bike", savenm="resnet_ssl_bike")
#%% ResNet robust
from NN_timescale_lib import *
targetnames = [ ".ReLUrelu",
               ".layer1.Bottleneck0",
               ".layer1.Bottleneck2",
               ".layer2.Bottleneck0",
               ".layer2.Bottleneck2",
               ".layer3.Bottleneck0",
               ".layer3.Bottleneck2",
               ".layer3.Bottleneck4",
               ".layer4.Bottleneck0",
               ".layer4.Bottleneck2",
               ".Linearfc"
               ]
scorer = TorchScorer("resnet50_linf_8")
# scorer.model = model.cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_linf8_bike", savenm="resnet_linf8_bike")

scorer = TorchScorer("resnet50_linf_8")
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_linf8_run", savenm="resnet_linf8_run")

#%% VGG16
from NN_timescale_lib import *
targetnames = [".features.ReLU1",
               ".features.ReLU3",
               ".features.ReLU8",
               ".features.ReLU15",
               ".features.ReLU22",
               ".features.ReLU29",
               ".classifier.Linear0",
               ".classifier.Linear3",
               ".classifier.Linear6"
               ]
scorer = TorchScorer("vgg16")
# scorer.model = model.cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=60, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\vgg16_bike", savenm="vgg16_linf8_bike")

scorer = TorchScorer("vgg16")
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=60, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\vgg16_run", savenm="vgg16_linf8_run")
#%% CorNet-s
from NN_timescale_lib import *
scorer = TorchScorer("cornet_s")
targetnames = [ ".V1.ReLUnonlin1",
                ".V1.ReLUnonlin2",
                ".V2.Conv2dconv_input",
                ".CORblock_SV2",
                ".V4.Conv2dconv_input",
                ".CORblock_SV4",
                ".IT.Conv2dconv_input",
                ".CORblock_SIT",
                ".decoder.Linearlinear",
               ]
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\cornet_run", savenm="cornet_run")

scorer = TorchScorer("cornet_s")
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\cornet_bike", savenm="cornet_bike")

#%% ResNet Trained with DINO
targetnames = [ ".ReLUrelu",
               ".layer1.Bottleneck0",
               ".layer1.Bottleneck2",
               ".layer2.Bottleneck0",
               ".layer2.Bottleneck2",
               ".layer3.Bottleneck0",
               ".layer3.Bottleneck2",
               ".layer3.Bottleneck4",
               ".layer4.Bottleneck0",
               ".layer4.Bottleneck2",
               ".Identityfc"
               ]

scorer = TorchScorer("resnet50")
scorer.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet50_dino_run", savenm="resnet50_dino_run")

scorer = TorchScorer("resnet50")
scorer.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=120, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet50_dino_bike", savenm="resnet50_dino_bike")

#%% Visual Transformer.
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
targetnames = [".blocks.Block0",
               ".blocks.Block1",
               ".blocks.Block7",
               ".blocks.Block11"]
scorer = TorchScorer("resnet50")
del scorer.model
scorer.model = vitb8.cuda()
#%%
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=1000, run_frames=50000, seglen=1000, batch=40, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\vitB8_dino_bike", savenm="vitB8_dino_bike")
