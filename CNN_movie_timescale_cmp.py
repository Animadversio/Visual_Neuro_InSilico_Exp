import sys
import torch
from os.path import join
simclr_path = r"D:\DL_Projects\Vision\SimCLRv2-Pytorch"
sys.path.append(simclr_path)
from resnet import get_resnet, name_to_params
pth_path = join(simclr_path,"r50_1x_sk0_ema.pth")
model, _ = get_resnet(*name_to_params(pth_path))
model.load_state_dict(torch.load(pth_path)['resnet'])
from NN_timescale_lib import *
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
        popsize=500, run_frames=50000, seglen=1000, batch=120, video_id="goprorun",
        video_path=r"E:\Datasets\POVfootage\GoPro Hero 8 Running Footage with Head Mount.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_ssl_run", savenm="resnet_ssl_run")
#%%
model, _ = get_resnet(*name_to_params(pth_path))
model.load_state_dict(torch.load(pth_path)['resnet'])
scorer = TorchScorer("resnet50")
scorer.model = model.cuda()
recordings_all, acf_arr_dict, df, unit_mask_dict, figh, figh2 = \
        timescale_analysis_pipeline(scorer, targetnames,
        popsize=2000, run_frames=50000, seglen=1000, batch=120, video_id="goprobike",
        video_path=r"E:\Datasets\POVfootage\One Hour of Beautiful MTB POV Footage _ The Loam Ranger.mp4",
        savedir=r"E:\insilico_exps\CNN_timescale\resnet_ssl_bike", savenm="resnet_ssl_bike")


