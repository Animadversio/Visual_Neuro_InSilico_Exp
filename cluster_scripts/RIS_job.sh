#!/bin/bash
echo "$LSB_JOBINDEX"

param_list='units=("resnet50_linf_8", ".ReLUrelu", 5, 57, 57); Xlim=(111, 118); Ylim=(111, 118); imgsize=(7, 7); corner=(111, 111); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer1.Bottleneck1", 5, 28, 28); Xlim=(101, 124); Ylim=(101, 124); imgsize=(23, 23); corner=(101, 101); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer2.Bottleneck0", 5, 14, 14); Xlim=(99, 128); Ylim=(99, 128); imgsize=(29, 29); corner=(99, 99); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer2.Bottleneck2", 5, 14, 14); Xlim=(89, 138); Ylim=(90, 139); imgsize=(49, 49); corner=(89, 90); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer3.Bottleneck0", 5, 7, 7); Xlim=(77, 152); Ylim=(78, 153); imgsize=(75, 75); corner=(77, 78); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer3.Bottleneck2", 5, 7, 7); Xlim=(47, 184); Ylim=(47, 184); imgsize=(137, 137); corner=(47, 47); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer3.Bottleneck4", 5, 7, 7); Xlim=(25, 210); Ylim=(27, 212); imgsize=(185, 185); corner=(25, 27); RFfit=True; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer4.Bottleneck0", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False; chan_rng=(0, 75);
units=("resnet50_linf_8", ".layer4.Bottleneck2", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False; chan_rng=(0, 75);
units=("resnet50_linf_8", ".Linearfc", 5); chan_rng=(0, 75);
units=("resnet50", ".ReLUrelu", 5, 57, 57); Xlim=(111, 118); Ylim=(111, 118); imgsize=(7, 7); corner=(111, 111); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer1.Bottleneck1", 5, 28, 28); Xlim=(101, 124); Ylim=(101, 124); imgsize=(23, 23); corner=(101, 101); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer2.Bottleneck0", 5, 14, 14); Xlim=(99, 128); Ylim=(99, 128); imgsize=(29, 29); corner=(99, 99); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer2.Bottleneck2", 5, 14, 14); Xlim=(89, 138); Ylim=(90, 139); imgsize=(49, 49); corner=(89, 90); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer3.Bottleneck0", 5, 7, 7); Xlim=(77, 152); Ylim=(78, 153); imgsize=(75, 75); corner=(77, 78); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer3.Bottleneck2", 5, 7, 7); Xlim=(47, 184); Ylim=(47, 184); imgsize=(137, 137); corner=(47, 47); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer3.Bottleneck4", 5, 7, 7); Xlim=(25, 210); Ylim=(27, 212); imgsize=(185, 185); corner=(25, 27); RFfit=True; chan_rng=(0, 75);
units=("resnet50", ".layer4.Bottleneck0", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False; chan_rng=(0, 75);
units=("resnet50", ".layer4.Bottleneck2", 5, 4, 4); Xlim=(0, 227); Ylim=(0, 227); imgsize=(227, 227); corner=(0, 0); RFfit=False; chan_rng=(0, 75);
units=("resnet50", ".Linearfc", 5); chan_rng=(0, 75);
'


export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
echo "$unit_name" 
# python insilico_ResizeManifold_torch_efficient_script.py  $unit_name # the real job