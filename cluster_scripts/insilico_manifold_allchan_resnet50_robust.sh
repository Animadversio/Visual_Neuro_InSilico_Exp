#!/bin/sh

#PBS -N insilico_manifold_allChan_RFfit_resnetRobust
#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb
#PBS -m be
#PBS -q dque
#PBS -t 1-74

export TORCH_HOME="/scratch/binxu/torch"

param_list='--units resnet50_linf_8 .ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 64
--units resnet50_linf_8 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit --chan_rng 256 384
--units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit --chan_rng 384 512
--units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit --chan_rng 256 384
--units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit --chan_rng 384 512
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 256 384
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 384 512
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 512 640
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 640 768
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 768 896
--units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit --chan_rng 896 1024
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 256 384
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 384 512
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 512 640
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 640 768
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 768 896
--units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit --chan_rng 896 1024
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 0 128
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 128 256
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 256 384
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 384 512
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 512 640
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 640 768
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 768 896
--units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit --chan_rng 896 1024
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 0 128
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 128 256
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 256 384
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 384 512
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 512 640
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 640 768
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 768 896
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 896 1024
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1024 1152
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1152 1280
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1280 1408
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1408 1536
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1536 1664
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1664 1792
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1792 1920
--units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1920 2048
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 0 128
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 128 256
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 256 384
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 384 512
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 512 640
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 640 768
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 768 896
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 896 1024
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1024 1152
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1152 1280
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1280 1408
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1408 1536
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1536 1664
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1664 1792
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1792 1920
--units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  --chan_rng 1920 2048
--units resnet50_linf_8 .Linearfc 5 --chan_rng 0 128
--units resnet50_linf_8 .Linearfc 5 --chan_rng 128 256
--units resnet50_linf_8 .Linearfc 5 --chan_rng 256 384
--units resnet50_linf_8 .Linearfc 5 --chan_rng 384 512
--units resnet50_linf_8 .Linearfc 5 --chan_rng 512 640
--units resnet50_linf_8 .Linearfc 5 --chan_rng 640 768
--units resnet50_linf_8 .Linearfc 5 --chan_rng 768 896
--units resnet50_linf_8 .Linearfc 5 --chan_rng 896 1000'

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
# export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`
# python_code_full=$unit_name$'\n'$python_code
# echo "$python_code_full"
# python -c "$python_code_full"

echo "$unit_name"
python insilico_ResizeManifold_torch_efficient_script.py  $unit_name


# --units resnet50_linf_8 .ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit 
# --units resnet50_linf_8 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101 --RFfit 
# --units resnet50_linf_8 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit 
# --units resnet50_linf_8 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit 
# --units resnet50_linf_8 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit 
# --units resnet50_linf_8 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit 
# --units resnet50_linf_8 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit 
# --units resnet50_linf_8 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  
# --units resnet50_linf_8 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  
# --units resnet50_linf_8 .Linearfc 5
# --units resnet50 .ReLUrelu 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit 
# --units resnet50 .layer1.Bottleneck1 5 28 28 --imgsize 23 23 --corner 101 101 --RFfit 
# --units resnet50 .layer2.Bottleneck0 5 14 14 --imgsize 29 29 --corner 99 99 --RFfit 
# --units resnet50 .layer2.Bottleneck2 5 14 14 --imgsize 49 49 --corner 89 90 --RFfit 
# --units resnet50 .layer3.Bottleneck0 5 7 7 --imgsize 75 75 --corner 77 78 --RFfit 
# --units resnet50 .layer3.Bottleneck2 5 7 7 --imgsize 137 137 --corner 47 47 --RFfit 
# --units resnet50 .layer3.Bottleneck4 5 7 7 --imgsize 185 185 --corner 25 27 --RFfit 
# --units resnet50 .layer4.Bottleneck0 5 4 4 --imgsize 227 227 --corner 0 0  
# --units resnet50 .layer4.Bottleneck2 5 4 4 --imgsize 227 227 --corner 0 0  
# --units resnet50 .Linearfc 5
