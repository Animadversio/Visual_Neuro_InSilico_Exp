#!/bin/sh

#PBS -N insilico_manifold_allChan_RFfit_DenseNet
#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb
#PBS -m be
#PBS -q dque
#PBS -t 1-46

export TORCH_HOME="/scratch/binxu/torch"

param_list='--units densenet169 .features.ReLUrelu0 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 64
--units densenet169 .features._DenseBlockdenseblock1 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 128
--units densenet169 .features._DenseBlockdenseblock1 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 128 256
--units densenet169 .features.transition1.Conv2dconv 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 128
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit --chan_rng 0 128
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit --chan_rng 128 256
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit --chan_rng 256 384
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit --chan_rng 384 512
--units densenet169 .features.transition2.Conv2dconv 5 14 14 --imgsize 85 85 --corner 73 72 --RFfit --chan_rng 0 128
--units densenet169 .features.transition2.Conv2dconv 5 14 14 --imgsize 85 85 --corner 73 72 --RFfit --chan_rng 128 256
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 0 128
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 128 256
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 256 384
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 384 512
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 512 640
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 640 768
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 768 896
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 896 1024
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 1024 1152
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 1152 1280
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 0 128
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 128 256
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 256 384
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 384 512
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 512 640
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 0 128
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 128 256
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 256 384
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 384 512
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 512 640
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 640 768
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 768 896
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 896 1024
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 1024 1152
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 1152 1280
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 1280 1408
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 1408 1536
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 1536 1664
--units densenet169 .Linearclassifier 5  --chan_rng 0 128
--units densenet169 .Linearclassifier 5  --chan_rng 128 256
--units densenet169 .Linearclassifier 5  --chan_rng 256 384
--units densenet169 .Linearclassifier 5  --chan_rng 384 512
--units densenet169 .Linearclassifier 5  --chan_rng 512 640
--units densenet169 .Linearclassifier 5  --chan_rng 640 768
--units densenet169 .Linearclassifier 5  --chan_rng 768 896
--units densenet169 .Linearclassifier 5  --chan_rng 896 1000'

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



# {"conv2": "conv2 5 112 112 --imgsize 5 5 --corner 110 110 --RFfit",
# "conv3": "conv3 5 56 56 --imgsize 10 10 --corner 108 108 --RFfit",
# "conv4": "conv4 5 56 56 --imgsize 14 14 --corner 106 106 --RFfit",
# "conv5": "conv5 5 28 28 --imgsize 24 24 --corner 102 102 --RFfit",
# "conv6": "conv6 5 28 28 --imgsize 31 31 --corner 99 98 --RFfit",
# "conv7": "conv7 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit",
# "conv9": "conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit",
# "conv10": "conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit",
# "conv12": "conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit",
# "conv13": "conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit",
# "fc1": "fc1 1", 
# "fc2": "fc2 1", 
# "fc3": "fc3 1", }