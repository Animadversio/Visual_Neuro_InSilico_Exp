#!/bin/sh

#PBS -N insilico_manifold_allChan_RFfit_VGG16
#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb
#PBS -m be
#PBS -q dque
#PBS -t 11,27

export TORCH_HOME="/scratch/binxu/torch"

param_list='--units vgg16 conv2 5 112 112 --imgsize 5 5 --corner 110 110 --RFfit --chan_rng 0 64
--units vgg16 conv3 5 56 56 --imgsize 10 10 --corner 108 108 --RFfit --chan_rng 0 128
--units vgg16 conv4 5 56 56 --imgsize 14 14 --corner 106 106 --RFfit --chan_rng 0 128
--units vgg16 conv5 5 28 28 --imgsize 24 24 --corner 102 102 --RFfit --chan_rng 0 128
--units vgg16 conv5 5 28 28 --imgsize 24 24 --corner 102 102 --RFfit --chan_rng 128 256
--units vgg16 conv6 5 28 28 --imgsize 31 31 --corner 99 98 --RFfit --chan_rng 0 128
--units vgg16 conv6 5 28 28 --imgsize 31 31 --corner 99 98 --RFfit --chan_rng 128 256
--units vgg16 conv7 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 128
--units vgg16 conv7 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 128 256
--units vgg16 conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit --chan_rng 0 128
--units vgg16 conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit --chan_rng 128 256
--units vgg16 conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit --chan_rng 256 384
--units vgg16 conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit --chan_rng 384 512
--units vgg16 conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit --chan_rng 0 128
--units vgg16 conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit --chan_rng 128 256
--units vgg16 conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit --chan_rng 256 384
--units vgg16 conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit --chan_rng 384 512
--units vgg16 conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit --chan_rng 0 128
--units vgg16 conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit --chan_rng 128 256
--units vgg16 conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit --chan_rng 256 384
--units vgg16 conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit --chan_rng 384 512
--units vgg16 conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit --chan_rng 0 128
--units vgg16 conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit --chan_rng 128 256
--units vgg16 conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit --chan_rng 256 384
--units vgg16 conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit --chan_rng 384 512
--units vgg16 fc1 1 --chan_rng 0 128
--units vgg16 fc1 1 --chan_rng 128 256
--units vgg16 fc1 1 --chan_rng 256 384
--units vgg16 fc1 1 --chan_rng 384 512
--units vgg16 fc1 1 --chan_rng 512 640
--units vgg16 fc1 1 --chan_rng 640 768
--units vgg16 fc1 1 --chan_rng 768 896
--units vgg16 fc1 1 --chan_rng 896 1024
--units vgg16 fc1 1 --chan_rng 1024 1152
--units vgg16 fc1 1 --chan_rng 1152 1280
--units vgg16 fc1 1 --chan_rng 1280 1408
--units vgg16 fc1 1 --chan_rng 1408 1536
--units vgg16 fc1 1 --chan_rng 1536 1664
--units vgg16 fc1 1 --chan_rng 1664 1792
--units vgg16 fc1 1 --chan_rng 1792 1920
--units vgg16 fc1 1 --chan_rng 1920 2048
--units vgg16 fc1 1 --chan_rng 2048 2176
--units vgg16 fc1 1 --chan_rng 2176 2304
--units vgg16 fc1 1 --chan_rng 2304 2432
--units vgg16 fc1 1 --chan_rng 2432 2560
--units vgg16 fc1 1 --chan_rng 2560 2688
--units vgg16 fc1 1 --chan_rng 2688 2816
--units vgg16 fc1 1 --chan_rng 2816 2944
--units vgg16 fc1 1 --chan_rng 2944 3072
--units vgg16 fc1 1 --chan_rng 3072 3200
--units vgg16 fc1 1 --chan_rng 3200 3328
--units vgg16 fc1 1 --chan_rng 3328 3456
--units vgg16 fc1 1 --chan_rng 3456 3584
--units vgg16 fc1 1 --chan_rng 3584 3712
--units vgg16 fc1 1 --chan_rng 3712 3840
--units vgg16 fc1 1 --chan_rng 3840 3968
--units vgg16 fc1 1 --chan_rng 3968 4096
--units vgg16 fc2 1 --chan_rng 0 128
--units vgg16 fc2 1 --chan_rng 128 256
--units vgg16 fc2 1 --chan_rng 256 384
--units vgg16 fc2 1 --chan_rng 384 512
--units vgg16 fc2 1 --chan_rng 512 640
--units vgg16 fc2 1 --chan_rng 640 768
--units vgg16 fc2 1 --chan_rng 768 896
--units vgg16 fc2 1 --chan_rng 896 1024
--units vgg16 fc2 1 --chan_rng 1024 1152
--units vgg16 fc2 1 --chan_rng 1152 1280
--units vgg16 fc2 1 --chan_rng 1280 1408
--units vgg16 fc2 1 --chan_rng 1408 1536
--units vgg16 fc2 1 --chan_rng 1536 1664
--units vgg16 fc2 1 --chan_rng 1664 1792
--units vgg16 fc2 1 --chan_rng 1792 1920
--units vgg16 fc2 1 --chan_rng 1920 2048
--units vgg16 fc2 1 --chan_rng 2048 2176
--units vgg16 fc2 1 --chan_rng 2176 2304
--units vgg16 fc2 1 --chan_rng 2304 2432
--units vgg16 fc2 1 --chan_rng 2432 2560
--units vgg16 fc2 1 --chan_rng 2560 2688
--units vgg16 fc2 1 --chan_rng 2688 2816
--units vgg16 fc2 1 --chan_rng 2816 2944
--units vgg16 fc2 1 --chan_rng 2944 3072
--units vgg16 fc2 1 --chan_rng 3072 3200
--units vgg16 fc2 1 --chan_rng 3200 3328
--units vgg16 fc2 1 --chan_rng 3328 3456
--units vgg16 fc2 1 --chan_rng 3456 3584
--units vgg16 fc2 1 --chan_rng 3584 3712
--units vgg16 fc2 1 --chan_rng 3712 3840
--units vgg16 fc2 1 --chan_rng 3840 3968
--units vgg16 fc2 1 --chan_rng 3968 4096
--units vgg16 fc3 1 --chan_rng 0 128
--units vgg16 fc3 1 --chan_rng 128 256
--units vgg16 fc3 1 --chan_rng 256 384
--units vgg16 fc3 1 --chan_rng 384 512
--units vgg16 fc3 1 --chan_rng 512 640
--units vgg16 fc3 1 --chan_rng 640 768
--units vgg16 fc3 1 --chan_rng 768 896
--units vgg16 fc3 1 --chan_rng 896 1000'

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