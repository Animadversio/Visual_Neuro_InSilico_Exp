#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_allchan_RFfit_AlexNet

#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-82

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='--units alexnet conv1_relu 5 28 28 --corner 110 110  --imgsize  11 11 --RFfit  --chan_rng 0 64
--units alexnet conv2_relu 5 13 13 --corner 86 86  --imgsize  51 51 --RFfit  --chan_rng 0 128
--units alexnet conv2_relu 5 13 13 --corner 86 86  --imgsize  51 51 --RFfit  --chan_rng 128 192
--units alexnet conv3_relu 5 6 6 --corner 62 62  --imgsize  99 99 --RFfit  --chan_rng 0 128
--units alexnet conv3_relu 5 6 6 --corner 62 62  --imgsize  99 99 --RFfit  --chan_rng 128 256
--units alexnet conv3_relu 5 6 6 --corner 62 62  --imgsize  99 99 --RFfit  --chan_rng 256 384
--units alexnet conv4_relu 5 6 6 --corner 46 46  --imgsize  131 131 --RFfit  --chan_rng 0 128
--units alexnet conv4_relu 5 6 6 --corner 46 46  --imgsize  131 131 --RFfit  --chan_rng 128 256
--units alexnet conv5_relu 5 6 6 --corner 30 30  --imgsize  163 163 --RFfit --chan_rng 0 128
--units alexnet conv5_relu 5 6 6 --corner 30 30  --imgsize  163 163 --RFfit --chan_rng 128 256
--units alexnet fc6 5 --chan_rng 0 128
--units alexnet fc6 5 --chan_rng 128 256
--units alexnet fc6 5 --chan_rng 256 384
--units alexnet fc6 5 --chan_rng 384 512
--units alexnet fc6 5 --chan_rng 512 640
--units alexnet fc6 5 --chan_rng 640 768
--units alexnet fc6 5 --chan_rng 768 896
--units alexnet fc6 5 --chan_rng 896 1024
--units alexnet fc6 5 --chan_rng 1024 1152
--units alexnet fc6 5 --chan_rng 1152 1280
--units alexnet fc6 5 --chan_rng 1280 1408
--units alexnet fc6 5 --chan_rng 1408 1536
--units alexnet fc6 5 --chan_rng 1536 1664
--units alexnet fc6 5 --chan_rng 1664 1792
--units alexnet fc6 5 --chan_rng 1792 1920
--units alexnet fc6 5 --chan_rng 1920 2048
--units alexnet fc6 5 --chan_rng 2048 2176
--units alexnet fc6 5 --chan_rng 2176 2304
--units alexnet fc6 5 --chan_rng 2304 2432
--units alexnet fc6 5 --chan_rng 2432 2560
--units alexnet fc6 5 --chan_rng 2560 2688
--units alexnet fc6 5 --chan_rng 2688 2816
--units alexnet fc6 5 --chan_rng 2816 2944
--units alexnet fc6 5 --chan_rng 2944 3072
--units alexnet fc6 5 --chan_rng 3072 3200
--units alexnet fc6 5 --chan_rng 3200 3328
--units alexnet fc6 5 --chan_rng 3328 3456
--units alexnet fc6 5 --chan_rng 3456 3584
--units alexnet fc6 5 --chan_rng 3584 3712
--units alexnet fc6 5 --chan_rng 3712 3840
--units alexnet fc6 5 --chan_rng 3840 3968
--units alexnet fc6 5 --chan_rng 3968 4096
--units alexnet fc7 5 --chan_rng 0 128
--units alexnet fc7 5 --chan_rng 128 256
--units alexnet fc7 5 --chan_rng 256 384
--units alexnet fc7 5 --chan_rng 384 512
--units alexnet fc7 5 --chan_rng 512 640
--units alexnet fc7 5 --chan_rng 640 768
--units alexnet fc7 5 --chan_rng 768 896
--units alexnet fc7 5 --chan_rng 896 1024
--units alexnet fc7 5 --chan_rng 1024 1152
--units alexnet fc7 5 --chan_rng 1152 1280
--units alexnet fc7 5 --chan_rng 1280 1408
--units alexnet fc7 5 --chan_rng 1408 1536
--units alexnet fc7 5 --chan_rng 1536 1664
--units alexnet fc7 5 --chan_rng 1664 1792
--units alexnet fc7 5 --chan_rng 1792 1920
--units alexnet fc7 5 --chan_rng 1920 2048
--units alexnet fc7 5 --chan_rng 2048 2176
--units alexnet fc7 5 --chan_rng 2176 2304
--units alexnet fc7 5 --chan_rng 2304 2432
--units alexnet fc7 5 --chan_rng 2432 2560
--units alexnet fc7 5 --chan_rng 2560 2688
--units alexnet fc7 5 --chan_rng 2688 2816
--units alexnet fc7 5 --chan_rng 2816 2944
--units alexnet fc7 5 --chan_rng 2944 3072
--units alexnet fc7 5 --chan_rng 3072 3200
--units alexnet fc7 5 --chan_rng 3200 3328
--units alexnet fc7 5 --chan_rng 3328 3456
--units alexnet fc7 5 --chan_rng 3456 3584
--units alexnet fc7 5 --chan_rng 3584 3712
--units alexnet fc7 5 --chan_rng 3712 3840
--units alexnet fc7 5 --chan_rng 3840 3968
--units alexnet fc7 5 --chan_rng 3968 4096
--units alexnet fc8 5 --chan_rng 0 128
--units alexnet fc8 5 --chan_rng 128 256
--units alexnet fc8 5 --chan_rng 256 384
--units alexnet fc8 5 --chan_rng 384 512
--units alexnet fc8 5 --chan_rng 512 640
--units alexnet fc8 5 --chan_rng 640 768
--units alexnet fc8 5 --chan_rng 768 896
--units alexnet fc8 5 --chan_rng 896 1000' 

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
# export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`

# python_code_full=$unit_name$'\n'$python_code
# echo "$python_code_full"
# #echo "$python_code_full" > ~\manifold_script.py
# python -c "$python_code_full"
cd ~/Visual_Neuro_InSilico_Exp/
echo "$unit_name"
python insilico_ResizeManifold_torch_efficient_script.py  $unit_name