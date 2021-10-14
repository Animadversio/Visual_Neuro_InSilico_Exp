#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_resize_vgg16face

# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).
#PBS -l nodes=1:ppn=1:gpus=1,walltime=12:00:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-13
# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='--units vgg16-face conv2 5 112 112 --imgsize 5 5 --corner 110 110 --RFfit --chan_rng 0 64
--units vgg16-face conv3 5 56 56 --imgsize 10 10 --corner 108 108 --RFfit --chan_rng 0 75
--units vgg16-face conv4 5 56 56 --imgsize 14 14 --corner 106 106 --RFfit --chan_rng 0 75
--units vgg16-face conv5 5 28 28 --imgsize 24 24 --corner 102 102 --RFfit --chan_rng 0 75
--units vgg16-face conv6 5 28 28 --imgsize 31 31 --corner 99 98 --RFfit --chan_rng 0 75
--units vgg16-face conv7 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 75
--units vgg16-face conv9 5 14 14 --imgsize 68 68 --corner 82 82 --RFfit --chan_rng 0 75
--units vgg16-face conv10 5 14 14 --imgsize 82 82 --corner 75 75 --RFfit --chan_rng 0 75
--units vgg16-face conv12 5 7 7 --imgsize 141 141 --corner 50 49 --RFfit --chan_rng 0 75
--units vgg16-face conv13 5 7 7 --imgsize 169 169 --corner 36 35 --RFfit --chan_rng 0 75
--units vgg16-face fc1 5 --chan_rng 0 75
--units vgg16-face fc2 5 --chan_rng 0 75
--units vgg16-face fc3 5 --chan_rng 0 75
'

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID

#echo "$python_code_full" > ~\manifold_script.py
echo "$unit_name"
python insilico_ResizeManifold_torch_script_CLI.py  $unit_name
