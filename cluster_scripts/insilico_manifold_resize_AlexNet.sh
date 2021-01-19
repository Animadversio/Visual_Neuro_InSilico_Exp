#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_RFfit_AlexNet

# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).
#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-13
# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='units = ("alexnet", "conv1_relu", 5, 28, 28); chan_rng = (0, 64); 
units = ("alexnet", "conv2_relu", 5, 13, 13); chan_rng = (0, 100); 
units = ("alexnet", "conv3_relu", 5, 6, 6); chan_rng = (0, 100); 
units = ("alexnet", "conv4_relu", 5, 6, 6); chan_rng = (0, 100); 
units = ("alexnet", "conv5_relu", 5, 6, 6); chan_rng = (0, 100); 
units = ("alexnet", "fc6"); chan_rng = (0, 100); 
units = ("alexnet", "fc7"); chan_rng = (0, 100); 
units = ("alexnet", "fc8"); chan_rng = (0, 100); 
units = ("alexnet", "conv1_relu", 5, 28, 28);corner = (110, 110); imgsize = (11, 11); RFfit = True; chan_rng = (0, 100); 
units = ("alexnet", "conv2_relu", 5, 13, 13);corner = (86, 86); imgsize = (51, 51); RFfit = True; chan_rng = (0, 100); 
units = ("alexnet", "conv3_relu", 5, 6, 6);corner = (62, 62); imgsize = (99, 99); RFfit = True; chan_rng = (0, 100); 
units = ("alexnet", "conv4_relu", 5, 6, 6);corner = (46, 46); imgsize = (131, 131); RFfit = True; chan_rng = (0, 100); 
units = ("alexnet", "conv5_relu", 5, 6, 6);corner = (30, 30); imgsize = (163, 163); RFfit = True; chan_rng = (0, 100);' 

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`

python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full"
#echo "$python_code_full" > ~\manifold_script.py
python -c "$python_code_full"
