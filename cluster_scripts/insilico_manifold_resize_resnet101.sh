#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_full_resnet101

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
#PBS -t 1-10
# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='units = ("resnet101", ".ReLUrelu", 5, 56, 56);
units = ("resnet101", ".layer1.Bottleneck0", 5, 28, 28);
units = ("resnet101", ".layer1.Bottleneck1", 5, 28, 28);
units = ("resnet101", ".layer2.Bottleneck0", 5, 14, 14);
units = ("resnet101", ".layer2.Bottleneck3", 5, 14, 14);
units = ("resnet101", ".layer3.Bottleneck0", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck2", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck6", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck10", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck14", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck18", 5, 7, 7);
units = ("resnet101", ".layer3.Bottleneck22", 5, 7, 7);
'

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code=`cat cluster_scripts/insilico_ResizeManifold_script.py`

python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full"
#echo "$python_code_full" > ~\manifold_script.py
python -c "$python_code_full"
