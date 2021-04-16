#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_RFfit_resnet101

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
#PBS -t 22-23
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
units = ("resnet101", ".ReLUrelu", 5, 56, 56); Xlim = (109, 116); Ylim = (109, 116); imgsize = (7, 7); corner = (109, 109); RFfit = True;
units = ("resnet101", ".layer1.Bottleneck0", 5, 28, 28); Xlim = (103, 122); Ylim = (103, 122); imgsize = (19, 19); corner = (103, 103); RFfit = True;
units = ("resnet101", ".layer1.Bottleneck1", 5, 28, 28); Xlim = (103, 124); Ylim = (103, 124); imgsize = (21, 21); corner = (103, 103); RFfit = True;
units = ("resnet101", ".layer2.Bottleneck0", 5, 14, 14); Xlim = (99, 129); Ylim = (99, 129); imgsize = (30, 30); corner = (99, 99); RFfit = True;
units = ("resnet101", ".layer2.Bottleneck3", 5, 14, 14); Xlim = (90, 139); Ylim = (91, 140); imgsize = (49, 49); corner = (90, 91); RFfit = True;
units = ("resnet101", ".layer3.Bottleneck0", 5, 7, 7); Xlim = (86, 145); Ylim = (86, 145); imgsize = (59, 59); corner = (86, 86); RFfit = True;
units = ("resnet101", ".layer3.Bottleneck2", 5, 7, 7); Xlim = (78, 153); Ylim = (78, 153); imgsize = (75, 75); corner = (78, 78); RFfit = True;
units = ("resnet101", ".layer3.Bottleneck6", 5, 7, 7); Xlim = (55, 180); Ylim = (54, 179); imgsize = (125, 125); corner = (55, 54); RFfit = True;
units = ("resnet101", ".layer3.Bottleneck10", 5, 7, 7); Xlim = (41, 191); Ylim = (40, 190); imgsize = (150, 150); corner = (41, 40); RFfit = True;
units = ("resnet101", ".layer4.Bottleneck0", 5, 4, 4); 
units = ("resnet101", ".layer4.Bottleneck2", 5, 4, 4); 
units = ("resnet101", ".Linearfc", 5); '

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`

python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full"
#echo "$python_code_full" > ~\manifold_script.py
python -c "$python_code_full"
