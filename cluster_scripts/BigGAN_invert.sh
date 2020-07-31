#!/bin/sh

#PBS -N BigGAN_invert
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=22:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"

param_list='--img  block079_thread000_gen_gen078_003146.jpg  --basis all
--img  block079_thread000_gen_gen078_003146.jpg  --basis sep
--img  block079_thread000_gen_gen078_003146.jpg  --basis none
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python BigGAN_invert_ADAM_BOtune.py $csr_lim

