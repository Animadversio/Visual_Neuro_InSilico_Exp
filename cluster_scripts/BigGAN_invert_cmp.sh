#!/bin/sh

#PBS -N BigGAN_invert_cmp
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=22:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-5

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"

param_list='200 250
250 300
300 350
350 400
400 450
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python BigGAN_invert_cluster.py $csr_lim

