#!/bin/sh

#PBS -N BigGAN_Optimizer_cmp
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=23:50:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-8

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"

param_list='--method CholCMA CholCMA_class HessCMA HessCMA_class HessCMA_noA_class --steps 100
250 300
300 350
350 400
400 450
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python BigGAN_Evol_cluster.py $csr_lim