#!/bin/sh

#PBS -N hessian_summary
#PBS -l nodes=1:ppn=1:gpus=1:K20x,walltime=23:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"

#param_list='--dataset pasu --method BP --idx_rg 0 100
#'
#export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python fc6GAN_Hess_summary_cluster.py $csr_lim

