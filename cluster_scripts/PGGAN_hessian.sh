#!/bin/sh

#PBS -N ProgGrowGAN_hessian
#PBS -l nodes=1:ppn=1:gpus=1:K20x,walltime=23:58:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-2


export TORCH_HOME="/scratch/binxu/torch"

param_list='--dataset rand --idx_rg 0 200
--dataset rand --idx_rg 200 400
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python PGGAN_hess_cluster.py $csr_lim

