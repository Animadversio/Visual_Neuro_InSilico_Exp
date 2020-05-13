#!/bin/sh
#PBS -N prepare_GAN_imagepair
#PBS -m be
#PBS -q dque
#PBS -t 1-8
#PBS -l nodes=1:ppn=1:gpus=1,walltime=10:00:00,mem=15gb

# Prepare the virtual env for python
export TORCH_HOME="/scratch/binxu/torch"
param_list='1 1000
1001 2000
2001 3000
3001 4000
4001 5000
5001 6000
6001 7000
7001 8000'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python prepare_img_pairs.py $csr_lim
