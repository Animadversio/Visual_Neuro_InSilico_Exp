#!/bin/sh

#PBS -N PGGAN_invert
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=23:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-3

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"

param_list='--type  celebA  --imgidx 0 25
--type  FFHQ  --imgidx 0 25
--type  PGGAN  --imgidx 0 25
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/GAN-Hessian-Geometry/
python PGGAN_inversion_cluster.py $csr_lim

