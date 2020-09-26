#!/bin/sh

#PBS -N StyleGAN2_hessian
#PBS -l nodes=1:ppn=1:gpus=1:V100,walltime=4:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-6

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
param_list='--modelname ffhq-256-config-e-003810 --fixed True
--modelname ffhq-512-avg-tpurun1 --fixed True
--modelname stylegan2-cat-config-f --fixed True

--modelname ffhq-256-config-e-003810 --fixed True --wspace True
--modelname ffhq-512-avg-tpurun1 --fixed True --wspace True
--modelname stylegan2-cat-config-f --fixed True --wspace True'
#--modelname stylegan2-ffhq-config-f  --fixed True
#--modelname 2020-01-11-skylion-stylegan2-animeportraits --fixed True

#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 1
#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 0.8
#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 0.6
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python StyleGAN2_spectrum_cluster.py $csr_lim
python StyleGAN2_spectrum_cluster.py --modelname ffhq-512-avg-tpurun1 --fixed True
python StyleGAN2_spectrum_cluster.py --modelname ffhq-512-avg-tpurun1 --fixed True --wspace True

