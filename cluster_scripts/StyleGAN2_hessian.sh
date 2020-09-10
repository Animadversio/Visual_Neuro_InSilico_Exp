#!/bin/sh

#PBS -N StyleGAN2_hessian
#PBS -l nodes=1:ppn=1:gpus=1:K20x,walltime=23:59:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 30-35

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
param_list='--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 20 --truncation 1
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 20 --truncation 0.8
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 20 --truncation 0.6
--ckpt_name model.ckpt-533504.pt  --size 512 --trialn 20 --truncation 1.0
--ckpt_name model.ckpt-533504.pt  --size 512 --trialn 20 --truncation 0.8
--ckpt_name model.ckpt-533504.pt  --size 512 --trialn 20 --truncation 0.6
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 1.0
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.8
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.6
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 1.0
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.8
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.6
--ckpt_name stylegan2-car-config-f.pt  --size 512  --trialn 20 --truncation 1.0
--ckpt_name stylegan2-car-config-f.pt  --size 512  --trialn 20 --truncation 0.8
--ckpt_name stylegan2-car-config-f.pt  --size 512  --trialn 20 --truncation 0.6
--ckpt_name ffhq-256-config-e-003810.pt  --size 256  --channel_multiplier 1  --trialn 20 --truncation 1.0
--ckpt_name ffhq-256-config-e-003810.pt  --size 256  --channel_multiplier 1  --trialn 20 --truncation 0.8
--ckpt_name ffhq-256-config-e-003810.pt  --size 256  --channel_multiplier 1  --trialn 20 --truncation 0.6
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 1.0
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.8
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.6
--ckpt_name ffhq-256-config-e-003810.pt  --size 256  --channel_multiplier 1 --trialn 50 --truncation 0.8  --method ForwardIter
--ckpt_name ffhq-256-config-e-003810.pt  --size 256  --channel_multiplier 1 --trialn 50 --truncation 0.6  --method ForwardIter
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 50 --truncation 0.8  --method ForwardIter
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 50 --truncation 0.6  --method ForwardIter
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 50 --truncation 0.8 --method ForwardIter
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 50 --truncation 0.6 --method ForwardIter
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.8 --method ForwardIter
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.6 --method ForwardIter
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.8 --method BackwardIter
--ckpt_name ffhq-512-avg-tpurun1.pt  --size 512 --trialn 20 --truncation 0.6 --method BackwardIter
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.8 --method BackwardIter
--ckpt_name stylegan2-cat-config-f.pt  --size 256  --trialn 20 --truncation 0.6 --method BackwardIter
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 50 --truncation 0.8 --method BackwardIter
--ckpt_name 2020-01-11-skylion-stylegan2-animeportraits.pt --size 512 --trialn 50 --truncation 0.6 --method BackwardIter'

#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 1
#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 0.8
#--ckpt_name AbstractArtFreaGAN.pt --size 1024 --trialn 20 --truncation 0.6
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python StyleGAN2_hess_cluster.py $csr_lim

