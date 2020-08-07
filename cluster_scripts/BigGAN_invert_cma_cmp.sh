#!/bin/sh

#PBS -N BigGAN_BasinCMA_invert
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=23:50:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 19-26

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
#--cmasteps 10 --gradsteps 10 --finalgradsteps 500
param_list='--imgidx 250 350 --CMApostGrad True --basis all
--imgidx 250 350 --CMApostGrad True --basis none
--imgidx 250 350 --CMApostGrad False --basis all
--imgidx 250 350 --CMApostGrad False --basis none
--imgidx 250 350 --cmasteps 50 --gradsteps 0 --basis all
--imgidx 250 350 --cmasteps 50 --gradsteps 0 --basis none
--imgidx 250 350 --cmasteps 1 --gradsteps 0 --finalgradsteps 600 --basis all
--imgidx 250 350 --cmasteps 1 --gradsteps 0 --finalgradsteps 600 --basis none
--imgidx 250 350 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis all
--imgidx 250 350 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis none
--dataset BigGAN_rnd --imgidx 0 100 --CMApostGrad True --basis all
--dataset BigGAN_rnd --imgidx 0 100 --CMApostGrad True --basis none
--dataset BigGAN_rnd --imgidx 0 100 --CMApostGrad False --basis all
--dataset BigGAN_rnd --imgidx 0 100 --CMApostGrad False --basis none
--dataset BigGAN_rnd --imgidx 0 100 --cmasteps 50 --gradsteps 0 --basis all
--dataset BigGAN_rnd --imgidx 0 100 --cmasteps 50 --gradsteps 0 --basis none
--dataset BigGAN_rnd --imgidx 0 100 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis all
--dataset BigGAN_rnd --imgidx 0 100 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis none
--dataset BigGAN_rnd --imgidx 100 200 --CMApostGrad True --basis all
--dataset BigGAN_rnd --imgidx 100 200 --CMApostGrad True --basis none
--dataset BigGAN_rnd --imgidx 100 200 --CMApostGrad False --basis all
--dataset BigGAN_rnd --imgidx 100 200 --CMApostGrad False --basis none
--dataset BigGAN_rnd --imgidx 100 200 --cmasteps 50 --gradsteps 0 --basis all
--dataset BigGAN_rnd --imgidx 100 200 --cmasteps 50 --gradsteps 0 --basis none
--dataset BigGAN_rnd --imgidx 100 200 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis all
--dataset BigGAN_rnd --imgidx 100 200 --cmasteps 1 --gradsteps 30 --finalgradsteps 600 --basis none'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python BasinCMA_cluster.py $csr_lim

