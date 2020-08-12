#!/bin/sh

#PBS -N BigGAN_Optimizer_cmp
#PBS -l nodes=1:ppn=1:gpus=1:K20,walltime=23:50:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-6

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
param_list='--layer fc6 --chans 30 40 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
--layer fc7 --chans 30 40 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
--layer fc8 --chans 30 40 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
--layer fc6 --chans 40 50 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
--layer fc7 --chans 40 50 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
--layer fc8 --chans 40 50 --method HessCMA HessCMA_class HessCMA_noA CholCMA CholCMA_prod CholCMA_class --steps 100 --reps 5
'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python BigGAN_Evol_cluster.py $csr_lim