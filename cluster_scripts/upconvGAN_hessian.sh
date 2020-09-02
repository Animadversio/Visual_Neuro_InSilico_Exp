#!/bin/sh

#PBS -N upconvGAN_hessian
#PBS -l nodes=1:ppn=1:gpus=1:K20x,walltime=22:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 26-27

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
cd ~/Visual_Neuro_InSilico_Exp/

param_list='--dataset pasu --method BP --idx_rg 0 100
--dataset pasu --method BP --idx_rg 100 200
--dataset evol --method BP --idx_rg 0 100
--dataset evol --method BP --idx_rg 100 200
--dataset evol --method BP --idx_rg 200 300
--dataset pasu --method BackwardIter --idx_rg 0 100
--dataset pasu --method BackwardIter --idx_rg 100 200
--dataset evol --method BackwardIter --idx_rg 0 100
--dataset evol --method BackwardIter --idx_rg 100 200
--dataset evol --method BackwardIter --idx_rg 200 300
--dataset pasu --method ForwardIter --idx_rg 0 100  --EPS 1E-2
--dataset pasu --method ForwardIter --idx_rg 100 200  --EPS 1E-2
--dataset evol --method ForwardIter --idx_rg 0 100  --EPS 1E-2
--dataset evol --method ForwardIter --idx_rg 100 200  --EPS 1E-2
--dataset evol --method ForwardIter --idx_rg 200 300  --EPS 1E-2
--dataset pasu --method BackwardIter --idx_rg 0 100  --GAN fc6_shfl
--dataset pasu --method BackwardIter --idx_rg 100 200  --GAN fc6_shfl
--dataset evol --method BackwardIter --idx_rg 0 100  --GAN fc6_shfl
--dataset evol --method BackwardIter --idx_rg 100 200  --GAN fc6_shfl
--dataset evol --method BackwardIter --idx_rg 200 300  --GAN fc6_shfl
--dataset pasu --method BP --idx_rg 0 100  --GAN fc6_shfl_fix
--dataset pasu --method BP --idx_rg 100 200  --GAN fc6_shfl_fix
--dataset evol --method BP --idx_rg 0 100  --GAN fc6_shfl_fix
--dataset evol --method BP --idx_rg 100 200  --GAN fc6_shfl_fix
--dataset evol --method BP --idx_rg 200 300  --GAN fc6_shfl_fix
--dataset text --method ForwardIter --idx_rg 0 30  --GAN fc6
--dataset text --method BP --idx_rg 0 30  --GAN fc6'
export csr_lim="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/Visual_Neuro_InSilico_Exp/
python hessian_null_space_analysis_cluster.py $csr_lim

