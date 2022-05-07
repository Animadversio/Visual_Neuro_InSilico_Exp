#!/bin/bash
#BSUB -n 4
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'cosine_exp_resnet[40-57]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>20G]'
#BSUB -R 'rusage[mem=20GB]'
#BSUB -M 20G
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/cosine_exp_resnet.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

echo "$LSB_JOBINDEX"

# --net resnet50 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 0 150 --G fc6 --optim CholCMA --score_method cosine corr MSE dot --steps 100 --reps 5 --RFresize

param_list=\
'--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize 
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_re
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
'

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$unit_name"

cd ~/Visual_Neuro_InSilico_Exp/
python cosine_evol_RIS_cluster.py $unit_name