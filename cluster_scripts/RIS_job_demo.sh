#!/bin/bash
#BSUB -n 1
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -R 'select[mem>8000]'
#BSUB -M 10G
#BSUB -J 'demo[1-10]'
#BSUB -o '/home/binxu.w/demo_out.%J'
#BSUB -oo /home/binxu.w/log/demo.log
#BSUB -a 'docker(alpine)'
#BSUM -u binxu.wang@wustl.edu
#BSUM -N

param_list='1
2
3
4
5
6
7
8
9
'

echo "$LSB_JOBINDEX"

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
echo "$unit_name"
# python insilico_ResizeManifold_torch_efficient_script.py  $unit_name