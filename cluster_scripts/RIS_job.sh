#!/bin/bash
echo "$LSB_JOBINDEX"

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

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
echo "$unit_name"



# python insilico_ResizeManifold_torch_efficient_script.py  $unit_name