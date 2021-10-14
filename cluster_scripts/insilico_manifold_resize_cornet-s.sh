#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_RFfit_cornet-s

#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-17
# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='--units cornet_s .V1.ReLUnonlin1 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 64
--units cornet_s .V1.ReLUnonlin2 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit --chan_rng 0 64
--units cornet_s .V2.Conv2dconv_input 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit --chan_rng 0 75
--units cornet_s .CORblock_SV2 5 14 14 --imgsize 42 42 --corner 92 93 --RFfit --chan_rng 0 75
--units cornet_s .V4.Conv2dconv_input 5 14 14 --imgsize 43 43 --corner 91 91 --RFfit --chan_rng 0 75
--units cornet_s .CORblock_SV4 5 7 7 --imgsize 144 144 --corner 42 43 --RFfit --chan_rng 0 75
--units cornet_s .IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit --chan_rng 0 75
--units cornet_s .CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit --chan_rng 0 75
--units cornet_s .decoder.Linearlinear 5  --chan_rng 0 75
--units cornet_s .V1.ReLUnonlin1 5 57 57 --chan_rng 0 64
--units cornet_s .V1.ReLUnonlin2 5 28 28 --chan_rng 0 64
--units cornet_s .V2.Conv2dconv_input 5 28 28 --chan_rng 0 75
--units cornet_s .CORblock_SV2 5 14 14 --chan_rng 0 75
--units cornet_s .V4.Conv2dconv_input 5 14 14 --chan_rng 0 75
--units cornet_s .CORblock_SV4 5 7 7 --chan_rng 0 75
--units cornet_s .IT.Conv2dconv_input 5 7 7 --chan_rng 0 75
--units cornet_s .CORblock_SIT 5 3 3 --chan_rng 0 75
'

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
# export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`
# python_code_full=$unit_name$'\n'$python_code
# echo "$python_code_full"
# python -c "$python_code_full"

echo "$unit_name"
python insilico_ResizeManifold_torch_script_CLI.py  $unit_name

