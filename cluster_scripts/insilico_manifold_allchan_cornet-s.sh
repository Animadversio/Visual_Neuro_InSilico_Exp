#!/bin/sh

#PBS -N insilico_manifold_allChan_RFfit_Cornet-s
#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb
#PBS -m be
#PBS -q dque
#PBS -t 1-24

export TORCH_HOME="/scratch/binxu/torch"

param_list='--units cornet_s .V1.ReLUnonlin1 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 64
--units cornet_s .V1.ReLUnonlin2 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit --chan_rng 0 64
--units cornet_s .V2.Conv2dconv_input 5 28 28 --imgsize 19 19 --corner 103 103 --RFfit --chan_rng 0 128
--units cornet_s .CORblock_SV2 5 14 14 --imgsize 42 42 --corner 92 93 --RFfit --chan_rng 0 128
--units cornet_s .V4.Conv2dconv_input 5 14 14 --imgsize 43 43 --corner 91 91 --RFfit --chan_rng 0 128
--units cornet_s .V4.Conv2dconv_input 5 14 14 --imgsize 43 43 --corner 91 91 --RFfit --chan_rng 128 256
--units cornet_s .CORblock_SV4 5 7 7 --imgsize 144 144 --corner 42 43 --RFfit --chan_rng 0 128
--units cornet_s .CORblock_SV4 5 7 7 --imgsize 144 144 --corner 42 43 --RFfit --chan_rng 128 256
--units cornet_s .IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit --chan_rng 0 128
--units cornet_s .IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit --chan_rng 128 256
--units cornet_s .IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit --chan_rng 256 384
--units cornet_s .IT.Conv2dconv_input 5 7 7 --imgsize 148 148 --corner 41 39 --RFfit --chan_rng 384 512
--units cornet_s .CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit --chan_rng 0 128
--units cornet_s .CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit --chan_rng 128 256
--units cornet_s .CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit --chan_rng 256 384
--units cornet_s .CORblock_SIT 5 3 3 --imgsize 222 222 --corner 0 0 --RFfit --chan_rng 384 512
--units cornet_s .decoder.Linearlinear 5  --chan_rng 0 128
--units cornet_s .decoder.Linearlinear 5  --chan_rng 128 256
--units cornet_s .decoder.Linearlinear 5  --chan_rng 256 384
--units cornet_s .decoder.Linearlinear 5  --chan_rng 384 512
--units cornet_s .decoder.Linearlinear 5  --chan_rng 512 640
--units cornet_s .decoder.Linearlinear 5  --chan_rng 640 768
--units cornet_s .decoder.Linearlinear 5  --chan_rng 768 896
--units cornet_s .decoder.Linearlinear 5  --chan_rng 896 1000'

export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
# Append the extra command to the script.
cd ~/Visual_Neuro_InSilico_Exp/
# export python_code=`cat cluster_scripts/insilico_ResizeManifold_torch_script.py`
# python_code_full=$unit_name$'\n'$python_code
# echo "$python_code_full"
# python -c "$python_code_full"

echo "$unit_name"
python insilico_ResizeManifold_torch_efficient_script.py  $unit_name