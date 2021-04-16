#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_RFfit_DenseNet

#PBS -l nodes=1:ppn=1:gpus=1,walltime=23:55:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-16
# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download
param_list='--units resnet101 .layer4.Bottleneck0 5 4 4
--units resnet101 .layer4.Bottleneck2 5 4 4
--units densenet169 .features.ReLUrelu0 5 57 57 --imgsize 7 7 --corner 111 111 --RFfit --chan_rng 0 64
--units densenet169 .features._DenseBlockdenseblock1 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 75
--units densenet169 .features.transition1.Conv2dconv 5 28 28 --imgsize 37 37 --corner 95 95 --RFfit --chan_rng 0 75
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --imgsize 75 75 --corner 78 78 --RFfit --chan_rng 0 75
--units densenet169 .features.transition2.Conv2dconv 5 14 14 --imgsize 85 85 --corner 73 72 --RFfit --chan_rng 0 75
--units densenet169 .features._DenseBlockdenseblock3 5 7 7  --chan_rng 0 75
--units densenet169 .features.transition3.Conv2dconv 5 7 7  --chan_rng 0 75
--units densenet169 .features._DenseBlockdenseblock4 5 3 3  --chan_rng 0 75
--units densenet169 .Linearclassifier 5  --chan_rng 0 75
--units densenet169 .features.ReLUrelu0 5 57 57 --chan_rng 0 64
--units densenet169 .features._DenseBlockdenseblock1 5 28 28 --chan_rng 0 75
--units densenet169 .features.transition1.Conv2dconv 5 28 28 --chan_rng 0 75
--units densenet169 .features._DenseBlockdenseblock2 5 14 14 --chan_rng 0 75
--units densenet169 .features.transition2.Conv2dconv 5 14 14 --chan_rng 0 75
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


# units = ("densenet169", ".features.ReLUrelu0", 5, 57, 57); imgsize = (7, 7); corner = (111, 111); RFfit=True; chan_rng = (0, 64);
# units = ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28); imgsize = (37, 37); corner = (95, 95); RFfit=True; chan_rng = (0, 75);
# units = ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28); imgsize = (37, 37); corner = (95, 95); RFfit=True; chan_rng = (0, 75);
# units = ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14); imgsize = (75, 75); corner = (78, 78); RFfit=True; chan_rng = (0, 75);
# units = ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14); imgsize = (85, 85); corner = (73, 72); RFfit=True; chan_rng = (0, 75);
# units = ("densenet169", ".features._DenseBlockdenseblock3", 5, 7, 7); chan_rng = (0, 75);
# units = ("densenet169", ".features.transition3.Conv2dconv", 5, 7, 7); chan_rng = (0, 75);
# units = ("densenet169", ".features._DenseBlockdenseblock4", 5, 3, 3); chan_rng = (0, 75);
# units = ("densenet169", ".Linearclassifier", 5); chan_rng = (0, 75);
# units = ("densenet169", ".features.ReLUrelu0", 5, 57, 57); chan_rng = (0, 75);
# units = ("densenet169", ".features._DenseBlockdenseblock1", 5, 28, 28); chan_rng = (0, 75);
# units = ("densenet169", ".features.transition1.Conv2dconv", 5, 28, 28); chan_rng = (0, 75);
# units = ("densenet169", ".features._DenseBlockdenseblock2", 5, 14, 14); chan_rng = (0, 75);
# units = ("densenet169", ".features.transition2.Conv2dconv", 5, 14, 14); chan_rng = (0, 75);