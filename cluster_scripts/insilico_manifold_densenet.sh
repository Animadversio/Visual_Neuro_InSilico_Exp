#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_densenet

# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less 
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only  
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).
#PBS -l nodes=1:ppn=1:gpus=1,walltime=12:00:00,mem=15gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-9

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env

cd ~/Visual_Neuro_InSilico_Exp/
export TORCH_HOME="/scratch/binxu/torch" # or it will download models to home folder. (and it will explode....)
# cd Into the run directory; I'll create a new directory to run under
# cd /scratch/binxu.wang

param_list='unit = ("densenet121", "bn1", 5, 56, 56);
unit = ("densenet121", "denseblock1", 5, 28, 28);
unit = ("densenet121", "transition1", 5, 14, 14);
unit = ("densenet121", "denseblock2", 5, 14, 14);
unit = ("densenet121", "transition2", 5, 7, 7);
unit = ("densenet121", "denseblock3", 5, 7, 7);
unit = ("densenet121", "transition3", 5, 4, 4);
unit = ("densenet121", "denseblock4", 5, 4, 4);
unit = ("densenet121", "fc1", 5);'
# unit = ("caffe-net", "conv2", 5, 10, 10);
# unit = ("caffe-net", "conv4", 5, 10, 10);
# unit = ("caffe-net", "conv5", 5, 10, 10);
# unit = ("caffe-net", "fc6", 1);
export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code='from insilico_Exp import *
import torch
from os.path import join
savedir = join(recorddir, "%s_%s_manifold" % (unit[0], unit[1]))
os.makedirs(savedir, exist_ok=True)
for chan in range(50):
    if "fc" in unit[1]:
        unit = (unit[0], unit[1], chan)
        label = "chan%d" % (chan, )
    else:
        unit = (unit[0], unit[1], chan, unit[3], unit[4])
        label = "chan%d-(%d,%d)" % (chan, unit[3], unit[4])
    experiment = ExperimentManifold(unit, max_step=100, savedir=savedir, backend="torch", explabel=label)
    experiment.run()
    experiment.analyze_traj()
    score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
    np.savez(join(savedir, "score_map_chan%d.npz" % chan), score_sum=score_sum,
             Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
    plt.close("all")
'
python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full" 
#echo "$python_code_full" > ~\manifold_script.py
python -c "$python_code_full"
