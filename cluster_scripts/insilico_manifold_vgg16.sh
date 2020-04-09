#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N insilico_manifold_vgg16

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
#PBS -t 1-8

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env

cd ~/Visual_Neuro_InSilico_Exp/

# cd Into the run directory; I'll create a new directory to run under
# cd /scratch/binxu.wang

param_list='unit = ("vgg16", "conv2", 5, 112, 112);
unit = ("vgg16", "conv4", 5, 56, 56);
unit = ("vgg16", "conv7", 5, 28, 28);
unit = ("vgg16", "conv10", 5, 14, 14);
unit = ("vgg16", "conv13", 5, 7, 7);
unit = ("vgg16", "fc1", 1);
unit = ("vgg16", "fc2", 1);
unit = ("vgg16", "fc3", 1);'
# unit = ("caffe-net", "conv2", 5, 10, 10);
# unit = ("caffe-net", "conv4", 5, 10, 10);
# unit = ("caffe-net", "conv5", 5, 10, 10);
# unit = ("caffe-net", "fc6", 1);
export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code='from insilico_Exp import *
from os.path import join
savedir = join(recorddir, "%s_%s_manifold" % (unit[0], unit[1]))
os.makedirs(savedir, exist_ok=True)
for chan in range(50):
    if "conv" not in unit[1]:
        unit = (unit[0], unit[1], chan)
    else:
        unit = (unit[0], unit[1], chan, unit[3], unit[4])
    experiment = ExperimentManifold(unit, max_step=100, savedir=savedir, backend="torch", explabel="chan%d-(%d,%d)" % (chan, unit[3], unit[4]))
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
