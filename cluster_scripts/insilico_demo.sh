#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N RestrEvol100D

# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less 
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only  
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).
#PBS -l nodes=1:ppn=1:gpus=1,walltime=8:00:00,vmem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 9-11

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env

cd ~/Activation-Maximization-for-Visual-System/

# cd Into the run directory; I'll create a new directory to run under
# cd /scratch/binxu.wang

param_list='unit = ("caffe-net", "conv1", 5, 10, 10);
unit = ("caffe-net", "conv2", 5, 10, 10);
unit = ("caffe-net", "conv3", 5, 10, 10);
unit = ("caffe-net", "conv4", 5, 10, 10);
unit = ("caffe-net", "conv5", 5, 10, 10);
unit = ("caffe-net", "fc6", 5);
unit = ("caffe-net", "fc7", 5);
unit = ("caffe-net", "fc8", 5);
unit = ("caffe-net", "fc6", 1);
unit = ("caffe-net", "fc7", 1);
unit = ("caffe-net", "fc8", 1);'
export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code='from insilico_Exp import *
subspace_d = 100
savedir = os.path.join(recorddir, "%s_%s_%d_subspac%d" % (unit[0], unit[1], unit[2], subspace_d))
os.makedirs(savedir, exist_ok=True)
best_scores_col = []
for triali in range(100):
    experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    experiment.get_basis()
    experiment.run()
    fig0 = experiment.visualize_best(show=False)
    fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    fig = experiment.visualize_trajectory(show=False)
    fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    fig2 = experiment.visualize_exp(show=False)
    fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    plt.close("all")
    np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
             generations=experiment.generations,
             scores_all=experiment.scores_all)
    lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
     range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    best_scores_col.append(lastgen_max)
best_scores_col = np.array(best_scores_col)
np.save(join(savedir, "best_scores.npy"), best_scores_col)
'
python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full" 
python -c "$python_code_full"
