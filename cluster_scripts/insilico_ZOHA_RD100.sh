#!/bin/sh

#PBS -N ZOHA_RD100
#PBS -l nodes=1:ppn=1:gpus=1,walltime=12:00:00,vmem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-8

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
cd ~/Visual_Neuro_InSilico_Exp/

param_list='unit = ("alexnet", "conv1");
unit = ("alexnet", "conv2");
unit = ("alexnet", "conv3");
unit = ("alexnet", "conv4");
unit = ("alexnet", "conv5");
unit = ("alexnet", "fc6");
unit = ("alexnet", "fc7");
unit = ("alexnet", "fc8");'
export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code='netname, layer = unit
subspace_d = 100
n_gen = 100
from insilico_Exp import *
from insilico_Exp import ExperimentEvolve, recorddir
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
savedir = os.path.join(recorddir, "%s_%s_subspac%d" % (netname, layer, subspace_d))
os.makedirs(savedir, exist_ok=True)
pos_dict = {"conv5": (7,7), "conv4": (7,7), "conv3": (7,7), "conv2": (14,14), "conv1": (28,28)}
best_scores_col = []
for chi in range(100):
    if "fc" in layer:
        unit = (netname, layer, chi)
    else:
        unit = (netname, layer, chi, *pos_dict[layer])
    for triali in range(10):
        optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, subspace_d, population_size=40, select_size=20)
        optimizer.lr_schedule(n_gen=n_gen, mode="inv")
        optimizer.get_basis("rand")
        experiment = ExperimentEvolve(unit, max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
        experiment.run()
        fig0 = experiment.visualize_best(show=False)
        fig0.savefig(join(savedir, "Subspc%dBestImgChan%02dtr%01d.png" % (subspace_d, chi, triali)))
        fig = experiment.visualize_trajectory(show=False)
        fig.savefig(join(savedir, "Subspc%dScoreTrajChan%02dtr%01d.png" % (subspace_d, chi, triali)))
        fig2 = experiment.visualize_exp(show=False)
        fig2.savefig(join(savedir, "Subspc%dEvolveChan%02dtr%01d.png" % (subspace_d, chi, triali)))
        plt.close("all")
        np.savez(join(savedir, "scores_subspc%dChan%02dtr%01d.npz" % (subspace_d, chi, triali)),
                 generations=experiment.generations,
                 scores_all=experiment.scores_all, codes_fin=experiment.codes_all[experiment.generations==experiment.max_steps-1,:])
        lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
         range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
        best_scores_col.append(lastgen_max)

best_scores_col = np.array(best_scores_col)
np.save(join(savedir, "best_scores.npy"), best_scores_col)
'
python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full"
python -c "$python_code_full"
