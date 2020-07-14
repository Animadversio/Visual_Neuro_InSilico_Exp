#!/bin/sh

#PBS -N ZOHA_RD_eig
#PBS -l nodes=1:ppn=1:gpus=1,walltime=22:00:00,mem=10gb

# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 1-8

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env
export TORCH_HOME="/scratch/binxu/torch"
cd ~/Visual_Neuro_InSilico_Exp/

param_list='cfg = ("alexnet", "conv1");
cfg = ("alexnet", "conv2");
cfg = ("alexnet", "conv3");
cfg = ("alexnet", "conv4");
cfg = ("alexnet", "conv5");
cfg = ("alexnet", "fc6");
cfg = ("alexnet", "fc7");
cfg = ("alexnet", "fc8");'
export unit_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
#$PBS_ARRAYID
export python_code='netname, layer = cfg
n_gen = 100
import os
from os.path import join
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.use("Agg")
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
from insilico_Exp import ExperimentEvolve, recorddir
hess_mat_path = r"/scratch/binxu/CNN_data/Pasu_Space_Avg_Hess.npz"
with np.load(hess_mat_path) as data:
    eigv_avg = data["eigv_avg"]
    eigvect_avg = data["eigvect_avg"]

pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}
for triali in range(5):
    for chi in range(4, 10):
        savedir = os.path.join(recorddir, "%s_%s_%d" % (netname, layer, chi))
        os.makedirs(savedir, exist_ok=True)
        unit = (netname, layer, chi) if "fc" in layer else (netname, layer, chi, *pos_dict[layer])
        # Full evolution
        optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
        optimizer.lr_schedule(n_gen=n_gen, mode="inv")
        experiment = ExperimentEvolve(unit, max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
        experiment.run()
        fig0 = experiment.visualize_best(show=False)
        fig0.savefig(join(savedir, "BestImgtr%01d.png" % (triali)))
        fig = experiment.visualize_trajectory(show=False)
        fig.savefig(join(savedir, "ScoreTrajtr%01d.png" % (triali)))
        fig2 = experiment.visualize_exp(show=False)
        fig2.savefig(join(savedir, "Evolvetr%01d.png" % (triali)))
        plt.close("all")
        np.savez(join(savedir, "scores_tr%01d.npz" % (triali)),
                 generations=experiment.generations,
                 scores_all=experiment.scores_all,
                 codes_fin=experiment.codes_all[experiment.generations == experiment.max_steps - 1, :])

        for subspace_d in [50, 100, 200, 400]:
            # Random Subspace evolution
            optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, subspace_d, population_size=40, select_size=20)
            optimizer.lr_schedule(n_gen=n_gen, mode="inv")
            optimizer.get_basis("rand")
            experiment = ExperimentEvolve(unit, max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
            experiment.run()
            fig0 = experiment.visualize_best(show=False)
            fig0.savefig(join(savedir, "randsubspc%dBestImgtr%01d.png" % (subspace_d, triali)))
            fig = experiment.visualize_trajectory(show=False)
            fig.savefig(join(savedir, "randsubspc%dScoreTrajtr%01d.png" % (subspace_d, triali)))
            fig2 = experiment.visualize_exp(show=False)
            fig2.savefig(join(savedir, "randsubspc%dEvolvetr%01d.png" % (subspace_d, triali)))
            plt.close("all")
            np.savez(join(savedir, "randsubspc%dscores_tr%01d.npz" % (subspace_d, triali)),
                     generations=experiment.generations,
                     scores_all=experiment.scores_all,
                     codes_fin=experiment.codes_all[experiment.generations == experiment.max_steps - 1, :])

            for ofs in [1, 100, 200, 500, 1000, 2000, 3000]:
                optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, subspace_d, population_size=40, select_size=20)
                optimizer.lr_schedule(n_gen=n_gen, mode="inv")
                optimizer.get_basis(eigvect_avg[:, -ofs-subspace_d:-ofs]) # same bug again, column vector is the real eigenvector not row.
                experiment = ExperimentEvolve(unit, max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
                experiment.run()
                fig0 = experiment.visualize_best(show=False)
                fig0.savefig(join(savedir, "eig%dsubspc%dBestImgtr%01d.png" % (ofs, subspace_d, triali)))
                fig = experiment.visualize_trajectory(show=False)
                fig.savefig(join(savedir, "eig%dsubspc%dScoreTrajtr%01d.png" % (ofs, subspace_d, triali)))
                fig2 = experiment.visualize_exp(show=False)
                fig2.savefig(join(savedir, "eig%dsubspc%dEvolvetr%01d.png" % (ofs, subspace_d, triali)))
                plt.close("all")
                np.savez(join(savedir, "eig%dsubspc%dscores_tr%01d.npz" % (ofs, subspace_d, triali)),
                         generations=experiment.generations,
                         scores_all=experiment.scores_all,
                         codes_fin=experiment.codes_all[experiment.generations==experiment.max_steps-1,:])
'
python_code_full=$unit_name$'\n'$python_code
echo "$python_code_full"
python -c "$python_code_full"
