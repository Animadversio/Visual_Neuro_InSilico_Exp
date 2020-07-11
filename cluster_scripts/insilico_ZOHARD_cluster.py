netname, layer = unit
n_gen = 100
subspace_d = 200
import os
from os.path import join
import numpy as np
import matplotlib.pylab as plt
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
from insilico_Exp import ExperimentEvolve, recorddir
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
    #     break
    # break
best_scores_col = np.array(best_scores_col)
np.save(join(savedir, "best_scores.npy"), best_scores_col)
