#%% random subspaaces
netname, layer = unit
subspace_d = 200
n_gen = 100
import os
from os.path import join
import numpy as np
import matplotlib.pylab as plt
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
from insilico_Exp import ExperimentEvolve, recorddir
hess_mat_path = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Evolution_Avg_Hess.npz"
with np.load(hess_mat_path) as data:
    eigv_avg = data["eigv_avg"]
    eigvect_avg = data["eigvect_avg"]

savedir = os.path.join(recorddir, "%s_%s_eig_subspac%d" % (netname, layer, subspace_d))
os.makedirs(savedir, exist_ok=True)
pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}
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
#%%  certain subspace along the eigen spectum
# unit = ("alexnet", 'fc8')
# netname, layer = unit
# subspace_d = 200  # 50 100 200 400 full
# ofs = 100  # 1 100 200 500 1000 2000 3000
import matplotlib
matplotlib.use("Agg")
n_gen = 100
import os
from os.path import join
import numpy as np
import matplotlib.pylab as plt
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
from insilico_Exp import ExperimentEvolve, recorddir
hess_mat_path = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Evolution_Avg_Hess.npz"
hess_mat_path = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace\Pasu_Space_Avg_Hess.npz"
with np.load(hess_mat_path) as data:
    eigv_avg = data["eigv_avg"]
    eigvect_avg = data["eigvect_avg"]

pos_dict = {"conv5": (7, 7), "conv4": (7, 7), "conv3": (7, 7), "conv2": (14, 14), "conv1": (28, 28)}
for triali in range(5):
    for cfg in [("alexnet", 'fc8'), ("alexnet", 'conv4'), ("alexnet", 'conv2'), \
                ("alexnet", 'fc6'), ("alexnet", 'conv5'), ("alexnet", 'conv1')]:
        netname, layer = cfg
        for chi in range(2,4):
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
#                 lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
#                  range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
#                 best_scores_col.append(lastgen_max)
#
# best_scores_col = np.array(best_scores_col)
# np.save(join(savedir, "best_scores.npy"), best_scores_col)
#%%

import numpy as np
import matplotlib.pylab as plt
import os
from os.path import join
recorddir = "E:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
def max_score_fun(scores_all, generations):
    lastgen_max = np.percentile(scores_all, 99.5)
    return lastgen_max
    # lastgen_max = [scores_all[generations == geni].max() for geni in
    #                range(generations.max() - 10, generations.max() + 1)]
best_scores_col = np.zeros((6, 1, 33, 5))
for triali in range(5):
    for ui, unit in enumerate([("alexnet", 'fc8'), ("alexnet", 'fc6'), ("alexnet", 'conv5'), ("alexnet", 'conv4'), ("alexnet", 'conv2'), ("alexnet", 'conv1')]):
        netname, layer = unit
        for chi in range(1, 2):
            savedir = os.path.join(recorddir, "%s_%s_%d" % (netname, layer, chi))
            for si, subspace_d in enumerate([50, 100, 200, 400]):
                for oi, ofs in enumerate([1, 100, 200, 500, 1000, 2000, 3000]):
                    with np.load(join(savedir, "eig%dsubspc%dscores_tr%01d.npz" % (ofs, subspace_d, triali)),) as data:
                        scores_all = data["scores_all"]
                        generations = data["generations"]
                        best_scores_col[ui, chi, si * 8 + oi, triali] = max_score_fun(scores_all, generations)

                with np.load(join(savedir, "randsubspc%dscores_tr%01d.npz" % (subspace_d, triali)),) as data:
                    scores_all = data["scores_all"]
                    generations = data["generations"]
                    best_scores_col[ui, chi, si * 8 + 7, triali] = max_score_fun(scores_all, generations)

            with np.load(join(savedir, "scores_tr%01d.npz" % (triali)),) as data:
                scores_all = data["scores_all"]
                generations = data["generations"]
                best_scores_col[ui, chi, -1, triali] = max_score_fun(scores_all, generations)
#%% Generate Labels for the figure
label_list = []
for subspace_d in [50, 100, 200, 400]:
    for ofs in [1, 100, 200, 500, 1000, 2000, 3000]:
        label_list.append("%d-%d"%(ofs, ofs+subspace_d))
    label_list.append("rand%d"%(subspace_d))
label_list.append("full")

#%% Plot Figure
figdir = r"E:\Monkey_Data\Generator_DB_Windows\data\with_CNN"
mean_score = best_scores_col.mean((1,3))
fig = plt.figure(figsize=[13, 3.5])
ax = fig.subplots()
MAT = ax.matshow(mean_score / mean_score[:,-1:])
for (i, j), z in np.ndenumerate(mean_score / mean_score[:,-1:]):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',color='white')
plt.ylim([-0.5, 5.5])
plt.xlim([-0.5, 32.5])
plt.yticks(range(6), ["fc8","fc6","conv5","conv4","conv2","conv1"])
plt.ylabel("layers in AlexNet")
plt.xlabel("Subspace in the fc6GAN")
# plt.clabel("Activation relative to full space evolution")
plt.xticks(ticks=range(33), labels=label_list, rotation=30)
fig.colorbar(MAT)
plt.tight_layout()
plt.savefig(join(figdir, "eig_subsp_scores_heatmap.jpg"))
plt.show()
np.savez(join(figdir, "eig_subsp_max_scores.npz"), label_list=label_list, layers=["fc8","fc6","conv5","conv4","conv2","conv1"], best_scores=best_scores_col)
#%%
netname, layer = unit
n_gen = 100
from insilico_Exp import *
from insilico_Exp import ExperimentEvolve, recorddir
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
savedir = os.path.join(recorddir, "%s_%s_full" % (netname, layer))
os.makedirs(savedir, exist_ok=True)
pos_dict = {"conv5": (7,7), "conv4": (7,7), "conv3": (7,7), "conv2": (14,14), "conv1": (28,28)}
best_scores_col = []
for chi in range(100):
    if "fc" in layer:
        unit = (netname, layer, chi)
    else:
        unit = (netname, layer, chi, *pos_dict[layer])
    for triali in range(10):
        optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
        optimizer.lr_schedule(n_gen=n_gen, mode="inv")
        experiment = ExperimentEvolve(unit, max_step=n_gen, backend="torch", optimizer=optimizer, GAN="fc6")
        experiment.run()
        fig0 = experiment.visualize_best(show=False)
        fig0.savefig(join(savedir, "BestImgChan%02dtr%01d.png" % (chi, triali)))
        fig = experiment.visualize_trajectory(show=False)
        fig.savefig(join(savedir, "ScoreTrajChan%02dtr%01d.png" % (chi, triali)))
        fig2 = experiment.visualize_exp(show=False)
        fig2.savefig(join(savedir, "EvolveChan%02dtr%01d.png" % (chi, triali)))
        plt.close("all")
        np.savez(join(savedir, "scores_Chan%02dtr%01d.npz" % (chi, triali)),
                 generations=experiment.generations,
                 scores_all=experiment.scores_all, codes_fin=experiment.codes_all[experiment.generations==experiment.max_steps-1,:])
        lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
         range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
        best_scores_col.append(lastgen_max)

best_scores_col = np.array(best_scores_col)
np.save(join(savedir, "best_scores.npy"), best_scores_col)