from insilico_Exp_torch import ExperimentManifold
from ZO_HessAware_Optimizers import CholeskyCMAES
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Manifold.Kent_fit_utils import fit_Kent_Stats, SO3
from sklearn.decomposition import PCA
from ZOHA_Optimizer import ZOHA_Sphere_lr_euclid, ZOHA_Sphere_lr_euclid_ReducDim
from os.path import join
from stats_utils import saveallforms
from Manifold.Manifold_Tuning_lib import integrate_VUS

figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\exps"
theta_arr, phi_arr = np.linspace(-np.pi/2, np.pi/2, 21), np.linspace(-np.pi/2, np.pi/2, 21) #np.meshgrid(np.linspace(-90, 91, 21), np.linspace(-90, 91, 21))
#%%
"""
a toy example showing how we could result in the manifold result 
due to the number of tuned axes
"""

#%%
#%%
# from scipy.stats import special_ortho_group, ortho_group
# Omat = ortho_group.rvs(4096)
# Hess = Omat.T @ np.diag(Hdiag) @ Omat
#%%
"""
Model neuron as Quadratic function with a Sinusoid nonlinearity 
"""
def model_neruon_constructer(center, Hdiag, bandwidth, basis=None, code_len=4096,
                             thresh=0.0, sphere_norm=300):
    """Linear
        r = Sigmoid(- x^T H x_0)
    """
    cvec = center[np.newaxis, :]
    Dvec = Hdiag[np.newaxis, :]
    normalizer = np.sum(cvec * Dvec * cvec, axis=1) / np.linalg.norm(cvec, axis=1) * sphere_norm
    def model_neuron(x, ):
        xarr = x.reshape((-1, code_len))
        if basis is None:
            dotprod = np.sum(xarr * Dvec * cvec, axis=1) / normalizer
        else:
            x_rot = xarr @ basis
            c_rot = cvec @ basis
            dotprod = np.sum(x_rot * Dvec * c_rot, axis=1) / normalizer
        return 1 / (1 + np.exp(- (dotprod - thresh) / bandwidth))
    return model_neuron


def quad_model_neruon_constructer(center, Hdiag, bandwidth, basis=None, code_len=4096,
                             sphere_norm=300):
    """
    r = exp( -(x - x_0)^T H (x - x_0) / bandwidth ** 2)
    :param center:
    :param Hdiag:
    :param bandwidth:
    :param basis:
    :param code_len:
    :param sphere_norm:
    :return:
    """
    cvec = center[np.newaxis, :] / np.linalg.norm(center, axis=0) * sphere_norm
    Dvec = Hdiag[np.newaxis, :]
    normalizer = np.sum(cvec * Dvec * cvec, axis=1) / np.linalg.norm(cvec, axis=1) * sphere_norm
    def model_neuron(x, ):
        xarr = x.reshape((-1, code_len))
        if basis is None:
            dotprod = np.sum((xarr - cvec) ** 2 * Dvec, axis=1) / bandwidth ** 2
        else:
            dx_rot = (xarr - cvec) @ basis
            dotprod = np.sum(dx_rot ** 2 * Dvec, axis=1) / bandwidth ** 2
        return np.exp(-dotprod)
    return model_neuron

#%
def run_manifold(model, center, perturbvecs, interval=9, sphere_norm=300,
                 code_len=4096):
    unit_center = center / np.linalg.norm(center)
    unit_center = unit_center[np.newaxis, :]
    if perturbvecs is None:
        perturbvecs = np.random.randn(2, code_len)
    unit_perturb = perturbvecs - (perturbvecs @ unit_center.T) @ unit_center
    unit_perturb = unit_perturb / np.linalg.norm(unit_perturb, axis=1, keepdims=True)
    unit_perturb[1, :] = unit_perturb[1, :] - (unit_perturb[1, :] @ unit_perturb[0, :].T) * unit_perturb[0, :]
    unit_perturb[1, :] = unit_perturb[1, :] / np.linalg.norm(unit_perturb[1, :])
    vectors = np.concatenate((unit_center, unit_perturb), axis=0)
    assert np.allclose(vectors @ vectors.T, np.eye(3))
    # self.Perturb_vec.append(vectors)
    # img_list = []
    code_list = []
    interv_n = int(90 / interval)
    for j in range(-interv_n, interv_n + 1):
        for k in range(-interv_n, interv_n + 1):
            theta = interval * j / 180 * np.pi
            phi = interval * k / 180 * np.pi
            code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                  np.sin(theta) * np.cos(phi),
                                  np.sin(phi)]]) @ vectors
            code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
            code_list.append(code_vec)
    code_arr = np.array(code_list)
    scores = model(code_arr)
    score_mat = np.array(scores).reshape((interv_n * 2 + 1, interv_n * 2 + 1))
    return score_mat


def run_evol(model, maxgen=50, optim=None, init_sigma=3.0, ):
    if optim is None:
        optim = CholeskyCMAES(4096, init_sigma=init_sigma)
    codes = np.random.randn(30, 4096)
    codes_all = []
    scores_all = []
    generations = []
    for i in range(maxgen):
        scores = model(codes)
        codes_new = optim.step_simple(scores, codes)
        codes_all.append(codes.copy())
        scores_all.append(scores.copy())
        generations.append(np.ones_like(scores) * i)
        codes = codes_new
        print(i, f"{scores.mean():.2f} {scores.std():.2f}")
    codes_arr = np.concatenate(codes_all, axis=0)
    scores_arr = np.concatenate(scores_all, axis=0)
    generations = np.concatenate(generations, axis=0)
    return codes_arr, scores_arr, generations


def analyze_evol(codes_arr):
    code_pca = PCA(n_components=50)
    PC_Proj_codes = code_pca.fit_transform(codes_arr)
    PC_vectors = code_pca.components_
    if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
        # this is important or the images we show will land in the opposite side of the globe.
        inv_PC1 = True
        PC_vectors[0, :] = - PC_vectors[0, :]
    return PC_vectors, PC_Proj_codes, code_pca
#%%
# NULLDIM = 3800  #3000
active_dim = 200  #800
bandwidth = 0.2
thresh = 2.0
Hdiag = np.ones(4096)
Hdiag[active_dim:] = 0.0001
center = np.random.randn(4096)
model = model_neruon_constructer(center, Hdiag, bandwidth=bandwidth, thresh=thresh)
score_mat = run_manifold(model, center, None, interval=9, sphere_norm=300, code_len=4096)
param, sigmas, res, R2 = fit_Kent_Stats(theta_arr, phi_arr, score_mat, )
theta, phi, psi, kappa, beta, A, bsl = param
sns.heatmap(score_mat, cmap='coolwarm', vmin=0, ) # vmax=1
plt.title(f"R2: {R2:.2f} kappa {kappa:.2f} beta {beta:.2f} A {A:.2f} bsl {bsl:.2f}")
plt.show()
#%%
model = model_neruon_constructer(center, Hdiag, bandwidth=0.2, thresh=2.0)
model(np.random.randn(50, 4096))
#%%
active_dim = 10  #800
bandwidth = 0.4
thresh = 0.8
for thresh in [2.0, 0.5, 1.0, ]:
    for active_dim in [5, 10, 50, 200, 800]:  # 20, 100,
        for bandwidth in [0.1, 0.2, 0.4, 0.8, 1.6]:
            Hdiag = np.ones(4096)
            Hdiag[active_dim:] = 0.000001
            center = np.random.randn(4096)
            model = model_neruon_constructer(center, Hdiag, bandwidth=bandwidth, thresh=thresh)
            #%%
            codes_arr, scores_arr, generations = run_evol(model, init_sigma=3.0, maxgen=50)
            PC_vectors, PC_Proj_codes, code_pca = analyze_evol(codes_arr)
            score_mat = run_manifold(model, PC_vectors[0, :], PC_vectors[1:3, :], interval=9, sphere_norm=300, code_len=4096)
            score_mat_RND = run_manifold(model, center, None, interval=9, sphere_norm=300, code_len=4096)
            #%%
            param_RND, sigmas, res, R2_RND = fit_Kent_Stats(theta_arr, phi_arr, score_mat_RND, )
            theta_RND, phi_RND, psi_RND, kappa_RND, beta_RND, A_RND, bsl_RND = param_RND
            param, sigmas, res, R2 = fit_Kent_Stats(theta_arr, phi_arr, score_mat, )
            theta, phi, psi, kappa, beta, A, bsl = param
            figh, axs = plt.subplots(1, 3, figsize=(10, 5))
            plt.sca(axs[0])
            sns.heatmap(score_mat, cmap='coolwarm', vmin=0, )#vmax=1
            plt.axis('equal')
            plt.title(f"Neuron model act dim {active_dim} band {bandwidth:.1f} thresh {thresh:.1f}\n"
                      f"R2: {R2:.2f} kappa {kappa:.2f} beta {beta:.2f} A {A:.2f} bsl {bsl:.2f}")
            plt.sca(axs[1])
            sns.heatmap(score_mat_RND, cmap='coolwarm', vmin=0, )  # vmax=1
            plt.axis('equal')
            plt.title(
                f"Neuron model act dim {active_dim} band {bandwidth:.1f} thresh {thresh:.1f}\n"
                f"R2: {R2_RND:.2f} kappa {kappa_RND:.2f} beta {beta_RND:.2f} A {A_RND:.2f} bsl {bsl_RND:.2f}")
            plt.sca(axs[2])
            plt.scatter(generations, scores_arr.T, alpha=0.5)
            plt.tight_layout()
            # plt.savefig(join(figdir, f"model_{active_dim}_{bandwidth:.1f}_{thresh:.1f}.png"))
            saveallforms(figdir, f"model_{active_dim}_{bandwidth:.1f}_{thresh:.1f}")
            plt.show()
#%%
"""Exp Quadratic model (Gaussian)"""
figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\expdata"
#%%
# for active_dim in [5, 10, 20, 50, 100, 200, 400, 800]:  # 100,
#     for bandwidth in [2, 5, 10, 20, 40, 80, 160, 320, ]:
for rep in range(5):
    for active_dim in [5, 10, 20, 40, 80, 160]:  # 100,
        for bandwidth in [2, 5, 10, 20, 40, 80, ]:
            Hdiag = np.ones(4096)
            Hdiag[active_dim:] = 0.000001
            center = np.random.randn(4096)
            qmodel = quad_model_neruon_constructer(center, Hdiag,
                                    bandwidth=bandwidth, sphere_norm=300)
            codes_arr, scores_arr, generations = run_evol(qmodel, init_sigma=3.0, maxgen=50)
            PC_vectors, PC_Proj_codes, code_pca = analyze_evol(codes_arr)
            score_mat = run_manifold(qmodel, PC_vectors[0, :], PC_vectors[1:3, :],
                                     interval=9, sphere_norm=300, code_len=4096)
            score_mat_RND = run_manifold(qmodel, center, None, interval=9, sphere_norm=300, code_len=4096)
            # %%
            param_RND, sigmas, res, R2_RND = fit_Kent_Stats(theta_arr, phi_arr, score_mat_RND, )
            theta_RND, phi_RND, psi_RND, kappa_RND, beta_RND, A_RND, bsl_RND = param_RND
            param, sigmas, res, R2 = fit_Kent_Stats(theta_arr, phi_arr, score_mat, )
            theta, phi, psi, kappa, beta, A, bsl = param
            #%%
            optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
            optim.lr_schedule(n_gen=50, mode="exp")
            codes_arr_sph, scores_arr_sph, generations_sph = run_evol(qmodel, maxgen=50, optim=optim)
            optim_RD = ZOHA_Sphere_lr_euclid_ReducDim(4096, 50, population_size=40, select_size=20)
            optim_RD.lr_schedule(n_gen=50, mode="exp")
            optim_RD.get_basis("rand")
            codes_arr_RD, scores_arr_RD, generations_RD = run_evol(qmodel, maxgen=50, optim=optim_RD)
            #%%
            np.savez(join(figdir, f"quadmodel_{active_dim}_{bandwidth:.1f}_rep{rep:d}_data"),
                     center=center, Hdiag=Hdiag, active_dim=active_dim, bandwidth=bandwidth,
                     scores_arr=scores_arr, generations=generations,
                     scores_arr_sph=scores_arr_sph, generations_sph=generations_sph,
                     scores_arr_RD=scores_arr_RD, generations_RD=generations_RD,
                     theta_arr=theta_arr, phi_arr=phi_arr,
                     score_mat=score_mat, score_mat_RND=score_mat_RND,
                     param_RND=param_RND, R2_RND=R2_RND, param=param, R2=R2,
            )
            #%%
            figh, axs = plt.subplots(1, 4, figsize=(15.5, 4.5))
            plt.sca(axs[0])
            sns.heatmap(score_mat, cmap='coolwarm', vmin=0, )  # vmax=1
            plt.axis('image')
            plt.title(f"Manifold in PC123 space\n"
                      f"R2: {R2:.2f} kappa {kappa:.2f} beta {beta:.2f}\n A {A:.2f} bsl {bsl:.2f}")
            plt.sca(axs[1])
            sns.heatmap(score_mat_RND, cmap='coolwarm', vmin=0, )  # vmax=1
            plt.axis('image')
            plt.title(
                f"Random Manifold from true center\n"
                f"R2: {R2_RND:.2f} kappa {kappa_RND:.2f} beta {beta_RND:.2f}\n A {A_RND:.2f} bsl {bsl_RND:.2f}")
            plt.sca(axs[2])
            plt.scatter(generations, scores_arr.T, alpha=0.5)
            plt.title(f"CMA Evolution")
            plt.sca(axs[3])
            plt.scatter(generations_sph, scores_arr_sph.T, alpha=0.3, label="full")
            plt.scatter(generations_RD, scores_arr_RD.T, alpha=0.3, label="50D")
            plt.legend()
            plt.title(f"Reduced Dimension Evolution comparison")
            plt.suptitle(f"Neuron model act dim {active_dim} band {bandwidth:.1f}")
            plt.tight_layout()
            # plt.savefig(join(figdir, f"quadmodel_{active_dim}_{bandwidth:.1f}.png"))
            saveallforms(figdir, f"quadmodel_{active_dim}_{bandwidth:.1f}_rep{rep:d}")
            plt.show()
#%%

datacol = []
for active_dim in [5, 10, 20, 40, 80, 160]:  # 100,
    for bandwidth in [2, 5, 10, 20, 40, 80, ]:
        for rep in range(5):
            data = edict(np.load(join(figdir,
              join(figdir, f"quadmodel_{active_dim}_{bandwidth:.1f}_rep{rep:d}_data.npz"))))
            kappa = data.param[3]
            kappa_RND = data.param_RND[3]
            meanscore = [data.scores_arr[data.generations == i].mean() for i in range(50)]
            meanscore_RD = [data.scores_arr_RD[data.generations_RD == i].mean() for i in range(50)]
            meanscore_sph = [data.scores_arr_sph[data.generations_sph == i].mean() for i in range(50)]
            ratio = (meanscore_RD[-1] - meanscore_RD[0]) / (meanscore_sph[-1] - meanscore_sph[0])
            convergtime = np.array(meanscore) < (meanscore[0] + 0.632 * (meanscore[-1] - meanscore[0])).sum()
            convergtime = convergtime.sum()
            if meanscore[-1] - meanscore[0] < 1E-6:
                convergtime = np.nan
            maxact = np.max(meanscore)
            VUS_int, norm_VUS = integrate_VUS(data.score_mat)
            VUS_int_RND, norm_VUS_RND = integrate_VUS(data.score_mat_RND)
            datacol.append(edict(active_dim=active_dim, bandwidth=bandwidth, rep=rep,
                                 kappa=kappa, kappa_RND=kappa_RND,
                                 ratio=ratio, convergtime=convergtime,
                                 maxact=maxact, norm_VUS=norm_VUS, norm_VUS_RND=norm_VUS_RND,))
df = pd.DataFrame(datacol)
#%% Summary figures
sumdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\summary"
#%%
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["kappa"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("kappa (manifold peak)")
saveallforms(sumdir, "toy_synopsis-kappa_manifold_annot")
plt.show()

sns.heatmap(df.groupby(["active_dim", "bandwidth"])["kappa_RND"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("kappa (random direction)")
saveallforms(sumdir, "toy_synopsis-kappa_random_annot")
plt.show()
#
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["ratio"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("50-D full space ratio")
saveallforms(sumdir, "toy_synopsis-ratio_full50D_annot")
plt.show()
#%
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["convergtime"].mean().unstack(),
            annot=True, fmt=".0f")
plt.title("Converge Time")
saveallforms(sumdir, "toy_synopsis-convergtime_annot")
plt.show()
#%
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["maxact"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("Max activation")
saveallforms(sumdir, "toy_synopsis-maxactiv_annot")
plt.show()
#%
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["norm_VUS"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("Normed Volume under Surface")
saveallforms(sumdir, "toy_synopsis-normVUS_manif_annot")
plt.show()
#%
sns.heatmap(df.groupby(["active_dim", "bandwidth"])["norm_VUS_RND"].mean().unstack(),
            annot=True, fmt=".1f")
plt.title("Normed Volume under Surface (Random Direction)")
saveallforms(sumdir, "toy_synopsis-normVUS_rand_annot")
plt.show()


#%% Best examples
from scipy.stats import sem
outdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Figure6Toy\source"
figdir = r"E:\OneDrive - Harvard University\Manifold_Toymodel\expdata"
def visualize_toymodel_exp(active_dim, bandwidth, rep):
    data = edict(np.load(join(figdir,
              f"quadmodel_{active_dim}_{bandwidth:.1f}_rep{rep:d}_data.npz")))
    kappa = data.param[3]
    beta  = data.param[4]
    kappa_RND = data.param_RND[3]
    meanscore = np.array([data.scores_arr[data.generations == i].mean() for i in range(50)])
    semscore = np.array([np.std(data.scores_arr[data.generations == i]) for i in range(50)])
    meanscore_RD = np.array([data.scores_arr_RD[data.generations_RD == i].mean() for i in range(50)])
    semscore_RD = np.array([np.std(data.scores_arr_RD[data.generations_RD == i]) for i in range(50)])
    meanscore_sph = np.array([data.scores_arr_sph[data.generations_sph == i].mean() for i in range(50)])
    semscore_sph = np.array([np.std(data.scores_arr_sph[data.generations_sph == i]) for i in range(50)])
    ratio = (meanscore_RD[-1] - meanscore_RD[0]) / (meanscore_sph[-1] - meanscore_sph[0])
    maxact = np.max(meanscore)
    convergtime = meanscore < (meanscore[0] + 0.632 * (meanscore[-1] - meanscore[0])).sum()
    convergtime = convergtime.sum()
    if meanscore[-1] - meanscore[0] < 1E-6:
        convergtime = np.nan
    VUS_int, norm_VUS = integrate_VUS(data.score_mat)
    VUS_int_RND, norm_VUS_RND = integrate_VUS(data.score_mat_RND)
    figh, axs = plt.subplots(1, 3, figsize=(12.5, 4.5))
    plt.sca(axs[0])
    sns.heatmap(data.score_mat, cmap='coolwarm', vmin=0,)
                # xticklabels=np.arange(-90,91,9), yticklabels=np.arange(-90,91,9))  # vmax=1
    plt.xticks([0, 10, 20], [-90, 0, 90])
    plt.yticks([0, 10, 20], [-90, 0, 90])
    plt.axis('image')
    plt.title(f"Manifold in PC123 space\n  normVUS {norm_VUS:.2f}\n"
              f"R2: {data.R2:.2f} kappa {kappa:.2f} beta {beta:.2f}") #\n A {A:.2f} bsl {bsl:.2f}
    # plt.sca(axs[1])
    # sns.heatmap(score_mat_RND, cmap='coolwarm', vmin=0, )  # vmax=1
    # plt.axis('image')
    # plt.title(
    #     f"Random Manifold from true center\n"
    #     f"R2: {R2_RND:.2f} kappa {kappa_RND:.2f} beta {beta_RND:.2f}\n A {A_RND:.2f} bsl {bsl_RND:.2f}")
    plt.sca(axs[1])
    # plt.scatter(data.generations, data.scores_arr.T, alpha=0.2)
    plt.plot(range(len(meanscore)), meanscore, 'r', color="k", lw=1.5)
    plt.fill_between(range(len(meanscore)),
                     meanscore - semscore, meanscore + semscore,alpha=0.4)
    plt.title(f"CMA Evolution\nConvergence Time: {convergtime:.0f}")
    plt.ylim(0, 1)
    plt.sca(axs[2])
    # plt.scatter(data.generations_sph, data.scores_arr_sph.T, alpha=0.2, label="full")
    # plt.scatter(data.generations_RD, data.scores_arr_RD.T, alpha=0.2, label="50D")
    plt.plot(range(len(meanscore_sph)), meanscore_sph, 'r', color="k", lw=1.5, label="full")
    plt.fill_between(range(len(meanscore_sph)),
                     meanscore_sph - semscore_sph, meanscore_sph + semscore_sph, alpha=0.4)
    plt.plot(range(len(meanscore_RD)), meanscore_RD, 'r', color="r", lw=1.5, label="50D")
    plt.fill_between(range(len(meanscore_RD)),
                     meanscore_RD - semscore_RD, meanscore_RD + semscore_RD, alpha=0.4)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"Reduced Dimension Evolution comparison\nRatio: {ratio:.2f}")
    plt.suptitle(f"Neuron model Tuned dim D={active_dim} tuning width $\sigma$={bandwidth:.1f}")
    plt.tight_layout()
    # plt.savefig(join(figdir, f"quadmodel_{active_dim}_{bandwidth:.1f}.png"))
    saveallforms(outdir, f"quadmodel_{active_dim}_{bandwidth:.1f}_rep{rep:d}")
    plt.show()
    return figh, axs


visualize_toymodel_exp(20, 20, 0)
# visualize_toymodel_exp(40, 40, 0)
visualize_toymodel_exp(40, 20, 0)
visualize_toymodel_exp(40, 40, 0)
visualize_toymodel_exp(80, 40, 2)
visualize_toymodel_exp(80, 80, 0)

#%%
active_dim = 50
Hdiag = np.ones(4096)
Hdiag[active_dim:] = 0.000001
center = np.random.randn(4096)
qmodel = quad_model_neruon_constructer(center, Hdiag, bandwidth=20, sphere_norm=300)
score_mat = run_manifold(qmodel, center, None, interval=9, sphere_norm=300, code_len=4096)
param, sigmas, res, R2 = fit_Kent_Stats(theta_arr, phi_arr, score_mat, )
theta, phi, psi, kappa, beta, A, bsl = param
sns.heatmap(score_mat, cmap='coolwarm', vmin=0, ) # vmax=1
plt.title(f"R2: {R2:.2f} kappa {kappa:.2f} beta {beta:.2f} A {A:.2f} bsl {bsl:.2f}")
plt.show()

#%%

active_dim = 35
Hdiag = np.ones(4096)
Hdiag[active_dim:] = 0.000001
center = np.random.randn(4096)
qmodel = quad_model_neruon_constructer(center, Hdiag, bandwidth=20, sphere_norm=300)
optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20)
optim.lr_schedule(n_gen=50, mode="exp")
codes_arr, scores_arr, generations = run_evol(qmodel, maxgen=50, optim=optim)
optim_RD = ZOHA_Sphere_lr_euclid_ReducDim(4096, 50, population_size=40, select_size=20)
optim_RD.lr_schedule(n_gen=50, mode="exp")
optim_RD.get_basis("rand")
codes_arr_RD, scores_arr_RD, generations_RD = run_evol(qmodel, maxgen=50, optim=optim_RD)

plt.figure()
plt.scatter(generations, scores_arr.T, alpha=0.5)
plt.scatter(generations_RD, scores_arr_RD.T, alpha=0.5)
plt.show()
#%%
plt.figure()
sns.heatmap(res, cmap='coolwarm', vmin=0, )#vmax=1
plt.axis('equal')
plt.show()
#%%
#%%
# def run_toy_manifold(model, subspace_list, interval=9, print_manifold=True):
#     '''Generate examples on manifold and run'''
#     score_sum = []
#     T0 = time()
#     figsum = plt.figure(figsize=[16.7, 4])
#     for spi, subspace in enumerate(subspace_list):
#         code_list = []
#         if subspace == "RND":
#             title = "Norm%dRND%dRND%d" % (self.sphere_norm, 0 + 1, 1 + 1)
#             print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d) " % self.sphere_norm)
#             rand_vec2 = np.random.randn(2, self.code_length)
#             rand_vec2 = rand_vec2 - (rand_vec2 @ self.PC_vectors.T) @ self.PC_vectors
#             rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
#             rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
#             rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
#             vectors = np.concatenate((self.PC_vectors[0:1, :], rand_vec2), axis=0)
#             self.Perturb_vec.append(vectors)
#             # img_list = []
#             interv_n = int(90 / interval)
#             for j in range(-interv_n, interv_n + 1):
#                 for k in range(-interv_n, interv_n + 1):
#                     theta = interval * j / 180 * np.pi
#                     phi = interval * k / 180 * np.pi
#                     code_vec = np.array([[np.cos(theta) * np.cos(phi),
#                                           np.sin(theta) * np.cos(phi),
#                                           np.sin(phi)]]) @ vectors
#                     code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
#                     code_list.append(code_vec)
#                     # img = self.G.visualize(code_vec)
#                     # img_list.append(img.copy())
#         else:
#             PCi, PCj = subspace
#             title = "Norm%dPC%dPC%d" % (self.sphere_norm, PCi + 1, PCj + 1)
#             print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (PCi + 1, PCj + 1, self.sphere_norm, ))
#             # img_list = []
#             interv_n = int(90 / interval)
#             self.Perturb_vec.append(self.PC_vectors[[0, PCi, PCj], :])
#             for j in range(-interv_n, interv_n + 1):
#                 for k in range(-interv_n, interv_n + 1):
#                     theta = interval * j / 180 * np.pi
#                     phi = interval * k / 180 * np.pi
#                     code_vec = np.array([[np.cos(theta) * np.cos(phi),
#                                           np.sin(theta) * np.cos(phi),
#                                           np.sin(phi)]]) @ self.PC_vectors[[0, PCi, PCj], :]
#                     code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
#                     code_list.append(code_vec)
#                     # img = self.G.visualize(code_vec)
#                     # img_list.append(img.copy())
#                     # plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (
#                     # self.sphere_norm, interval * j, interval * k)), img)
#
#         # pad_img_list = resize_and_pad(img_list, self.imgsize, self.corner) # Show image as given size at given location
#         # scores = self.CNNmodel.score(pad_img_list)
#         # print("Latent vectors ready, rendering. (%.3f sec passed)"%(time()-T0))
#         code_arr = np.array(code_list)
#         scores = model(code_arr)
#         # print("Image and score ready! Figure printing (%.3f sec passed)"%(time()-T0))
#         # fig = utils.visualize_img_list(img_list, scores=scores, ncol=2*interv_n+1, nrow=2*interv_n+1, )
#         # subsample images for better visualization
#         scores = np.array(scores).reshape((2*interv_n+1, 2*interv_n+1)) # Reshape score as heatmap.
#         score_sum.append(scores)
#         ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
#         im = ax.imshow(scores)
#         plt.colorbar(im, ax=ax)
#         ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_xticklabels([-90,45,0,45,90])
#         ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interv_n]); ax.set_yticklabels([-90,45,0,45,90])
#         ax.set_title(title+"_Hemisphere")
#     figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
#     # figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.png" % (self.explabel, self.sphere_norm)))
#     # # figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.pdf" % (self.explabel, self.sphere_norm)))
#     # self.Perturb_vec = np.concatenate(tuple(self.Perturb_vec), axis=0)
#     # np.save(join(self.savedir, "Manifold_score_%s" % (self.explabel)), self.score_sum)
#     # np.savez(join(self.savedir, "Manifold_set_%s.npz" % (self.explabel)),
#     #          Perturb_vec=self.Perturb_vec, imgsize=self.imgsize, corner=self.corner,
#     #          evol_score=self.scores_all, evol_gen=self.generations, sphere_norm=self.sphere_norm)
#     return self.score_sum, figsum