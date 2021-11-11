from os.path import join
import numpy as np
from random import randint
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dataroot = r"E:\Cluster_Backup\CNN_manifold"
sumdir = r"E:\Cluster_Backup\CNN_manifold\summary"
#%%
from Manifold.Kent_fit_utils import fit_Kent_Stats

#%%
unit_list = [("vgg16", "conv2", 5, 112, 112),
            ("vgg16", "conv3", 5, 56, 56),
            ("vgg16", "conv4", 5, 56, 56),
            ("vgg16", "conv5", 5, 28, 28),
            ("vgg16", "conv6", 5, 28, 28),
            ("vgg16", "conv7", 5, 28, 28),
            # ("vgg16", "conv9", 5, 14, 14),
            ("vgg16", "conv10", 5, 14, 14),
            ("vgg16", "conv12", 5, 7, 7),
            ("vgg16", "conv13", 5, 7, 7)]
subsp_nm = ["PC23", "PC2526", "PC4950", "RND12"]
theta_arr = np.arange(-90, 90.1, 9) / 180 * np.pi
phi_arr = np.arange(-90, 90.1, 9) / 180 * np.pi
interv_n = 10
def map_layout(ax, title, interv_n=interv_n, xylabel=True):
    ax.set_title(title)
    ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2 * interv_n]);
    ax.set_xticklabels([-90, 45, 0, 45, 90])
    ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2 * interv_n]);
    ax.set_yticklabels([-90, 45, 0, 45, 90])
    if xylabel:
        ax.set_ylabel("PC2")
        ax.set_xlabel("PC3")

fig = plt.figure(figsize=[20, 2.25])
fig_rf = plt.figure(figsize=[20, 2.25])
axs = fig.subplots(1, len(unit_list))
axs_rf = fig_rf.subplots(1, len(unit_list))
for i, unit in enumerate(unit_list):
    subfolder = "%s_%s_manifold" % (unit[0], unit[1])
    layerdir = join(dataroot, subfolder)
    while True:
        iCh = randint(1, 50)
        unit_lab = "%s_%d_%d_%d" % (unit[1], iCh, unit[3], unit[4])
        Edata = np.load(join(layerdir, "Manifold_set_%s_orig.npz" % unit_lab))
        # ['Perturb_vec', 'imgsize', 'corner', 'evol_score', 'evol_gen']
        Mdata = np.load(join(layerdir, "Manifold_score_%s_orig.npy" % unit_lab))
        Edata_rf = np.load(join(layerdir, "Manifold_set_%s_rf_fit.npz" % unit_lab))
        Mdata_rf = np.load(join(layerdir, "Manifold_score_%s_rf_fit.npy" % unit_lab))
        # ax = fig.add_subplot(1, len(subspace_list), spi + 1)
        actmap = Mdata[0, :, :]
        actmap_rf = Mdata_rf[0, :, :]
        if not np.isclose(actmap.std(), 0.0) and not np.isclose(actmap_rf.std(), 0.0):
            break
    param, _, _, R2 = fit_Kent_Stats(theta_arr=theta_arr, phi_arr=phi_arr, act_map=actmap)
    titstr = "%s Ch%d\nk%.2f R2:%.2f"%(unit[1], iCh, param[3], R2)
    im = axs[i].imshow(actmap)
    plt.colorbar(im, ax=axs[i])
    map_layout(axs[i], titstr, interv_n=interv_n, xylabel=(i == 0))
    param_rf, _, _, R2_rf = fit_Kent_Stats(theta_arr=theta_arr, phi_arr=phi_arr, act_map=actmap_rf)
    titstr = "%s Ch%d\nk%.2f R2:%.2f"%(unit[1], iCh, param_rf[3], R2_rf)
    im = axs_rf[i].imshow(actmap_rf)
    plt.colorbar(im, ax=axs_rf[i])
    map_layout(axs_rf[i], titstr, interv_n=interv_n, xylabel=(i == 0))
fig.savefig(join(sumdir, "Progress_Sample.png"))
fig.savefig(join(sumdir, "Progress_Sample.pdf"))
fig_rf.savefig(join(sumdir, "Progress_Sample_rffit.png"))
fig_rf.savefig(join(sumdir, "Progress_Sample_rffit.pdf"))
fig.show()
fig_rf.show()

#%% Get FC layers results 
unit_list2 = [("vgg16", "fc1", 5),
            ("vgg16", "fc2", 5),
            ("vgg16", "fc3", 5)]
dataroot2 = r"E:\OneDrive - Washington University in St. Louis\Artiphysiology\Manifold"
fig_fc = plt.figure(figsize=[7, 2.25])
axs_fc = fig_fc.subplots(1, len(unit_list2))
for i, unit in enumerate(unit_list2):
    subfolder = "%s_%s_manifold" % (unit[0], unit[1])
    layerdir = join(dataroot2, subfolder)
    while True:
        iCh = randint(0, 49)
        Mdata = np.load(join(layerdir, "score_map_chan%d.npz" % iCh))
        actmap = Mdata["score_sum"][0, :, :]
        if not np.isclose(actmap.std(), 0.0):
            break
    param_fc, _, _, R2_fc = fit_Kent_Stats(theta_arr=theta_arr, phi_arr=phi_arr, act_map=actmap)
    titstr = "%s Ch%d\nk%.2f R2:%.2f"%(unit[1], iCh, param_fc[3], R2_fc)
    # ax = fig.add_subplot(1, len(subspace_list), spi + 1)
    im = axs_fc[i].imshow(actmap)
    plt.colorbar(im, ax=axs_fc[i])
    map_layout(axs_fc[i], titstr, interv_n=interv_n, xylabel=(i == 0))
fig_fc.savefig(join(sumdir, "Progress_Sample_FC.png"))
fig_fc.savefig(join(sumdir, "Progress_Sample_FC.pdf"))
fig_fc.show()