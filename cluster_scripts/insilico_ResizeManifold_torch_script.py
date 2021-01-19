# %% Preparation for RF computation.
# units=("resnet50_linf_8", ".layer3.Bottleneck2", 5, 7, 7); Xlim=(47, 184); Ylim=(47, 184); imgsize=(137, 137); corner=(47, 47); RFfit=True; chan_rng=(0, 75);
import matplotlib.pylab as plt
plt.ioff()
import matplotlib
matplotlib.use('Agg')
from insilico_Exp_torch import *
# from insilico_Exp import * # Obsolete, use torch version instead. 
import torchvision, torch
# alexnet = torchvision.models.AlexNet()  # using the pytorch alexnet as proxy for caffenet.
# rf_dict = receptive_field(alexnet.features, (3, 227, 227), device="cpu")
# layer_name_map = {"conv1": "1", "conv2": "4", "conv3": "7", "conv4": "9", "conv5": "11"}
# vgg16 = torchvision.models.vgg16()  # using the pytorch alexnet as proxy for caffenet.
# rf_dict = receptive_field(vgg16.features, (3, 227, 227), device="cpu")
# layername = layername_dict["vgg16"]
# layer_name_map = {}
# for i in range(31):
#     layer = layername[i]
#     layer_name_map[layer] = str(i + 1)
# how names in unit tuple maps to the numering in rf_dict. Can use this to fetch rf in exp
# %%
from time import time
# %%
# units = ("vgg16", "conv10", 5, 14, 14);
# layer_list = ["conv5", "conv4", "conv3", "conv1", "conv2"]  #
# unit_arr = [('caffe-net', 'conv5', 10, 7, 7),
#             ('caffe-net', 'conv1', 5, 28, 28),
#             ('caffe-net', 'conv2', 5, 13, 13),
#             ('caffe-net', 'conv3', 5, 7, 7),
#             ('caffe-net', 'conv4', 5, 7, 7),
#             ]
recorddir = "/scratch/binxu/CNN_data/"
# for units in unit_arr:
GANspace = "" # default for GANspace "" which is FC6 GAN. 
netname = units[0]
layer = units[1]
savedir = join(recorddir, "resize_data", "%s_%s_manifold-%s" % (netname, layer, GANspace))
os.makedirs(savedir, exist_ok=True)
try:
    RFfit
except NameError:
    RFfit = False
    imgsize = (227, 227)
    corner = (0, 0)
    Xlim = (corner[0], corner[0]+imgsize[0])
    Ylim = (corner[1], corner[1]+imgsize[1])
    print("RF info not found from config, no image resizing")
else:
    Xlim = (corner[0], corner[0]+imgsize[0])
    Ylim = (corner[1], corner[1]+imgsize[1])
    print("RF info found from config!")

try:
    chan_rng
except NameError:
    chan_rng = (0, 75)
    print("Use the default channel range %d %d"%chan_rng)
else:
    print("Use the user defined channel range %d %d"%chan_rng)

print("Exp Config: Unit %s %s (%d, %d)\n corner: %s imgsize: %s\n Xlim %s Ylim %s"%(units[0], units[1], units[3], units[4], corner, imgsize, Xlim, Ylim))

for channel in range(chan_rng[0], chan_rng[1]):
    if len(units) == 5:
        unit = (netname, layer, channel, units[3], units[4])
        unit_lab = "%s_%d_%d_%d" % (unit[1], unit[2], unit[3], unit[4])
    elif len(units) == 3:
        unit = (netname, layer, channel,)
        unit_lab = "%s_%d" % (unit[1], unit[2])
    t0 = time()
    if not RFfit:
        # Original experiment
        exp = ExperimentManifold(unit, max_step=100, imgsize=(227, 227), corner=(0, 0), backend="torch", savedir=savedir, explabel="%s_original" % (unit_lab))
        # exp.load_traj("Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, figsum = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
        plt.close(figsum)
        # np.save(join(savedir, "Manifold_score_%s_orig" % (unit_lab)), score_sum)
        # np.savez(join(savedir, "Manifold_set_%s_orig.npz" % (unit_lab)),
        #          Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
        #          evol_score=exp.scores_all, evol_gen=exp.generations)
        t1 = time()
        print("Original Exp Processing time %.f" % (t1 - t0))
    else:
        # Resized Manifold experiment
        exp = ExperimentManifold(unit, max_step=100, imgsize=imgsize, corner=corner, backend="torch", savedir=savedir, explabel="%s_rf_fit" % (unit_lab))
        # exp.load_traj("Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, figsum = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
        plt.close(figsum)
        # np.save(join(savedir, "Manifold_score_%s_rf_fit" % (unit_lab)), score_sum)
        # np.savez(join(savedir, "Manifold_set_%s_rf_fit.npz" % (unit_lab)),
        #          Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
        #          evol_score=exp.scores_all, evol_gen=exp.generations)
        t2 = time()
        print("RF fitting Exp Processing time %.f" % (t2 - t0))
    plt.close("all")
    print("Existing figures %d" % (len(plt.get_fignums())))
    # print("Pair Processing time %.f" % (t2 - t0))

# %%
