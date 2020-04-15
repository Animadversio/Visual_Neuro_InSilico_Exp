#%% Preparation for RF computation.
import torchvision
from torch_net_utils import receptive_field, receptive_field_for_unit, layername_dict
# alexnet = torchvision.models.AlexNet()  # using the pytorch alexnet as proxy for caffenet.
# rf_dict = receptive_field(alexnet.features, (3, 227, 227), device="cpu")
# layer_name_map = {"conv1": "1", "conv2": "4", "conv3": "7", "conv4": "9", "conv5": "11"}
vgg16 = torchvision.models.vgg16()  # using the pytorch alexnet as proxy for caffenet.
rf_dict = receptive_field(vgg16.features, (3, 227, 227), device="cpu")
layername = layername_dict["vgg16"]
layer_name_map = {}
for i in range(31):
    layer = layername[i]
    layer_name_map[layer] = str(i+1)
# how names in unit tuple maps to the numering in rf_dict. Can use this to fetch rf in exp
#%%
from insilico_Exp import *
from time import time
plt.ioff()
import matplotlib
matplotlib.use('Agg')
#%%
# units = ("vgg16", "conv10", 5, 14, 14);
# layer_list = ["conv5", "conv4", "conv3", "conv1", "conv2"]  #
# unit_arr = [('caffe-net', 'conv5', 10, 7, 7),
#             ('caffe-net', 'conv1', 5, 28, 28),
#             ('caffe-net', 'conv2', 5, 13, 13),
#             ('caffe-net', 'conv3', 5, 7, 7),
#             ('caffe-net', 'conv4', 5, 7, 7),
#             ]
#for units in unit_arr:
GANspace = ""
netname = units[0]
layer = units[1]
savedir = join(recorddir, "resize_data", "%s_%s_manifold-%s" % (netname, layer, GANspace))
os.makedirs(savedir, exist_ok=True)
for channel in range(1, 51):
    if len(units) == 5:
        unit = (netname, layer, channel, units[3], units[4])
    elif len(units) == 3:
        unit = (netname, layer, channel, )
    if "conv" in layer:
        rf_pos = receptive_field_for_unit(rf_dict, (3, 227, 227), layer_name_map[layer], (unit[3], unit[4]))
        imgsize = (int(rf_pos[0][1] - rf_pos[0][0]), int(rf_pos[1][1] - rf_pos[1][0]))
        corner = (int(rf_pos[0][0]), int(rf_pos[1][0]))
    else:
        rf_pos = [(0, 227), (0, 227)]
        imgsize = (227, 227)
        corner = (0, 0)
    # Original experiment
    t0 = time()
    exp = ExperimentManifold(unit, max_step=100, imgsize=(227, 227), corner=(0, 0), backend="torch", savedir=savedir,
                             explabel="%s_%d_%d_%d_original" % (unit[1], unit[2], unit[3], unit[4]))
    # exp.load_traj("Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
    exp.run()
    exp.analyze_traj()
    exp.visualize_trajectory()
    exp.visualize_best()
    score_sum, _ = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
    np.save(join(savedir, "Manifold_score_%s_%d_%d_%d_orig" %
                 (unit[1], unit[2], unit[3], unit[4])), score_sum)
    np.savez(join(savedir, "Manifold_set_%s_%d_%d_%d_orig.npz" %
                  (unit[1], unit[2], unit[3], unit[4])),
             Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
             evol_score=exp.scores_all, evol_gen=exp.generations)
    plt.clf()
    t1 = time()
    print("Original Exp Processing time %.f" % (t1 - t0))
    # Resized Manifold experiment
    exp = ExperimentManifold(unit, max_step=100, imgsize=imgsize, corner=corner, backend="torch", savedir=savedir,
                             explabel="%s_%d_%d_%d_rf_fit" % (unit[1], unit[2], unit[3], unit[4]))
    # exp.load_traj("Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
    exp.run()
    exp.analyze_traj()
    exp.visualize_trajectory()
    exp.visualize_best()
    score_sum, _ = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
    np.save(join(savedir, "Manifold_score_%s_%d_%d_%d_rf_fit" %
                 (unit[1], unit[2], unit[3], unit[4])), score_sum)
    np.savez(join(savedir, "Manifold_set_%s_%d_%d_%d_rf_fit.npz" %
                  (unit[1], unit[2], unit[3], unit[4])),
             Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
             evol_score=exp.scores_all, evol_gen=exp.generations)
    plt.clf()
    plt.close("all")
    t2 = time()
    print("Pair Processing time %.f" % (t2-t0) )
    print("Existing figures %d" % (len(plt.get_fignums())))
    
#%%
