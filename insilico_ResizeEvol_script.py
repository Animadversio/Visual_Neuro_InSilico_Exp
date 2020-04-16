#%% Preparation for RF computation.
import torchvision
from torch_net_utils import receptive_field, receptive_field_for_unit, layername_dict
# alexnet = torchvision.models.AlexNet()  # using the pytorch alexnet as proxy for caffenet.
# rf_dict = receptive_field(alexnet.features, (3, 227, 227), device="cpu")
# layer_name_map = {"conv1": "1", "conv2": "4", "conv3": "7", "conv4": "9", "conv5": "11"}  # how names in unit tuple maps to the
vgg16 = torchvision.models.vgg16()  # using the pytorch alexnet as proxy for caffenet.
rf_dict = receptive_field(vgg16.features, (3, 227, 227), device="cpu")
layername = layername_dict["vgg16"]
layer_name_map = {}
for i in range(31):
    layer = layername[i]
    layer_name_map[layer] = str(i+1)
#%%
from insilico_Exp import *
from time import time
plt.ioff()
import matplotlib
matplotlib.use('Agg')
#%%
# savedir = join(recorddir, "resize_data")
layer_list = ["conv5", "conv4", "conv3", "conv1", "conv2"]  #
unit_arr = [('caffe-net', 'conv1', 5, 28, 28),
            ('caffe-net', 'conv2', 5, 13, 13),
            ('caffe-net', 'conv3', 5, 7, 7),
            ('caffe-net', 'conv4', 5, 7, 7),
            ('caffe-net', 'conv5', 10, 7, 7),
            ]
unit_arr = [('vgg16', 'fc1', 10), ]
GANspace = "fc7"
for units in unit_arr:
    netname = units[0]
    layer = units[1]
    savedir = join(recorddir, "resize_data", "%s_%s_manifold-%s" % (netname, layer, GANspace))
    os.makedirs(savedir, exist_ok=True)
    for channel in range(1, 51):
        if len(units) == 5:
            unit = (netname, layer, channel, units[3], units[4])
            unit_lab = "%s_%d_%d_%d" % (unit[1], unit[2], unit[3], unit[4])
        elif len(units) == 3:
            unit = (netname, layer, channel, )
            unit_lab = "%s_%d" % (unit[1], unit[2])
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
        exp = ExperimentManifold(unit, max_step=150, imgsize=(227, 227), corner=(0, 0), backend="torch", savedir=savedir,
                                 explabel="%s_original" % (unit_lab), GAN=GANspace)
        # exp.load_traj("Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, _ = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
        np.save(join(savedir, "Manifold_score_%s_orig" %
                     (unit_lab)), score_sum)
        np.savez(join(savedir, "Manifold_set_%s_orig.npz" %
                      (unit_lab)),
                 Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
                 evol_score=exp.scores_all, evol_gen=exp.generations)
        plt.clf()
        t1 = time()
        print("Original Exp Processing time %.f" % (t1 - t0))
        # Resized Manifold experiment
        exp = ExperimentManifold(unit, max_step=150, imgsize=imgsize, corner=corner, backend="torch", savedir=savedir,
                                 explabel="%s_rf_fit" % (unit_lab), GAN=GANspace)
        # exp.load_traj("Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, _ = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9)
        np.save(join(savedir, "Manifold_score_%s_rf_fit" % (unit_lab)), score_sum)
        np.savez(join(savedir, "Manifold_set_%s_rf_fit.npz" % (unit_lab)),
                 Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
                 evol_score=exp.scores_all, evol_gen=exp.generations)
        plt.clf()
        plt.close("all")
        t2 = time()
        print("Pair Processing time %.f" % (t2-t0) )
        print("Existing figures %d" % (len(plt.get_fignums())))
        break
    break
#%%
# img_tmp = generator.visualize(exp.Perturb_vec[3,:])
# fig = visualize_img_list([img_tmp]*121, scores=np.random.randn(121), ncol=11, nrow=11, title_cmap=plt.cm.viridis, show=False, title_str="")
# fig.savefig(join(savedir,"tmp.jpg"))
# #%%
# fig.subplots_adjust(left=0.005,bottom=0.005,right=0.995,top=0.97, wspace=0.025, hspace=0.16)
# fig.savefig(join(savedir,"tmp.jpg"))
#%%

# savedir = join(recorddir, "resize_data")
# os.makedirs(savedir, exist_ok=True)
# unit_arr = [#('caffe-net', 'conv5', 5, 10, 10),
#             ('caffe-net', 'conv5', 6, 10, 10),
#             ('caffe-net', 'conv5', 7, 10, 10),
#             ('caffe-net', 'conv5', 8, 10, 10),
#             ('caffe-net', 'conv5', 9, 10, 10),
#             ('caffe-net', 'conv5', 10, 10, 10),
#             # ('caffe-net', 'conv4', 5, 10, 10),
#             # ('caffe-net', 'conv3', 5, 10, 10),
#             # ('caffe-net', 'conv2', 5, 10, 10),
#             # ('caffe-net', 'conv1', 5, 10, 10),
#             # ('caffe-net', 'fc6', 1),
#             # ('caffe-net', 'fc7', 1),
#             # ('caffe-net', 'fc8', 1),
#             ]
# layer_list = ["conv1", "conv3", "conv5", "conv2", "conv4"]
# for layer in layer_list:
#     for channel in range(1, 51):
#         unit = ('caffe-net', layer, channel, 7, 7)
#         print(unit)
#     #for unit in unit_arr:
#         if "conv" in unit[1]:
#             rf_pos = receptive_field_for_unit(rf_dict, (3, 227, 227), layer_name_map[unit[1]], (unit[3], unit[4]))
#             imgsize = (int(rf_pos[0][1] - rf_pos[0][0]), int(rf_pos[1][1] - rf_pos[1][0]))
#             corner = (int(rf_pos[0][0]), int(rf_pos[1][0]))
#         else:
#             rf_pos = [(0, 227), (0, 227)]
#             imgsize = (227, 227)
#             corner = (0, 0)
#         exp = ExperimentResizeEvolve(unit, imgsize=imgsize, corner=corner,
#                                          max_step=100, savedir=savedir, explabel="%s_%d_rf_fit" % (unit[1], unit[2]))
#         exp.run()
#         exp.visualize_best()
#         exp.visualize_trajectory()
#         exp.visualize_exp()
#         np.savez(join(savedir, "Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4])), scores_all=exp.scores_all,
#                  codes_all=exp.codes_all, generations=exp.generations)
#
#         expo = ExperimentResizeEvolve(unit, imgsize=(227, 227), corner=(0, 0),
#                                      max_step=100, savedir=savedir, explabel="%s_%d_origin" % (unit[1], unit[2]))
#         expo.run()
#         expo.visualize_best()
#         expo.visualize_trajectory()
#         expo.visualize_exp()
#         np.savez(join(savedir, "Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4])), scores_all=expo.scores_all,
#                  codes_all=expo.codes_all, generations=expo.generations)  # Bug is here the codes saved is expo
#         plt.close("all")