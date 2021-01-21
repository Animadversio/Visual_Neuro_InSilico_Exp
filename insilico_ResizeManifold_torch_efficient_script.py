# units=("resnet50_linf_8", ".Linearfc", 5); chan_rng=(0, 75);
# units=("resnet50_linf_8", ".layer3.Bottleneck2", 5, 7, 7); Xlim=(47, 184); Ylim=(47, 184); imgsize=(137, 137); corner=(47, 47); RFfit=True; chan_rng=(0, 75);
# %% Preparation for RF computation.
import matplotlib.pylab as plt
plt.ioff()
import matplotlib
matplotlib.use('Agg')
from insilico_Exp_torch import *
from time import time
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--units', nargs='+', type=str, required=True)
parser.add_argument('--chan_rng', nargs=2, type=int, default=[0, 75])
parser.add_argument('--RFfit', action='store_true')  # will be false if not specified.
parser.add_argument('--imgsize', nargs=2, type=int, default=[227, 227])
parser.add_argument('--corner', nargs=2, type=int, default=[0, 0])
parser.add_argument('--evosteps', type=int, default=100)
args = parser.parse_args()

# %%
# units = ("vgg16", "conv10", 5, 14, 14);
# layer_list = ["conv5", "conv4", "conv3", "conv1", "conv2"]  #
# unit_arr = [('caffe-net', 'conv5', 10, 7, 7),
#             ('caffe-net', 'conv1', 5, 28, 28),
#             ('caffe-net', 'conv2', 5, 13, 13),
#             ('caffe-net', 'conv3', 5, 7, 7),
#             ('caffe-net', 'conv4', 5, 7, 7),
#             ]
# for units in unit_arr:
recorddir = "/scratch/binxu/CNN_data/"
GANspace = ""  # default for GANspace "" which is FC6 GAN.
netname = args.units[0]
layer = args.units[1]

savedir = join(recorddir, "manif_allchan", "%s_%s_manifold-%s" % (netname, layer, GANspace))
os.makedirs(savedir, exist_ok=True)

RFfit = args.RFfit
imgsize = tuple(args.imgsize) if RFfit else (227, 227)  # have to be a tuple for resizing to behave correctly
corner = tuple(args.corner) if RFfit else (0, 0)
Xlim = (corner[0], corner[0]+imgsize[0])
Ylim = (corner[1], corner[1]+imgsize[1])
chan_rng = args.chan_rng

if len(args.units) == 5:
    units = (netname, layer, int(args.units[2]), int(args.units[3]), int(args.units[4]))
    print("Exp Config: Unit %s %s (%d, %d)\n corner: %s imgsize: %s\n Xlim %s Ylim %s" % (
        units[0], units[1], units[3], units[4], corner, imgsize, Xlim, Ylim))
elif len(args.units) == 3:
    units = (netname, layer, int(args.units[2]))
    print("Exp Config: Unit %s %s\n corner: %s imgsize: %s\n Xlim %s Ylim %s" % (
        units[0], units[1], corner, imgsize, Xlim, Ylim))
else:
    raise ValueError("args.units should be a 3 element or 5 element tuple!")

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
        exp = ExperimentManifold(unit, max_step=args.evosteps, imgsize=(227, 227), corner=(0, 0), backend="torch",
                                 savedir=savedir, explabel="%s_original" % (unit_lab))
        # exp.load_traj("Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.save_last_gen()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, figsum = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9, print_manifold=False)
        plt.close(figsum)
        # np.save(join(savedir, "Manifold_score_%s_orig" % (unit_lab)), score_sum)
        # np.savez(join(savedir, "Manifold_set_%s_orig.npz" % (unit_lab)),
        #          Perturb_vec=exp.Perturb_vec, imgsize=exp.imgsize, corner=exp.corner,
        #          evol_score=exp.scores_all, evol_gen=exp.generations)
        t1 = time()
        print("Original Exp Processing time %.f" % (t1 - t0))
    else:
        # Resized Manifold experiment
        exp = ExperimentManifold(unit, max_step=args.evosteps, imgsize=imgsize, corner=corner, backend="torch", savedir=savedir, explabel="%s_rf_fit" % (unit_lab))
        # exp.load_traj("Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4]))  # load saved traj
        exp.run()
        exp.save_last_gen()
        exp.analyze_traj()
        exp.visualize_trajectory()
        exp.visualize_best()
        score_sum, figsum = exp.run_manifold([(1, 2), (24, 25), (48, 49), "RND"], interval=9, print_manifold=False)
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


