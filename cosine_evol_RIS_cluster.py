from Cosine.cosine_evol_lib import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--net", type=str, default="resnet50", help="Network model to use for Image distance computation")
parser.add_argument("--layer", type=str, default=".layer3.Bottleneck0", help="Network model to use for Image distance computation")
parser.add_argument("--popsize", type=int, default=500, help="Number of units in the population recording")
parser.add_argument("--pop_rand_seed", type=int, default=0, help="Random seed to reproduce population selection across machines")

parser.add_argument("--target_idx", type=int, nargs="+", default=[0, 150], 
            help="Random seed to reproduce population selection across machines")
parser.add_argument("--G", type=str, default="fc6", help="")
parser.add_argument("--optim", type=str, default="CholCMA", nargs=1, 
    choices=["HessCMA", "HessCMA_class", "CholCMA", "CholCMA_prod", "CholCMA_class"], help="")
parser.add_argument("--score_method", type=str, default=["cosine", "corr", "MSE", "dot"], nargs="+", 
    choices=["cosine", "corr", "dot", "MSE", "L1"], help="")
parser.add_argument("--steps", type=int, default=100, help="Evolution Steps")
parser.add_argument("--reps", type=int, default=5, help="Number of replications for each condition")
parser.add_argument("--RFresize", action='store_true', help="Resize image to RFsize during reconstruction?")
parser.add_argument("--resize_ref", action='store_true', help="Resize image to RFsize for reference and target images?")
# ["--G", "BigGAN", "--optim", "HessCMA", "CholCMA","--chans",'1','2','--steps','100',"--reps",'2']

import sys 
import os
from os.path import join
if sys.platform == "linux":
    exproot = "/scratch1/fs1/crponce/Cosine_insilico"
    refimgdir = "/scratch1/fs1/crponce/cos_refimgs"
    scratchdir = os.environ['SCRATCH1']
    rootdir = join(scratchdir, "GAN_Evol_cmp")
    Hdir_BigGAN = join(scratchdir, "Hessian", "H_avg_1000cls.npz")  #r"/scratch/binxu/GAN_hessian/BigGAN/summary/H_avg_1000cls.npz"
    Hdir_fc6 = join(scratchdir, "Hessian", "Evolution_Avg_Hess.npz")  #r"/scratch/binxu/GAN_hessian/FC6GAN/summary/Evolution_Avg_Hess.npz"
    sys.path.append("/home/binxu.w/Visual_Neuro_InSilico_Exp")

else:
    exproot = r"E:\insilico_exps\Cosine_insilico"
    refimgdir = r"E:\Network_Data_Sync\Stimuli\cos_refimgs"
    # refimgdir = r"E:\Network_Data_Sync\Stimuli\2019-Selectivity\2019-Selectivity-Big-Set-01"

os.makedirs(join(exproot, "popul_idx"), exist_ok=True)
import pickle as pkl
from easydict import EasyDict
if __name__=="__main__":
    #%%
    args = parser.parse_args() 
    from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
    from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
    from grad_RF_estim import grad_RF_estimate, gradmap2RF_square
    score_method_col = args.score_method
    # Set population recording
    print("Load up CNN scorer", args.net)
    scorer = TorchScorer(args.net)
    module_names, module_types, module_spec = get_module_names(scorer.model, (3, 227, 227), "cuda", False)

    print("Set up recording hooks in CNN: ", args.layer)
    pop_RND = args.pop_rand_seed
    if args.pop_rand_seed == -1:
        print("using a random number as seed to sample random population")
        args.pop_rand_seed = np.random.randint(1E4)
    unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [args.layer], popsize=args.popsize, 
            seed=pop_RND)

    print("Computing RF by direct backprop (RFMapping)...")
    Cids, Hids, Wids = unit_tsridx_dict[args.layer]
    gradAmpmap = grad_RF_estimate(scorer.model, args.layer, (slice(None), Hids[0], Wids[0]),
                                  input_size=(3, 227, 227), device="cuda", show=False, reps=30, batch=1)
    Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
    corner = (Xlim[0], Ylim[0])
    imgsize = (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0])
    print("Xlim %s Ylim %s \n imgsize %s corner %s" % (Xlim, Ylim, imgsize, corner))

    print("Encode a population of images to set the normalizer and mask.")
    refimgnms, refimgtsr = load_ref_imgs(imgdir=refimgdir,
                                         preprocess=Compose([Resize((227, 227)), ToTensor()]))
    ref_actmat = encode_image(scorer, refimgtsr, key=args.layer,
                              RFresize=args.resize_ref, corner=corner, imgsize=imgsize)
    popul_m, popul_s = set_normalizer(ref_actmat)
    popul_mask = set_popul_mask(ref_actmat)
    print("Save basic information and selectivity of the population.")
    pkl.dump(EasyDict(unit_mask_dict=unit_mask_dict, unit_tsridx_dict=unit_tsridx_dict,
             refimgnms=refimgnms, ref_actmat=ref_actmat, popul_m=popul_m, popul_s=popul_s, popul_mask=popul_mask,
             corner=corner, imgsize=imgsize, gradAmpmap=gradAmpmap, args=args.__dict__),
             open(join(exproot, "popul_idx", "popul_idx_%s-%s-%d-%04d.pkl" % (args.net, args.layer, args.popsize, pop_RND)), "wb"))
    # np.savez(join(exproot, "popul_idx", "popul_idx_%s_%s_%06d.npz" % (args.net, args.layer, pop_RND)),
    #          unit_mask_dict=unit_mask_dict, unit_tsridx_dict=unit_tsridx_dict,
    #          refimgnms=refimgnms, ref_actmat=ref_actmat, popul_m=popul_m, popul_s=popul_s, popul_mask=popul_mask,
    #          corner=corner, imgsize=imgsize, gradAmpmap=gradAmpmap, args=args.__dict__, )

    GANname = args.G
    if GANname == "fc6":
        G = upconvGAN("fc6").cuda()
        G.requires_grad_(False)
        code_length = G.codelen
    else:
        raise NotImplementedError

    np.random.seed(None)
    LB = args.target_idx[0]
    UB = min(args.target_idx[1], len(refimgnms))  # not over reach
    for imgid in range(LB, UB):
        # Select target image and add target vector. 
        targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
        target_imgtsr_rsz = resize_and_pad_tsr(target_imgtsr, imgsize, corner, ) if args.resize_ref \
                 else target_imgtsr

        targ_actmat = encode_image(scorer, target_imgtsr, key=args.layer,
                                   RFresize=args.resize_ref, corner=corner, imgsize=imgsize)  # 1, unitN
        targlabel = os.path.splitext(targnm)[0]
        # organize data with the targetlabel
        expdir = os.path.join(exproot, "rec_%s%s"%(targlabel, "_rsz" if args.resize_ref else ""))
        os.makedirs(expdir, exist_ok=True)
        for triali in range(args.reps):
            for score_method in score_method_col:
                print(f"Current experiment {args.net}-{args.layer}-{pop_RND} (N={args.popsize})\n"
                      f"Target {targlabel}, Objective {score_method}, GAN {GANname}, Optim {args.optim}, trial {triali}")
                explabel = "%s-%s-%d-%04d_%s_%s"%(args.net, args.layer, args.popsize, pop_RND, score_method, GANname, )
                objfunc = set_objective(score_method, targ_actmat, popul_mask, popul_m, popul_s)
                optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                maximize=True, random_seed=None, optim_params={})
                codes_all, scores_all, actmat_all, generations, RND = run_evol(scorer, objfunc, optimizer, G, 
                            reckey=args.layer, label=explabel, savedir=expdir,
                            steps=args.steps, RFresize=args.RFresize, corner=corner, imgsize=imgsize,)
                figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
                figh.savefig(join(expdir, "popul_act_evol_%s_%05d.png" % (explabel, RND)))
                ToPILImage()(target_imgtsr_rsz[0]).save(join(expdir, "targetimg_%s_%05d.png" % (explabel, RND)))
