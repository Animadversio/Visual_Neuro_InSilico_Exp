from NN_PC_visualize.NN_PC_lib import *
from robustCNN_utils import load_pretrained_robust_model
"""Visualize the SVD directions by different means """
#% Load up FC6 GAN
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)

#%% Load up svd tensor
netname = "resnet50_linf8"
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*tsr_svds.keys()]
feat_mean, U, S, V = tsr_svds[reclayers[0]]

#%% Load up network
model, model_full = load_featnet("resnet50_linf8")
model.eval().cuda()
model.requires_grad_(False)

#%% Full image version (no rf fit)
score_method = "cosine"
for PCi in range(50, 100):
    for layeri in range(3):
        layername = reclayers[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=outdir,
                                             savestr="%s_%s_PC%03d_pos_%s"%(netname, layersn, PCi, score_method),
                                             lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8,)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=outdir,
                                             savestr="%s_%s_PC%03d_neg_%s"%(netname, layersn, PCi, score_method),
                                             lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8,)


#%% RF fit version
#%% precompute the rf locations for recorded units.
rfdict = {}
for layeri in range(3):
    cent_pos = get_cent_pos(model, reclayers[layeri], imgfullpix=256)
    corner, imgpix = get_RF_location(model, reclayers[layeri], cent_pos, imgfullpix=256)
    rfdict[layeri] = (corner, imgpix)

#%% RF fit version (add normalization now)
figdir = join(outdir, "RFfit_norm_vis")
score_method = "cosine"
for PCi in range(0, 100):
    for layeri in range(3):
        layername = reclayers[layeri]
        corner, imgpix = rfdict[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_pos_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8, imcorner=corner, imgpix=imgpix,)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_neg_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.02, langevin_eps=0.01, MAXSTEP=150, Bsize=8, imcorner=corner, imgpix=imgpix,)

#%% We need to rethink the objective, cosine or something else.....
#%% Using normalized image but much larger learning rate , larger explore and shorter time.
figdir = join(outdir, "RFfit_norm_lr_vis")
score_method = "cosine"
for PCi in range(100): #:range(4,5)
    for layeri in range(3):
        layername = reclayers[layeri]
        corner, imgpix = rfdict[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_pos_%s_RFfit_norm_nogradnorm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=False)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_neg_%s_RFfit_norm_nogradnorm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=False)

#%%
"""
The methods above maximize cosine of featvec and PC direction 
Here we maximize cosine of (featvec - featmean) and PC direction. 
so not it's relative to the center of the distribution
"""
figdir = join(outdir, "RFfit_norm_lr_reldir_vis")
score_method = "cosine_dir"
for PCi in range(100): #:range(4,5)
    for layeri in range(3):
        layername = reclayers[layeri]
        corner, imgpix = rfdict[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda(), featmean=feat_mean.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_pos_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=True)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_neg_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=True)

#%%
#%% Load up svd tensor
netname = "vgg16"
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*tsr_svds.keys()]
feat_mean, U, S, V = tsr_svds[reclayers[0]]

#%% Load up network
_, model = load_featnet(netname)
model.eval().cuda()
model.requires_grad_(False)
#%% Get all the rf data.
rfdict = {}
for layeri in range(len(reclayers)):
    try:
        cent_pos = get_cent_pos(model, reclayers[layeri], imgfullpix=256)
        corner, imgpix = get_RF_location(model, reclayers[layeri], cent_pos, imgfullpix=256)
    except NotImplementedError:
        corner = (0, 0)
        imgpix = 256
    rfdict[layeri] = (corner, imgpix)
#%% Evolve the VGG16 PCA directions.
figdir = join(outdir, "VGG16_RFfit_norm_lr_reldir_vis")
os.makedirs(figdir, exist_ok=True)
score_method = "cosine_dir"
for PCi in range(100):  #:range(4,5)
    for layeri in range(1, len(reclayers)):
        layername = reclayers[layeri]
        corner, imgpix = rfdict[layeri]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda(), featmean=feat_mean.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_pos_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=True)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_neg_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=8, imcorner=corner, imgpix=imgpix,gradnorm=True)


#%% Load up svd tensor fpr DenseNet-robust
netname = "densenet_robust"
tsr_svds = torch.load(join(outdir, "%s_INvalid_tsr_svds.pt"%(netname)))
reclayers = [*tsr_svds.keys()]
#%%
feat_mean, U, S, V = tsr_svds[reclayers[3]]
plt.plot(np.cumsum(S**2))
plt.show()
#%% Load up network
model = load_pretrained_robust_model("densenet")
# _, model = load_featnet(netname)
model.eval().cuda()
model.requires_grad_(False)
#%%
rfdict = \
{'.features._DenseBlockdenseblock1': ((109, 110), 40),
 '.features._DenseBlockdenseblock2': ((83, 83), 96),
 '.features._DenseBlockdenseblock3': ((0, 0), 256),
 '.features._DenseBlockdenseblock4': ((0, 0), 256)}
#%% Get all the rf data.
rfdict = {}
for layeri in range(len(reclayers)):
    layer = reclayers[layeri]
    try:
        cent_pos = get_cent_pos(model, reclayers[layeri], imgfullpix=256)
        corner, imgpix = get_RF_location(model, reclayers[layeri], cent_pos, imgfullpix=256)
    except NotImplementedError:
        corner = (0, 0)
        imgpix = 256
    rfdict[layer] = (corner, imgpix)
torch.cuda.empty_cache()
#%% Evolve the DenseNet-robust PCA directions.
figdir = join(outdir, "densenet_robust_RFfit_norm_lr_reldir_vis")
os.makedirs(figdir, exist_ok=True)
score_method = "cosine_dir"
for PCi in range(0, 100):  #:range(4,5)
    for layeri in range(1):#1, len(reclayers) #1 # layeri=0, PCi=1 is so hard to visualize constantly OOM
        layername = reclayers[layeri]
        corner, imgpix = rfdict[layername]
        feat_mean, U, S, V = tsr_svds[layername]
        targdir = V[:, PCi]
        layersn = shorten_layername(layername)
        objfunc = set_objective_torch(score_method, targdir.cuda(), featmean=feat_mean.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_pos_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=6, imcorner=corner, imgpix=imgpix,gradnorm=True)

        objfunc = set_objective_torch(score_method, -targdir.cuda())
        finimgs, mtg, score_traj = featdir_GAN_visualize(G, model, layername, objfunc, figdir=figdir, tfms=[normalize],
                                 savestr="%s_%s_PC%03d_neg_%s_RFfit_norm"%(netname, layersn, PCi, score_method),
                                 lr=0.1, langevin_eps=0.02, MAXSTEP=100, Bsize=6, imcorner=corner, imgpix=imgpix,gradnorm=True)







#%% Dev zone.
#%% CMA evolution
from Cosine.cosine_evol_lib import run_evol, sample_center_column_units_idx, set_objective, set_random_population_recording
from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
from ZO_HessAware_Optimizers import HessAware_Gauss_DC, CholeskyCMAES
from insilico_Exp_torch import TorchScorer, visualize_trajectory, resize_and_pad_tsr
import time

score_method = "cosine"
popul_mask = np.ones((2048,), dtype=bool)
popul_m = np.zeros((1, 2048,),)
popul_s = np.ones((1, 2048,),)

scorer = TorchScorer("resnet50_linf8")
module_names, module_types, module_spec = get_module_names(scorer.model, (3, 256, 256), "cuda", False)
unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [".layer4.Bottleneck2"], randomize=False)
#
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
code_length = G.codelen
objfunc = set_objective(score_method, components[1:2, :], popul_mask, popul_m, popul_s)
optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                maximize=True, random_seed=None, optim_params={})
codes_all, scores_all, actmat_all, generations, RND = run_evol(scorer, objfunc, optimizer, G,
                                   reckey=".layer4.Bottleneck2",
                               label="PC2-cosine", savedir=r"H:\CNN-PCs\resnet50-PC1",
                               steps=100, RFresize=False, corner=(0, 0), imgsize=(256, 256))
# codes_fin, scores_all, generations, RND = run_evol_lowmem
#corner=(20, 20), imgsize=(187, 187))
# def run_evol_lowmem(scorer, objfunc, optimizer, G, reckey=None, steps=100, label="obj-target-G", savedir="",
#             RFresize=True, corner=(0, 0), imgsize=(224, 224), init_code=None):
#     if init_code is None:
#         init_code = np.zeros((1, G.codelen))
#     RND = np.random.randint(1E5)
#     new_codes = init_code
#     # new_codes = init_code + np.random.randn(25, 256) * 0.06
#     scores_all = []
#     actmat_all = []
#     generations = []
#     # codes_all = []
#     best_imgs = []
#     for i in range(steps,):
#         # codes_all.append(new_codes.copy())
#         T0 = time.time() #process_
#         imgs = G.visualize_batch_np(new_codes)  # B=1
#         latent_code = torch.from_numpy(np.array(new_codes)).float()
#         T1 = time.time() #process_
#         if RFresize: imgs = resize_and_pad_tsr(imgs, imgsize, corner, )
#         T2 = time.time() #process_
#         _, recordings = scorer.score_tsr(imgs)
#         actmat = recordings[reckey]
#         T3 = time.time() #process_
#         scores = objfunc(actmat, )  # targ_actmat
#         T4 = time.time() #process_
#         new_codes = optimizer.step_simple(scores, new_codes, )
#         T5 = time.time() #process_
#         if "BigGAN" in str(G.__class__):
#             print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
#                 i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
#                 latent_code[:, :128].norm(dim=1).mean()))
#         else:
#             print("step %d score %.3f (%.3f) (norm %.2f )" % (
#                 i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
#         print(f"GANvis {T1-T0:.3f} RFresize {T2-T1:.3f} CNNforw {T3-T2:.3f}  "
#             f"objfunc {T4-T3:.3f}  optim {T5-T4:.3f} total {T5-T0:.3f}")
#         scores_all.extend(list(scores))
#         generations.extend([i] * len(scores))
#         best_imgs.append(imgs[scores.argmax(),:,:,:].detach().clone()) # debug at
#         if i < steps - 1:
#             del imgs
#     # codes_all = np.concatenate(tuple(codes_all), axis=0)
#     scores_all = np.array(scores_all)
#     generations = np.array(generations)
#     mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
#     mtg_exp.save(join(savedir, "besteachgen_%s_%05d.jpg" % (label, RND,)))
#     mtg = ToPILImage()(make_grid(imgs, nrow=7))
#     mtg.save(join(savedir, "lastgen_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
#     np.savez(join(savedir, "scores_%s_%05d.npz" % (label, RND)), generations=generations,
#              scores_all=scores_all, actmat_all=actmat_all, codes_fin=new_codes)#, codes_all=codes_all)
#     visualize_trajectory(scores_all, generations, title_str=label).savefig(
#         join(savedir, "traj_%s_%05d_score%.1f.jpg" % (label, RND, scores.mean())))
#     return new_codes, scores_all, generations, RND
#%%
