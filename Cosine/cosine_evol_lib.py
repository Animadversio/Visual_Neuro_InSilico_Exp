# Set objective function
def select_popul_record(model, layer, size=50, chan="rand", x=None, y=None):
    return popul_idcs

["cos", "MSE", "L1", "dot", "corr"]
def set_objective(score_method, grad=False):
    def objfunc(actmat, targmat):
        return 
    # return an array / tensor of scores for an array of activations 
    # Noise form 
    return objfunc 

def encode_image(imgtsr):
    """return a 2d array / tensor of activations for a image tensor 
    imgtsr: (Nimgs, C, H, W) 
    actmat: (Npop, Nimages) torch tensor
    """

    return actmat

def set_popul_mask(ref_actmat):

    return popul_mask


def run_evol(scorer, objfunc, optimizer, G, steps=100, label="obj-target-G", savedir="",
            RFresize=True, corner=(0, 0), imgsize=(224, 224)):
    new_codes = init_code
    # new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    generations = []
    codes_all = []
    best_imgs = []
    for i in range(steps,):
        codes_all.append(new_codes.copy())
        imgs = G.visualize_batch_np(new_codes) # B=1
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        if RFresize: imgs = resize_and_pad(imgs, corner, imgsize)
        actmat = scorer.score_tsr(imgs)
        scores = objfunc(actmat, targ_actmat)

        if "BigGAN" in G.__class__:
            print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                latent_code[:, :128].norm(dim=1).mean()))
        else:
            print("step %d score %.3f (%.3f) (norm %.2f )" % (
                i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
        new_codes = optimizer.step_simple(scores, new_codes, )
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))
        best_imgs.append(imgs[scores.argmax(),:,:,:])

    codes_all = np.concatenate(tuple(codes_all), axis=0)
    scores_all = np.array(scores_all)
    generations = np.array(generations)
    mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
    mtg_exp.save(join(savedir, "besteachgen%s_%05d.jpg" % (methodlab, RND,)))
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
    if args.G == "fc6":
        np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all, codes_fin=codes_all[-80:,:])
    else:
        np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all, codes_all=codes_all)
    visualize_trajectory(scores_all, generations, codes_arr=codes_all, title_str=methodlab).savefig(
        join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
    return codes_all, scores_all, generations

["FC6", "BigGAN"]



from GAN_utils import upconvGAN, loadBigGAN, BigGAN_wrapper
Optimizer = ["CholCMA", "HessCMA", "Adam"]
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)


scorer = ...
optimizer = ...
optimfun = lambda z: objfunc(render(z))

score_method = "cosine"
popul_idxs = select_popul_record()
refimgtsr = load_ref_imgs(preprocess=...)
ref_actmat = encode_image(refimgtsr)
popul_m, popul_s = set_normalizer(ref_actmat)
popul_mask = set_popul_mask(ref_actmat)
objfunc = set_objective(score_method, grad=False)
targ_actmat = encode_image(target_imgtsr)
run_evol(scorer, objfunc, optimizer, G, label="obj-target-G", savedir="",
            steps=100, RFresize=True, corner=(0, 0), imgsize=(224, 224))









def run(objfunc, init_code=None):
    # resize and pad 
    self.recording = []
    self.scores_all = []
    self.codes_all = []
    self.generations = []
    for self.istep in range(self.max_steps):
        if self.istep == 0:
            if init_code is None:
                codes = np.zeros([1, self.code_length])
            else:
                codes = init_code
        print('\n>>> step %d' % self.istep)
        t0 = time()
        self.current_images = self.render_tsr(codes)
        t1 = time()  # generate image from code
        self.current_images = resize_and_pad_tsr(self.current_images, self.imgsize, self.corner)
        synscores = objfunc(self.current_images)
        t2 = time()  # score images
        codes_new = self.optimizer.step_simple(synscores, codes)
        t3 = time()  # use results to update optimizer
        self.codes_all.append(codes)
        self.scores_all = self.scores_all + list(synscores)
        self.generations = self.generations + [self.istep] * len(synscores)
        codes = codes_new
        # summarize scores & delays
        print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
        print(('step %d time: total %.2fs | ' +
               'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
              % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
    self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
    self.scores_all = np.array(self.scores_all)
    self.generations = np.array(self.generations)