from Cosine.cosine_evol_lib import *
from tqdm import tqdm
from glob import glob
from easydict import EasyDict
import scipy.optimize as opt
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square


def twoD_Gaussian(XYstack, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    # From https://stackoverflow.com/a/21566831
    xo = float(xo)
    yo = float(yo)
    x = XYstack[0]
    y = XYstack[1]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def fit_2dgauss(gradAmpmap_, pop_str, outdir, plot=True):
    if isinstance(gradAmpmap_, torch.Tensor):
        gradAmpmap_ = gradAmpmap_.numpy()

    H, W = gradAmpmap_.shape
    YY, XX = np.meshgrid(np.arange(H), np.arange(W))
    Xcenter = (gradAmpmap_ * XX).sum() / gradAmpmap_.sum()
    Ycenter = (gradAmpmap_ * YY).sum() / gradAmpmap_.sum()
    XXVar = (gradAmpmap_ * (XX - Xcenter) ** 2).sum() / gradAmpmap_.sum()
    YYVar = (gradAmpmap_ * (YY - Ycenter) ** 2).sum() / gradAmpmap_.sum()
    XYCov = (gradAmpmap_ * (XX - Xcenter) * (YY - Ycenter)).sum() / gradAmpmap_.sum()
    print(f"Gaussian Fitting center ({Xcenter:.1f}, {Ycenter:.1f})\n"
          f" Cov mat XX {XXVar:.1f} YY {YYVar:.1f} XY {XYCov:.1f}")
    #% covariance

    # MLE estimate? not good... Not going to use
    # covmat = torch.tensor([[XXVar, XYCov], [XYCov, YYVar]]).double()
    # precmat = torch.linalg.inv(covmat)
    # normdensity = torch.exp(-((XX - Xcenter)**2*precmat[0, 0] +
    #                           (YY-Ycenter)**2*precmat[1, 1] +
    #                           2*(XX - Xcenter)*(YY - Ycenter)*precmat[0, 1]))
    # var = multivariate_normal(mean=torch.tensor([Xcenter, Ycenter]), cov=covmat)
    # xystack = np.dstack((xplot, yplot))
    # densitymap = var.pdf(xystack)

    # curve fitting , pretty good.
    xplot, yplot = np.mgrid[0:227:1, 0:227:1]
    initial_guess = (gradAmpmap_.max().item(),
                     Xcenter.item(), Ycenter.item(),
                     np.sqrt(XXVar).item()/4, np.sqrt(YYVar).item()/4,
                     0, 0)  # 5, 5, 0, 0)
    popt, pcov = opt.curve_fit(twoD_Gaussian, np.stack((xplot, yplot)).reshape(2, -1),
                               gradAmpmap_.reshape(-1), p0=initial_guess,
                               maxfev=10000)
    ffitval = twoD_Gaussian(np.stack((xplot, yplot)).reshape(2, -1),
                            *popt).reshape(H, W)
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
    fitdict = EasyDict(popt=popt, amplitude=amplitude, xo=xo, yo=yo,
            sigma_x=sigma_x, sigma_y=sigma_y, theta=theta, offset=offset,
            gradAmpmap=gradAmpmap_, fitmap=ffitval)
    np.savez(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit.npz"),
            **fitdict)
    if plot:
        plt.figure(figsize=[5.8, 5])
        plt.imshow(ffitval)
        plt.colorbar()
        plt.title(f"{pop_str}\n"
                  f"Ampl {amplitude:.1e} Cent ({xo:.1f}, {yo:.1f}) std: ({sigma_x:.1f}, {sigma_y:.1f})\n Theta: {theta:.2f}, Offset: {offset:.1e}", fontsize=14)
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit.png"))
        plt.show()

        figh, axs = plt.subplots(1,2,figsize=[9.8, 6])
        axs[0].imshow(gradAmpmap_)
        # plt.colorbar(axs[0])
        axs[1].imshow(ffitval)
        # plt.colorbar(axs[1])
        plt.suptitle(f"{pop_str}\n"
                  f"Ampl {amplitude:.1e} Cent ({xo:.1f}, {yo:.1f}) std: ({sigma_x:.1f}, {sigma_y:.1f})\n Theta: {theta:.2f}, Offset: {offset:.1e}", fontsize=14)
        plt.savefig(join(outdir, f"{pop_str}_gradAmpMap_GaussianFit_cmp.png"))
        plt.show()
    return fitdict


outdir = r"E:\insilico_exps\Cosine_insilico\summary\RF_estimate"
layers = ['.layer2.Bottleneck2',
          '.layer3.Bottleneck0',
          '.layer4.Bottleneck2']
popsize = 500
scorer = TorchScorer("resnet50")
for layer in layers:
    pop_str = f"resnet50-{layer}"
    unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [layer], popsize=popsize,
                                                                       seed=0)

    print("Computing RF by direct backprop (RFMapping)...")
    Cids, Hids, Wids = unit_tsridx_dict[layer]
    gradAmpmap = grad_RF_estimate(scorer.model, layer, (slice(None), Hids[0], Wids[0]),
                          input_size=(3, 227, 227), device="cuda", show=False, reps=200, batch=4)
    plt.figure(figsize=[5.8, 5])
    plt.imshow(gradAmpmap)
    plt.colorbar()
    plt.title(f"{pop_str}", fontsize=14)
    plt.savefig(join(outdir, f"{pop_str}_gradAmpMap.png"))
    plt.show()
    fitdict = fit_2dgauss(gradAmpmap, pop_str, outdir)


#%%
rfdict_all_raw = {}
rfdict_all = {}
for layer in layers:
    fitdict = np.load(join(outdir, f"resnet50-{layer}_gradAmpMap_GaussianFit.npz"),)
    alphamsk_raw = np.clip(fitdict["gradAmpmap"] / (0.606 * fitdict["gradAmpmap"].max()), 0, 1)
    alphamsk = np.clip(fitdict["fitmap"] / (0.606 * fitdict["fitmap"].max()), 0, 1)
    rfdict_all_raw[layer] = alphamsk_raw
    rfdict_all[layer] = alphamsk

#%%
def shorten_layer(layer):
    return layer.replace(".layer", "L").replace("Bottleneck", "Btn")


def apply_mask2montage(mtg, alphamsk, imgsize=227, pad=2):
    new_mtg = np.zeros_like(mtg)
    nrow, ncol = (mtg.shape[0] - pad) // (imgsize + pad), (mtg.shape[1] - pad) // (imgsize + pad)
    for ri in range(nrow):
        for ci in range(ncol):
            img_crop = mtg[pad + (pad+imgsize)*ri:pad + imgsize + (pad+imgsize)*ri, \
                   pad + (pad+imgsize)*ci:pad + imgsize + (pad+imgsize)*ci, :]
            new_mtg[pad + (pad+imgsize)*ri:pad + imgsize + (pad+imgsize)*ri, \
                   pad + (pad+imgsize)*ci:pad + imgsize + (pad+imgsize)*ci, :] = \
                img_crop * alphamsk[:,:,np.newaxis]
    
    return new_mtg


mtgdir = r"E:\insilico_exps\Cosine_insilico\summary\proto_summary"
newmtgdir = r"E:\insilico_exps\Cosine_insilico\summary\proto_summary_w_rffitmsk"
mtgimglist = glob(join(mtgdir, "*.jpg"))
for mtgfp in tqdm(mtgimglist):
    mtg = plt.imread(mtgfp)
    for layer in layers:
        if shorten_layer(layer) in mtgfp:
            alphamsk = rfdict_all[layer]
            break

    mtg_w_msk = apply_mask2montage(mtg, alphamsk)
    fnm = os.path.split(mtgfp)[1]
    fnmain, ext = os.path.splitext(fnm)
    newmtgfp = join(newmtgdir, fnmain+"_w_rffit"+ext)
    plt.imsave(newmtgfp, mtg_w_msk)

