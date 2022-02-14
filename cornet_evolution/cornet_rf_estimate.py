""" Scripts to estimate RF of CorNet Units by gradient backprop
average across hundreds of white noise images.
"""
import sys
import torch
import numpy as np
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from layer_hook_utils import featureFetcher_recurrent
from scipy.stats import multivariate_normal
import scipy.optimize as opt
sys.path.append("D:\Github\CORnet")
import cornet
def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  # remove DataParallel
    return model


outdir = r"F:\insilico_exps\CorNet-recurrent-evol\RF_estimate"
model = get_model(pretrained=True)
#%%
def grad_RF_estimate_CorNet(model, area, sublayer, input_size=(3, 227, 227),
                     findcenter=True, reps=100, batch=8, device="cuda", show=True):
    # findcenter = True
    # batch = 8
    # reps = 100
    # area, sublayer = "V2", "output"
    # input_size = (3, 227, 227)

    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    with torch.no_grad():
        model(torch.rand((batch, *input_size)).cuda() * 2 - 1)
    tsr = fetcher["target"][0]
    _, C, H, W = tsr.shape
    Tsteps = len(fetcher["target"])
    print(f"tensor shape {C},{H},{W}. recurrent steps {Tsteps}")
    h.remove()
    if findcenter:
        pos = (H // 2, W // 2)
    #
    h = fetcher.record(area, sublayer, "target", ingraph=True)
    cnt = torch.zeros(Tsteps)
    gradabsdata = torch.zeros(Tsteps, *input_size).cuda()
    for _ in tqdm(range(reps)):
        intsr = torch.rand((batch, *input_size)).cuda() * 2 - 1
        intsr.requires_grad_(True)
        fetcher.activations["target"] = []
        model(intsr)
        for time_step in range(Tsteps):
            act_tsr = fetcher['target'][time_step]
            act = act_tsr[:, :, pos[0], pos[1]].pow(2).mean()
            if not torch.isclose(act, torch.tensor(0.0)):
                act.backward(retain_graph=True)
                gradabsdata[time_step, :] += intsr.grad.abs().mean(dim=0) # average across batch
                cnt[time_step] += 1
            else:
                continue

    fetcher.remove_hook()
    gradAmpmap = gradabsdata.mean(dim=1).cpu() / cnt.view([-1, 1, 1])
    if show:
        for time_step in range(Tsteps):
            plt.figure(figsize=[5.5, 5])
            plt.imshow(gradAmpmap[time_step].cpu())
            plt.title(f"{area}-{sublayer}-Tstep{time_step:d}")
            plt.colorbar()
            plt.show()
    return gradAmpmap


# compute the grad Amp Map
gradAmpmap_V2 = grad_RF_estimate_CorNet(model, "V2", "output",
                    input_size=(3, 227, 227), reps=100, batch=8)
torch.save(gradAmpmap_V2, join(outdir, "V2_RF_gradAmpmap.pt"))
gradAmpmap_V4 = grad_RF_estimate_CorNet(model, "V4", "output",
                    input_size=(3, 227, 227), reps=100, batch=8)
torch.save(gradAmpmap_V4, join(outdir, "V4_RF_gradAmpmap.pt"))
gradAmpmap_IT = grad_RF_estimate_CorNet(model, "IT", "output",
                    input_size=(3, 227, 227), reps=200, batch=4)
torch.save(gradAmpmap_IT, join(outdir, "IT_RF_gradAmpmap.pt"))
#%% visualize
for area in ["V2", "V4", "IT"]:
    gradAmpmap = torch.load(join(outdir, f"{area}_RF_gradAmpmap.pt"))
    for time_step in range(gradAmpmap.size(0)):
        plt.figure(figsize=[5.8, 5])
        plt.imshow(gradAmpmap[time_step].cpu())
        plt.title(f"{area}-{'output'}-Tstep{time_step:d}", fontsize=14)
        plt.colorbar()
        plt.savefig(join(outdir, f"{area}-{'output'}-Tstep{time_step:d}_gradAmpMap.png"))
        plt.show()

#%% Fit covariance and map
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


for area in ["V2", "V4", "IT"]:
    gradAmpmap = torch.load(join(outdir, f"{area}_RF_gradAmpmap.pt"))
    for time_step in range(gradAmpmap.size(0)):
        gradAmpmap_ = gradAmpmap[time_step]
        # gradAmpmap_ = gradAmpmap_ - gradAmpmap_.min()
        YY, XX = torch.meshgrid(torch.arange(227), torch.arange(227))
        Xcenter = (gradAmpmap_ * XX.float()).sum() / gradAmpmap_.sum()
        Ycenter = (gradAmpmap_ * YY.float()).sum() / gradAmpmap_.sum()
        XXVar = (gradAmpmap_ * (XX - Xcenter) ** 2).sum() / gradAmpmap_.sum()
        YYVar = (gradAmpmap_ * (YY - Ycenter) ** 2).sum() / gradAmpmap_.sum()
        XYCov = (gradAmpmap_ * (XX - Xcenter) * (YY - Ycenter)).sum() / gradAmpmap_.sum()
        print(f"Gaussian Fitting center ({Xcenter:.1f}, {Ycenter:.1f})\n"
              f" Cov mat XX {XXVar:.1f} YY {YYVar:.1f} XY {XYCov:.1f}")
        #%% covariance
        xplot, yplot = np.mgrid[0:227:1, 0:227:1]

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
        initial_guess = (gradAmpmap_.max().item(),
                         Xcenter.item(), Ycenter.item(),
                         XXVar.sqrt().item(), YYVar.sqrt().item(),
                         0, 0)  # 5, 5, 0, 0)
        popt, pcov = opt.curve_fit(twoD_Gaussian, np.stack((xplot, yplot)).reshape(2, -1),
                                   gradAmpmap_.numpy().reshape(-1), p0=initial_guess,
                                   maxfev=10000)
        ffitval = twoD_Gaussian(np.stack((xplot, yplot)).reshape(2, -1),
                                *popt).reshape(227, 227)
        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
        np.savez(join(outdir, f"{area}-{'output'}-Tstep{time_step:d}_gradAmpMap_GaussianFit.npz"),
                popt=popt, amplitude=amplitude, xo=xo, yo=yo, 
                sigma_x=sigma_x, sigma_y=sigma_y, theta=theta, offset=offset,
                gradAmpmap=gradAmpmap_.numpy(), fitmap=ffitval)
        #%%
        plt.figure(figsize=[5.8, 5])
        plt.imshow(ffitval)
        plt.colorbar()
        plt.title(f"{area}-{'output'}-Tstep{time_step:d}\n"
                  f"Ampl {amplitude:.1e} Cent ({xo:.1f}, {yo:.1f}) std: ({sigma_x:.1f}, {sigma_y:.1f})\n Theta: {theta:.2f}, Offset: {offset:.1e}", fontsize=14)
        plt.savefig(join(outdir, f"{area}-{'output'}-Tstep{time_step:d}_gradAmpMap_GaussianFit.png"))
        plt.show()


#%% dev zone
#%%
var = multivariate_normal(mean=torch.tensor([Xcenter, Ycenter]), cov=covmat)
xplot, yplot = np.mgrid[0:227:1, 0:227:1]
xystack = np.dstack((xplot, yplot))
densitymap = var.pdf(xystack)
plt.figure(figsize=[5.8, 5])
plt.imshow(densitymap)
plt.title(f"{area}-{'output'}-Tstep{time_step:d}", fontsize=14)
plt.colorbar()
# plt.savefig(join(outdir, f"{area}-{'output'}-Tstep{time_step:d}_gradAmpMap_GaussianFit.png"))
plt.show()

#%%
# initial_guess = (gradAmpmap_.max().item(), Xcenter.item(), Ycenter.item(),
#                  XXVar.sqrt().item(), YYVar.sqrt().item(), 0, 0 )
initial_guess = (gradAmpmap_.max().item(),
                 Xcenter.item(), Ycenter.item(),
                 XXVar.sqrt().item(), YYVar.sqrt().item(),
                 0, 0 )#5, 5, 0, 0)
popt, pcov = opt.curve_fit(twoD_Gaussian, np.stack((xplot, yplot)).reshape(2,-1),
                           gradAmpmap_.numpy().reshape(-1), p0=initial_guess,
                           maxfev=10000)
ffitval = twoD_Gaussian(np.stack((xplot, yplot)).reshape(2, -1),
                        *popt).reshape(227,227)
plt.imshow(ffitval)
plt.show()
