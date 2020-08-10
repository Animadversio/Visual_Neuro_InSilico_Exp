import torch
import numpy as np
import matplotlib.pylab as plt
from os.path import join
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, PchipInterpolator
from scipy.optimize import newton, root_scalar, minimize_scalar

def dist_step2(BGAN, ImDist, ticks, refvec, tanvec, refimg):
    step_latents = torch.tensor(ticks).float().cuda().view(-1, 1) @ tanvec + refvec
    with torch.no_grad():
        step_imgs = BGAN.generator(step_latents, 0.7)
        step_imgs = (step_imgs + 1.0) / 2.0
        dist_steps = ImDist(step_imgs, refimg).squeeze()
    return dist_steps.squeeze().cpu().numpy(), step_imgs

def find_level_step(BGAN, ImDist, targ_val, reftsr, tan_vec, refimg, iter=2, pos=True, maxdist=20):
    xval = [0]
    yval = [0]
    sign = 1 if pos else -1
    ntarget = len(targ_val)
    xnext = sign * np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3])
    for step in range(1 + iter):
        xcur = xnext
        ycur, imgs = dist_step2(BGAN, ImDist, xcur, reftsr, tan_vec, refimg)
        fit_val = ycur[-ntarget:] # target values are suffixed in the computation
        xval.extend(list(xcur))
        yval.extend(list(ycur))
        uniq_x, uniq_idx = np.unique(xval, return_index=True)  # sort and unique x data
        uniq_y = np.array(yval)[uniq_idx]
        interp_fn = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
        # interp_fn = InterpolatedUnivariateSpline(uniq_x, uniq_y,  k=3, ext=0)#bbox=bbox,
        # idx = np.argsort(xval)
        # interp_fn = interp1d(np.array(xval)[idx], np.array(yval)[idx], 'quadratic')
        xnext = []
        sol = []
        converge_flag = True
        for fval in targ_val:
            lowidx = np.where((uniq_y < fval))[0]  # * (uniq_x >= 0 if pos else uniq_x <= 0)
            highidx = np.where((uniq_y > fval))[0]  # * (uniq_x >= 0 if pos else uniq_x <= 0)
            # lowidx should NEVER be empty. 0 should bound it
            lowrelidx = np.abs(uniq_y[lowidx] - fval).argmin()
            lowbnd_x = uniq_x[lowidx[lowrelidx]]  # this should be closer to 0.
            if len(highidx) == 0:  # no point reach such distance, have to choose points with lower distances
                # lowbnd_x2 = uniq_x[lowidx[lowrelidx] - 1] if pos else uniq_x[lowidx[lowrelidx] + 1]
                unbound = True
            else:  # some point reaches higher distance, so you can bound your search in a bracket.
                highrelidx = np.abs(uniq_y[highidx] - fval).argmin()
                highbnd_x = uniq_x[highidx[highrelidx]]
                unbound = False
            try:
                if unbound:
                    interp_fn2 = lambda x: np.abs(interp_fn(x) - fval)
                    result = minimize_scalar(interp_fn2, bounds=[lowbnd_x, maxdist] if pos else [-maxdist, lowbnd_x],
                                             method='bounded', )
                    xhat = result.x
                    converge_flag = converge_flag and result.success
                    # If un bound add exploration points to the set for better fit
                    if pos:
                        basis = min(maxdist, max(uniq_x))
                        xnext.extend([basis+1, basis+2, basis+4])
                    else:
                        basis = max(-maxdist, min(uniq_x))
                        xnext.extend([basis-1, basis-2, basis-4])
                    # xhat = root_scalar(interp_fn2, x0=lowbnd_x, x1=lowbnd_x2, )
                else:
                    interp_fn2 = lambda x: interp_fn(x) - fval
                    result = root_scalar(interp_fn2, x0=lowbnd_x, x1=highbnd_x, bracket=(lowbnd_x, highbnd_x))
                    xhat = result.root
                    converge_flag = converge_flag and result.converged
                sol.append(xhat)
            except RuntimeError as e:
                print(e.args)
                xfiller = min(maxdist, max(xval)) + 1 if pos else max(-maxdist, min(xval)) - 1
                sol.append(xfiller)
        xnext = list(np.unique(xnext))
        xnext.extend(sol) # solutions are suffixed in the next batch of evaluations
        if step > 0:
            if np.max(np.abs(targ_val - fit_val)) < 1E-5:
                break
        if step == iter and not converge_flag:
            print("Line Seaarch doesn't converge even at last generation %d."%step)
    ycur, imgs = dist_step2(BGAN, ImDist, sol, reftsr, tan_vec, refimg)
    print(np.abs(targ_val - ycur))
    return sol, ycur, imgs

if __name__ == "__main__":
    import os
    newimg_dir = r"N:\Hess_imgs_large_dif"
    summary_dir = r"N:\Hess_imgs_large_dif\summary"
    os.makedirs(newimg_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    from time import time
    from torchvision.transforms import ToPILImage
    from torchvision.utils import make_grid
    from imageio import imwrite
    from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
    import sys
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    import models  # from PerceptualSimilarity folder
    ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
    for param in ImDist.parameters():
        param.requires_grad_(False)

    from GAN_utils import BigGAN_wrapper
    BGAN = BigGAN.from_pretrained("biggan-deep-256")
    BGAN.cuda()
    BGAN.eval()
    for param in BGAN.parameters():
        param.requires_grad_(False)
    EmbedMat = BGAN.embeddings.weight
    G = BigGAN_wrapper(BGAN)

    data = np.load("N:\Hess_imgs\summary\Hess_mat.npz")
    refvec = data["vect"]
    evc_clas = data['eigvects_clas']
    evc_clas_tsr = torch.from_numpy(data['eigvects_clas'][:, ::-1].copy()).float().cuda()
    eva_clas = data['eigvals_clas'][::-1]
    evc_nois = data['eigvects_nois']
    evc_nois_tsr = torch.from_numpy(data['eigvects_nois'][:, ::-1].copy()).float().cuda()
    eva_nois = data['eigvals_nois'][::-1]
    reftsr = torch.tensor(refvec).float().cuda()
    refimg = G.visualize(reftsr)
    ToPILImage()(refimg[0, :].cpu()).show()
    #%%
    # targ_val = np.array([0.12, 0.24, 0.36, 0.48, 0.6])
    targ_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    space = "noise"
    imgall = None
    xtick_col = []
    dsim_col = []
    vecs_col = []
    img_names = []
    img_labels = list(-targ_val[::-1]) + [0] + list(targ_val)  # -0.5, -0.4 ...  0.4, 0.5
    t0 = time()
    for eigid in [0,1,2,3,4,5,6,7,8,10,20,30,40, 50, 60, 70, 80]:#range(128):  # #
        if space == "class":
            tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid + 1].T), dim=1)
        elif space == "noise":
            tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid + 1].T, torch.zeros(1, 128).cuda()), dim=1)
        xtar_pos, ytar_pos, stepimgs_pos = find_level_step(BGAN, ImDist, targ_val, reftsr, tan_vec, refimg, iter=20,
                                                           pos=True)
        xtar_neg, ytar_neg, stepimgs_neg = find_level_step(BGAN, ImDist, targ_val, reftsr, tan_vec, refimg, iter=20,
                                                           pos=False)
        imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
        xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
        dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
        vecs_row = torch.tensor(xticks_row).cuda().view(-1, 1) @ tan_vec + reftsr

        xtick_col.append(xticks_row)
        dsim_col.append(dsim_row)
        vecs_col.append(vecs_row.cpu().numpy())
        img_names.extend("noise_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in img_labels)  # dsim_row)
        imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))
        print(time() - t0)

    mtg1 = ToPILImage()(make_grid(imgall, nrow=11).cpu())  # 20sec for 13 rows not bad
    mtg1.show()
    mtg1.save(join(summary_dir, "noise_space_all_var.jpg"))
    npimgs = imgall.permute([2, 3, 1, 0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]),
                                                  np.uint8(npimgs[:, :, :, imgi] * 255))
    # %%
    xtick_arr = np.array(xtick_col)
    dsim_arr = np.array(dsim_col)
    vecs_arr = np.array(vecs_col)
    np.savez(join(summary_dir, "noise_ImDist_root_data.npz"), xtick_arr=xtick_arr, dsim_arr=dsim_arr, vecs_arr=vecs_arr,
             targ_val=targ_val)
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(xtick_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("L2 deviation from center")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Distance Travel Along Given Eigen vector to achieve certain Image Distance")
    plt.savefig(join(summary_dir, "noise_code_deviation.jpg"))
    plt.show()
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(dsim_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Image Distance")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Achieved Image Distance Along Each Axis")
    plt.savefig(join(summary_dir, "noise_space_dist_curv.jpg"))
    plt.show()
    # %
    plt.figure()
    plt.matshow(dsim_arr, fignum=0)
    plt.colorbar()
    plt.savefig(join(summary_dir, "noise_space_distmat.jpg"))
    plt.show()
    #%%
    space = "class"
    imgall = None
    xtick_col = []
    dsim_col = []
    vecs_col = []
    img_names = []
    img_labels = list(-targ_val[::-1]) + [0] + list(targ_val)
    t0 = time()
    for eigid in [0, 1, 2, 3, 6, 9, 11, 13, 15, 17, 19, 21, 25, 40, 50, 60, 70, 80]:  # [0,1,2,3,4,5,6,7,8,10,20,30,
        # 40]:#
        if space == "class":
            tan_vec = torch.cat((torch.zeros(1, 128).cuda(), evc_clas_tsr[:, eigid:eigid + 1].T), dim=1)
        elif space == "noise":
            tan_vec = torch.cat((evc_nois_tsr[:, eigid:eigid + 1].T, torch.zeros(1, 128).cuda()), dim=1)
        xtar_pos, ytar_pos, stepimgs_pos = find_level_step(BGAN, ImDist, targ_val, reftsr, tan_vec, refimg, iter=20,
                                                           pos=True)
        xtar_neg, ytar_neg, stepimgs_neg = find_level_step(BGAN, ImDist, targ_val, reftsr, tan_vec, refimg, iter=20,
                                                           pos=False)
        imgrow = torch.cat((torch.flip(stepimgs_neg, (0,)), refimg, stepimgs_pos)).cpu()
        xticks_row = xtar_neg[::-1] + [0.0] + xtar_pos
        dsim_row = list(ytar_neg[::-1]) + [0.0] + list(ytar_pos)
        vecs_row = torch.tensor(xticks_row).cuda().view(-1, 1) @ tan_vec + reftsr
        xtick_col.append(xticks_row)
        dsim_col.append(dsim_row)
        vecs_col.append(vecs_row.cpu().numpy())
        img_names.extend("class_eig%d_lin%.2f.jpg" % (eigid, dist) for dist in img_labels) #np.linspace(-0.4, 0.4,11))
        #
        imgall = imgrow if imgall is None else torch.cat((imgall, imgrow))
        print(time() - t0)

    mtg2 = ToPILImage()(make_grid(imgall, nrow=11).cpu())  # 20sec for 13 rows not bad
    mtg2.show()
    mtg2.save(join(summary_dir, "class_space_all_var.jpg"))
    npimgs = imgall.permute([2, 3, 1, 0]).numpy()
    for imgi in range(npimgs.shape[-1]):  imwrite(join(newimg_dir, img_names[imgi]),
                                                  np.uint8(npimgs[:, :, :, imgi] * 255))
    # %%
    xtick_arr = np.array(xtick_col)
    dsim_arr = np.array(dsim_col)
    vecs_arr = np.array(vecs_col)
    np.savez(join(summary_dir, "class_ImDist_root_data.npz"), xtick_arr=xtick_arr, dsim_arr=dsim_arr, vecs_arr=vecs_arr,
             targ_val=targ_val)
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(xtick_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("L2 deviation from center")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Distance Travel Along Given Eigen vector to achieve certain Image Distance")
    plt.savefig(join(summary_dir, "class_code_deviation.jpg"))
    plt.show()
    # %
    plt.figure(figsize=[10, 7])
    plt.plot(dsim_arr)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Image Distance")
    plt.legend(["Neg%.2f" % d for d in targ_val[::-1]] + ["orig"] + ["Pos%.2f" % d for d in targ_val])
    plt.title("Achieved Image Distance Along Each Axis")
    plt.savefig(join(summary_dir, "class_space_dist_curv.jpg"))
    plt.show()
    # %
    plt.figure()
    plt.matshow(dsim_arr, fignum=0)
    plt.colorbar()
    plt.savefig(join(summary_dir, "class_space_distmat.jpg"))
    plt.show()