import numpy as np
from hessian_eigenthings.lanczos import lanczos
from GAN_hvp_operator import GANForwardMetricHVPOperator, GANHVPOperator, get_full_hessian

def hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=None, preprocess=lambda img: img):
    """Higher level API for GAN hessian compute
    Parameters:
        G: GAN, usually wrapped up by a custom class. Equipped with a `visualize` function that takes a torch vector and
           output a torch image
        feat: a latent code as input to the GAN.
        ImDist: the image distance function. Support dsim = ImDist(img1, img2). takes in 2 torch images and output a
           scalar distance. Pass gradient.
       hessian_method: Currently, "BP" "ForwardIter" "BackwardIter" are supported
       preprocess: or post processing is the operation on the image generated by GAN. Default to be an identity map.
            `lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)` is a common choice.
        cutoff: For iterative methods, "ForwardIter" "BackwardIter" this specify how many eigenvectors it's going to
            compute.
    """
    if cutoff is None: cutoff = feat.numel() // 2 - 1
    if hessian_method == "BackwardIter":
        metricHVP = GANHVPOperator(G, feat, ImDist, preprocess=preprocess)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=cutoff, use_gpu=True)  # takes 113 sec on K20x cluster,
        eigvects = eigvects.T  # note the output shape from lanczos is different from that of linalg.eigh, row is eigvec
        H = eigvects @ np.diag(eigvals) @ eigvects.T
        # the spectrum has a close correspondance with the full Hessian. since they use the same graph.
    elif hessian_method == "ForwardIter":
        metricHVP = GANForwardMetricHVPOperator(G, feat, ImDist, preprocess=preprocess, EPS=EPS)  # 1E-3,)
        eigvals, eigvects = lanczos(metricHVP, num_eigenthings=cutoff, use_gpu=True, max_steps=200, tol=1e-6, )
        eigvects = eigvects.T
        H = eigvects @ np.diag(eigvals) @ eigvects.T
        # EPS=1E-2, max_steps=20 takes 84 sec on K20x cluster.
        # The hessian is not so close
    elif hessian_method == "BP":  # 240 sec on cluster
        ref_vect = feat.detach().clone().float().cuda()
        mov_vect = ref_vect.float().detach().clone().requires_grad_(True)
        imgs1 = G.visualize(ref_vect)
        imgs2 = G.visualize(mov_vect)
        dsim = ImDist(preprocess(imgs1), preprocess(imgs2))
        H = get_full_hessian(dsim, mov_vect)  # 122 sec for a 256d hessian, # 240 sec on cluster for 4096d hessian
        eigvals, eigvects = np.linalg.eigh(H)
    else:
        raise NotImplementedError
    return eigvals, eigvects, H

if __name__ == "__main__":
    import sys
    from time import time
    import torch
    from GAN_utils import loadBigBiGAN, loadStyleGAN, BigBiGAN_wrapper, StyleGAN_wrapper
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    sys.path.append(r"D:\Github\PerceptualSimilarity")
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    import models
    ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])

    BBGAN = loadBigBiGAN()
    G = BigBiGAN_wrapper(BBGAN)
    noisevect = torch.randn(1, 120)
    feat = 0.5 * noisevect.detach().clone().cuda()
    EPS = 1E-2
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("%.2f sec" % (time() - T0))  # 16.22 sec
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="ForwardIter")
    print("%.2f sec" % (time() - T0))  # 16.22 sec
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 16.22 sec
    #%%
    SGAN = loadStyleGAN("ffhq-512-avg-tpurun1.pt", size=512)
    G = StyleGAN_wrapper(SGAN)
    feat = 0.5 * torch.randn(1, 512).detach().clone().cuda()
    EPS = 1E-2
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="BackwardIter")
    print("%.2f sec" % (time() - T0))  # 16.22 sec
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="ForwardIter")
    print("%.2f sec" % (time() - T0))  # 16.22 sec
    T0 = time()
    hessian_compute(G, feat, ImDist, hessian_method="BP")
    print("%.2f sec" % (time() - T0))  # 16.22 sec