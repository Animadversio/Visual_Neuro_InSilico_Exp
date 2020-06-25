import torch
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from time import time
import sys
#%%
class GANHVPOperator(Operator):
    def __init__(
            self,
            model,
            code,
            criterion,
            use_gpu=True,
            activation=False,
    ):
        if use_gpu:
            device = "cuda"
            self.device = device
        if hasattr(model,"parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(criterion,"parameters"):
            for param in criterion.parameters():
                param.requires_grad_(False)
        self.model = model
        self.criterion = criterion
        self.code = code.clone().requires_grad_(False).float().to(device) # torch.float32
        # self.perturb_vec = torch.zeros((1, 4096), dtype=torch.float32).requires_grad_(True).to(device)
        self.perturb_vec = 0.0001 * torch.randn((1, 4096), dtype=torch.float32).requires_grad_(True).to(device)
        self.activation = activation
        if activation:
            self.img_ref = self.model.visualize(self.code + self.perturb_vec)
            activ = self.criterion(self.img_ref)
            gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=True, retain_graph=True)[0]
        else:
            self.img_ref = self.model.visualize(self.code, )  # forward the feature vector through the GAN
            img_pertb = self.model.visualize(self.code + self.perturb_vec)
            d_sim = self.criterion(self.img_ref, img_pertb)  # similarity metric between 2 images.
            gradient = torch.autograd.grad(d_sim, self.perturb_vec, create_graph=True, retain_graph=True)[0]
        self.gradient = gradient.view(-1)
        self.size = self.perturb_vec.numel()

    def select_code(self, code):
        self.code = code.clone().requires_grad_(False).float().to(self.device) # torch.float32
        self.perturb_vec = torch.zeros((1, 4096), dtype=torch.float32).requires_grad_(True).to(self.device)
        self.img_ref = self.model.visualize(self.code, )  # forward the feature vector through the GAN
        img_pertb = self.model.visualize(self.code + self.perturb_vec)
        d_sim = self.criterion(self.img_ref, img_pertb)
        gradient = torch.autograd.grad(d_sim, self.perturb_vec, create_graph=True, retain_graph=True)[0]
        self.gradient = gradient.view(-1)
        self.size = self.perturb_vec.numel()

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            self.gradient, self.perturb_vec, grad_outputs=vec, only_inputs=True, retain_graph=True
        )
        hessian_vec_prod = grad_grad[0].view(-1) #torch.cat([g.view(-1) for g in grad_grad]) #.contiguous()
        return hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in [self.perturb_vec]:
            if p.grad is not None:
                p.grad.data.zero_()


def compute_hessian_eigenthings(
    model,
    code,
    loss,
    num_eigenthings=40,
    mode="power_iter",
    use_gpu=True,
    **kwargs
):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_eigenthings : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
    mode : str ['power_iter', 'lanczos']
        which backend to use to compute the top eigenvalues.
    use_gpu:
        if true, attempt to use cuda for all lin alg computatoins
    max_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = GANHVPOperator(
        model,
        code,
        loss,
        use_gpu=use_gpu,
    )
    eigenvals, eigenvecs = None, None
    if mode == "power_iter":
        eigenvals, eigenvecs = deflated_power_iteration(
            hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
        )
    elif mode == "lanczos":
        eigenvals, eigenvecs = lanczos(
            hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
        )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return eigenvals, eigenvecs

#%% Test the module
if __name__=="__main__":
    sys.path.append(r"E:\Github_Projects\PerceptualSimilarity")
    import models  # from PerceptualSimilarity folder
    # model_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
    model_squ = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])
    from GAN_utils import upconvGAN
    G = upconvGAN("fc6")
    G.requires_grad_(False).cuda() # this notation is incorrect in older pytorch
    model_squ.requires_grad_(False).cuda()
    #%%
    feat = torch.randn((4096), dtype=torch.float32).requires_grad_(False).cuda()
    GHVP = GANHVPOperator(G, feat, model_squ)
    GHVP.apply(torch.randn((4096)).requires_grad_(False).cuda())

    #%% 300 vectors
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=300, mode="lanczos", use_gpu=True,)
    print(time() - t0,"\n")  # 81.02 s
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals3, eigenvecs3 = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=300, mode="lanczos", use_gpu=True, max_steps=50,)
    print(time() - t0, "\n")  # 82.15 s
    t0 = time()
    eigenvals2, eigenvecs2 = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=300, mode="power_iter", use_gpu=True,)
    print(time() - t0)   # 936.246 / 1002.95
    #%% 100 vectors
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=100, mode="lanczos", use_gpu=True,)
    print(time() - t0) # 79.466
    t0 = time()
    eigenvals2, eigenvecs2 = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=100, mode="power_iter", use_gpu=True,)
    print(time() - t0) # 227.1 s
    #%%
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=40, mode="lanczos", use_gpu=True,)
    print(time() - t0) # 13.09 sec
    t0 = time()
    eigenvals2, eigenvecs2 = compute_hessian_eigenthings(G, feat, model_vgg,
        num_eigenthings=40, mode="power_iter", use_gpu=True,)
    print(time() - t0) # 70.09 sec
    #%%
    import numpy as np
    import matplotlib.pylab as plt
    innerprod = eigenvecs @ eigenvecs2[::-1,:].T
    np.diag(innerprod)
    #%%
    innerprod = eigenvecs @ eigenvecs3[::-1,:].T
    np.diag(innerprod)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(eigenvals[::-1], alpha=0.5, lw=2, label="lanczos")
    # plt.plot(eigenvals2, alpha=0.5, lw=2, label="power_iter")
    plt.plot(eigenvals3[::-1], alpha=0.5, lw=2, label="lanczos")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.abs(np.diag(eigenvecs[::-1] @ eigenvecs3[::-1].T)))
    # plt.plot(np.abs(np.diag(eigenvecs[::-1] @ eigenvecs2.T)))
    plt.ylabel("Inner prod of eigenvector")
    plt.title("Compare Lanczos and Power iter method in computing eigen vectors")
    plt.show()
    #%%
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_squ,
        num_eigenthings=100, mode="lanczos", use_gpu=True,)
    print(time() - t0)  # 79.466
    t0 = time()
    eigenvals3, eigenvecs3 = compute_hessian_eigenthings(G, feat, model_squ,
        num_eigenthings=100, mode="lanczos", use_gpu=True, max_steps=50,)
    print(time() - t0)
    #%%
    t0 = time()
    eigenvals2, eigenvecs2 = compute_hessian_eigenthings(G, feat, model_squ,
              num_eigenthings=100, mode="power_iter", use_gpu=True, )
    print(time() - t0)
    #%%
    from sklearn.cross_decomposition import CCA
    t0 = time()
    cca = CCA(n_components=100)
    evec1_c, evec3_c = cca.fit_transform(eigenvecs.T, eigenvecs3.T)
    print(time() - t0)
    ccmat = np.corrcoef(evec1_c.T, evec3_c.T, )
    np.diag(ccmat[50:,:50])
    #%% Wrap up function
    def cca_correlation(X, Y, n_comp=50):
        """
        :param X, Y: should be N-by-p, N-by-q matrices,
        :param n_comp: a integer, how many components we want to create and compare.
        :return: cca_corr, n_comp-by-n_comp matrix
           X_c, Y_c will be the linear mapped version of X, Y with shape  N-by-n_comp, N-by-n_comp shape
           cc_mat is the
        """
        cca = CCA(n_components=n_comp)
        X_c, Y_c = cca.fit_transform(X, Y)
        ccmat = np.corrcoef(X_c, Y_c, rowvar=False)
        cca_corr = np.diag(ccmat[n_comp:, :n_comp])  # slice out the cross corr part
        return cca_corr
    #%%
    t0 = time()
    n_comp = 100
    cca_corr = cca_correlation(eigenvecs.T, eigenvecs3.T, n_comp=n_comp)
    cca_corr_baseline = cca_correlation(eigenvecs.T, np.random.randn(*eigenvecs.T.shape), n_comp=n_comp)
    print("%d components CCA corr %.2f, (baseline %.2f) (%.2fsec)" % (n_comp, cca_corr.mean(), cca_corr_baseline.mean(), time() - t0))
    # 50 components CCA corr 1.00, (baseline 0.20) (15.43sec)
    # 100 components CCA corr 1.00, (baseline 0.13) (22.50sec)
    # %%
    t0 = time()
    n_comp = 50
    cca_corr = cca_correlation(eigenvecs.T, eigenvecs2.T, n_comp=n_comp)
    cca_corr_baseline = cca_correlation(eigenvecs2.T, np.random.randn(*eigenvecs.T.shape), n_comp=n_comp)
    print("%d components CCA corr %.2f, (baseline %.2f) (%.2fsec)" % (
    n_comp, cca_corr.mean(), cca_corr_baseline.mean(), time() - t0))
    # 100 components CCA corr 0.99, (baseline 0.13) (18.22sec)
    # 50 components CCA corr 0.98, (baseline 0.20) (10.19sec)