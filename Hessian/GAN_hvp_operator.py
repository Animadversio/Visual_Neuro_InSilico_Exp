import torch
import torch.nn.functional as F
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from sklearn.cross_decomposition import CCA
from time import time
import sys
#%% This operator could be used as a local distance metric on the GAN image manifold.
#   This version copied from hessian_eigenthings use backward autodifferencing
class GANHVPOperator(Operator):
    def __init__(
            self,
            model,
            code,
            criterion,
            use_gpu=True,
            preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True),
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
        self.preprocess = preprocess
        self.criterion = criterion
        self.code = code.clone().requires_grad_(False).float().to(device) # torch.float32
        self.size = self.code.numel()
        # self.perturb_vec = torch.zeros((1, 4096), dtype=torch.float32).requires_grad_(True).to(device)
        self.perturb_vec = 0.0001 * torch.randn((1, self.size), dtype=torch.float32).requires_grad_(True).to(
            device) # dimension debugged Sep 10
        self.activation = activation
        if activation:  # then criterion is a single entry objective function
            self.img_ref = self.model.visualize(self.code + self.perturb_vec)
            activ = self.criterion(self.preprocess(self.img_ref))
            gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=True, retain_graph=True)[0]
        else:
            self.img_ref = self.model.visualize(self.code, )  # forward the feature vector through the GAN
            img_pertb = self.model.visualize(self.code + self.perturb_vec)
            d_sim = self.criterion(self.preprocess(self.img_ref), self.preprocess(img_pertb))
            # similarity metric between 2 images.
            gradient = torch.autograd.grad(d_sim, self.perturb_vec, create_graph=True, retain_graph=True)[0]
            # 1st order gradient
        self.gradient = gradient.view(-1)

    def select_code(self, code):
        self.code = code.clone().requires_grad_(False).float().to(self.device) # torch.float32
        self.size = self.code.numel()
        self.perturb_vec = torch.zeros((1, self.size), dtype=torch.float32).requires_grad_(True).to(self.device)
        self.img_ref = self.model.visualize(self.code, )  # forward the feature vector through the GAN
        img_pertb = self.model.visualize(self.code + self.perturb_vec)
        d_sim = self.criterion(self.preprocess(self.img_ref), self.preprocess(img_pertb))
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
        hessian_vec_prod = grad_grad[0].view(-1)  # torch.cat([g.view(-1) for g in grad_grad]) #.contiguous()
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            self.gradient, self.perturb_vec, grad_outputs=vec, only_inputs=True, retain_graph=True
        )
        hessian_vec_prod = grad_grad[0].view(-1)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in [self.perturb_vec]:
            if p.grad is not None:
                p.grad.data.zero_()


class GANForwardHVPOperator(Operator):
    """This part amalgamates the structure of Lucent and hessian_eigenthings"""
    def __init__(
            self,
            model,
            code,
            objective,
            preprocess=lambda img: F.interpolate(img, (224, 224), mode='bilinear', align_corners=True),
            use_gpu=True,
            EPS=1E-2,
    ):
        device = "cuda" if use_gpu else "cpu"
        self.device = device
        if hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(objective, "parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.model = model
        self.objective = objective
        self.preprocess = preprocess
        self.code = code.clone().requires_grad_(False).float().to(device)  # torch.float32
        self.img_ref = self.model.visualize(self.code)
        resz_img = self.preprocess(self.img_ref)  # F.interpolate(self.img_ref, (224, 224), mode='bilinear', align_corners=True)
        activ = self.objective(resz_img)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.code.norm() * self.EPS

    def select_code(self, code):
        """Change the reference code"""
        self.code = code.clone().requires_grad_(False).float().to(self.device)  # torch.float32
        self.perturb_norm = self.code.norm() * self.EPS
        self.img_ref = self.model.visualize(self.code + self.perturb_vec)
        resz_img = self.preprocess(self.img_ref)
        activ = self.objective(resz_img)
        gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=False, retain_graph=False)[0]
        self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        EPS = self.EPS if EPS is None else EPS
        self.perturb_norm = self.code.norm() * EPS
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * torch.tensor([1, -1.0], device=self.device).view(-1, 1) * vec.detach()
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(perturb_vecs)
        resz_img = self.preprocess(img)
        activs = self.objective(resz_img)  # , scaler=True
        # obj = alexnet.features[:10](resz_img)[:, :, 6, 6].sum()  # esz_img.std()
        ftgrad_both = torch.autograd.grad(activs.sum(), perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[0, :] - ftgrad_both[1, :]) / (2 * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass


class GANForwardHVPOperator_multiscale(Operator):
    """This part amalgamates the structure of Lucent and hessian_eigenthings"""
    def __init__(
            self,
            model,
            code,
            objective,
            preprocess=lambda img: F.interpolate(img, (224, 224), mode='bilinear', align_corners=True),
            use_gpu=True,
            scalevect=(0.5, 1.0, 2.0),
            EPS=1E-2,
    ):
        device = "cuda" if use_gpu else "cpu"
        self.device = device
        if hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(objective, "parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.model = model
        self.objective = objective
        self.preprocess = preprocess
        self.code = code.clone().requires_grad_(False).float().to(device)  # torch.float32
        self.img_ref = self.model.visualize(self.code)
        resz_img = self.preprocess(self.img_ref)  # F.interpolate(self.img_ref, (224, 224), mode='bilinear', align_corners=True)
        activ = self.objective(resz_img)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.code.norm() * self.EPS
        self.ticks = torch.tensor(list(scalevect),
                                  device=self.device).reshape(-1, 1)
        self.ticks = torch.concat([self.ticks, -self.ticks], dim=0)
        self.ticks_divisor = torch.tensor(sum(scalevect), device=self.device)
        self.ticks_N = len(scalevect)

    # def select_code(self, code):
    #     """Change the reference code"""
    #     self.code = code.clone().requires_grad_(False).float().to(self.device)  # torch.float32
    #     self.perturb_norm = self.code.norm() * self.EPS
    #     self.img_ref = self.model.visualize(self.code + self.perturb_vec)
    #     resz_img = self.preprocess(self.img_ref)
    #     activ = self.objective(resz_img)
    #     gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=False, retain_graph=False)[0]
    #     self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        EPS = self.EPS if EPS is None else EPS
        self.perturb_norm = self.code.norm() * EPS
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * self.ticks * vec.detach()
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(perturb_vecs)
        resz_img = self.preprocess(img)
        activs = self.objective(resz_img)  # , scaler=True
        ftgrad_both = torch.autograd.grad(activs.sum(), perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[:self.ticks_N, :].sum(dim=0)
                            - ftgrad_both[-self.ticks_N:, :].sum(dim=0)) \
                           / (2 * self.ticks_divisor * eps)
        return hessian_vec_prod

    def apply_batch(self, vecs, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if vecs.ndim == 1:
            vecs = vecs.unsqueeze(0)
        if vecs.size(0) == self.size:
            vecs = vecs.T
        assert vecs.size(1) == self.size
        vecnorm = vecs.norm(dim=1)
        if vecnorm.mean() < 1E-8:
            return torch.zeros_like(vecs).cuda()
        EPS = self.EPS if EPS is None else EPS
        self.perturb_norm = self.code.norm() * EPS
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = eps.unsqueeze(0).unsqueeze(2) * \
                       torch.einsum("Ti,iBC->TBC",
                       self.ticks, vecs.detach().unsqueeze(0))
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(self.code.detach() + perturb_vecs.reshape(-1, self.size))
        resz_img = self.preprocess(img)
        activs = self.objective(resz_img)  # , scaler=True
        ftgrad_both = torch.autograd.grad(activs.sum(), perturb_vecs, retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[:self.ticks_N, :, :].sum(dim=0)
                            - ftgrad_both[-self.ticks_N:, :, :].sum(dim=0)) \
                           / (2 * self.ticks_divisor * eps.unsqueeze(1))
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass


class NNForwardHVPOperator(Operator):
    """This part amalgamates the structure of Lucent and hessian_eigenthings"""
    def __init__(
            self,
            objective,
            input,
            use_gpu=True,
            EPS=1E-2,
    ):
        device = "cuda" if use_gpu else "cpu"
        self.device = device
        if hasattr(objective, "parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.objective = objective
        self.code = input.detach().clone().float().to(device)  # torch.float32
        activ = self.objective(self.code)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.EPS  # * torch.randn(self.size).norm() *

    # def select_code(self, code):
    #     """Change the reference code"""
    #     self.code = code.clone().detach().float().to(self.device)  # torch.float32
    #     self.perturb_norm = self.code.norm() * self.EPS
    #     activ = self.objective(self.code)
    #     gradient = torch.autograd.grad(activ, self.perturb_vec, create_graph=False, retain_graph=False)[0]
    #     self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        if EPS is None:
            EPS = self.EPS
        self.perturb_norm = EPS  # * self.code.norm()
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * torch.tensor([1, -1.0], device=self.device).view(-1, 1) * vec.detach()
        perturb_vecs.requires_grad_(True)
        activs = self.objective(perturb_vecs)  # , scaler=True
        ftgrad_both = torch.autograd.grad(activs.sum(), perturb_vecs,
                      retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[0, :] - ftgrad_both[1, :]) / (2 * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass


class NNForwardHVPOperator_multiscale(Operator):
    """This part amalgamates the structure of Lucent and hessian_eigenthings"""
    def __init__(
            self,
            objective,
            input,
            use_gpu=True,
            EPS=1E-2,
            scalevect=(0.5, 1.0, 2.0)
    ):
        device = "cuda" if use_gpu else "cpu"
        self.device = device
        if hasattr(objective, "parameters"):
            for param in objective.parameters():
                param.requires_grad_(False)
        self.objective = objective
        self.code = input.detach().clone().float().to(device)  # torch.float32
        activ = self.objective(self.code)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.EPS  # * torch.randn(self.size).norm() *
        self.ticks = torch.tensor(list(scalevect),
                                  device=self.device).reshape(-1, 1)
        self.ticks = torch.concat([self.ticks, -self.ticks], dim=0)
        self.ticks_divisor = torch.tensor(sum(scalevect), device=self.device)
        self.ticks_N = len(scalevect)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        if EPS is None:
            EPS = self.EPS
        # self.perturb_norm = EPS  # * self.code.norm()
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * self.ticks * vec.detach()
        perturb_vecs.requires_grad_(True)
        activs = self.objective(perturb_vecs)  # , scaler=True
        ftgrad_both = torch.autograd.grad(activs.sum(), perturb_vecs,
                      retain_graph=False, create_graph=False, only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[:self.ticks_N, :].sum(dim=0)
                            - ftgrad_both[-self.ticks_N:, :].sum(dim=0)) \
                           / (2 * self.ticks_divisor * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass
#%%
class GANForwardMetricHVPOperator(Operator):
    """This part amalgamates the structure of Lucent and hessian_eigenthings
    It adapts GANForwardHVPOperator for binary metric function
    """
    def __init__(
            self,
            model,
            code,
            criterion,
            preprocess=lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True),
            use_gpu=True,
            EPS=1E-2,
    ):
        device = "cuda" if use_gpu else "cpu"
        self.device = device
        if hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(criterion, "parameters"):
            for param in criterion.parameters():
                param.requires_grad_(False)
        self.model = model
        self.criterion = criterion  # metric function use to determine the image distance
        self.preprocess = preprocess
        self.code = code.clone().requires_grad_(False).float().to(device)  # reference code
        self.img_ref = self.model.visualize(self.code)
        self.img_ref = self.preprocess(self.img_ref)  # F.interpolate(self.img_ref, (224, 224), mode='bilinear', align_corners=True)
        activ = self.criterion(self.img_ref, self.img_ref)
        self.size = self.code.numel()
        self.EPS = EPS
        self.perturb_norm = self.code.norm() * self.EPS  # norm

    def select_code(self, code):
        self.code = code.clone().requires_grad_(False).float().to(self.device)  # torch.float32
        self.perturb_norm = self.code.norm() * self.EPS
        self.img_ref = self.model.visualize(self.code)
        self.img_ref = self.preprocess(self.img_ref)
        # dsim = self.criterion(self.img_ref, self.img_ref)
        # gradient = torch.autograd.grad(dsim, self.perturb_vec, create_graph=False, retain_graph=False)[0]
        # self.gradient = gradient.view(-1)

    def apply(self, vec, EPS=None):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters.
        Here we implement the forward approximation of HVP.
         Hv|_x \approx (g(x + eps*v) - g(x - eps*v)) / (2*eps)
        """
        vecnorm = vec.norm()
        if vecnorm < 1E-8:
            return torch.zeros_like(vec).cuda()
        EPS = self.EPS if EPS is None else EPS
        self.perturb_norm = self.code.norm() * EPS
        eps = self.perturb_norm / vecnorm
        # take the second gradient by comparing 2 first order gradient.
        perturb_vecs = self.code.detach() + eps * torch.tensor([1, -1.0]).view(-1, 1).to(self.device) * vec.to(self.device).detach()
        perturb_vecs.requires_grad_(True)
        img = self.model.visualize(perturb_vecs)
        resz_img = self.preprocess(img)
        dsim = self.criterion(self.img_ref, resz_img)
        # size 2, 1, 1, 1. Distance from reference to 2 perturbed images. Do mean before grad
        ftgrad_both = torch.autograd.grad(dsim.mean(), perturb_vecs, retain_graph=False, create_graph=False,
                                          only_inputs=True)[0]
        hessian_vec_prod = (ftgrad_both[0, :] - ftgrad_both[1, :]) / (2 * eps)
        return hessian_vec_prod

    def vHv_form(self, vec):
        """
        Returns Bilinear form vec.T*H*vec where H is the hessian of the loss.
        If vec is eigen vector of H this will return the eigen value.
        """
        hessian_vec_prod = self.apply(vec)
        vhv = (hessian_vec_prod * vec).sum()
        return vhv

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        pass

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
    (This function just change the Operator from the original HVPOperator to GANHVPOperator.)

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
#%%
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
def get_full_hessian(loss, param):
    # from https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    # modified from hessian_eigenthings repo. api follows hessian.hessian
    hessian_size = param.numel()
    hessian = torch.zeros(hessian_size, hessian_size)
    loss_grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True, only_inputs=True)[0].view(-1)
    for idx in range(hessian_size):
        clear_output(wait = True)
        progress_bar(
            idx, hessian_size, "full hessian columns: %d of %d" % (idx, hessian_size)
        )
        grad2rd = torch.autograd.grad(loss_grad[idx], param, create_graph=False, retain_graph=True, only_inputs=True)
        hessian[idx] = grad2rd[0].view(-1)
    return hessian.cpu().data.numpy()
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
    from os.path import join
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
    t0 = time()
    cca = CCA(n_components=100)
    evec1_c, evec3_c = cca.fit_transform(eigenvecs.T, eigenvecs3.T)
    print(time() - t0)
    ccmat = np.corrcoef(evec1_c.T, evec3_c.T, )
    np.diag(ccmat[50:,:50])

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
    #%% Test the eigenvalues are close to that found by vHv bilinear form.
    t0 = time()
    feat = torch.randn((1, 4096), dtype=torch.float32).requires_grad_(False).cuda()
    eigenvals, eigenvecs = compute_hessian_eigenthings(G, feat, model_squ,
                       num_eigenthings=100, mode="lanczos", use_gpu=True, )
    print(time() - t0)  # 18.45
    #%%
    GHVP = GANHVPOperator(G, feat, model_squ, use_gpu=True)
    # GHVP.vHv_form(torch.tensor(eigenvecs[1, :]).cuda())
    # eigenvals
    t0 = time()
    vHv_vals = []
    eigenvecs_tsr = torch.tensor(eigenvecs).cuda()
    for i in range(eigenvecs_tsr.shape[0]):
        vHv_vals.append(GHVP.vHv_form(eigenvecs_tsr[i, :]).item())
    print(time()-t0)
    #%%
    savedir = r"E:\OneDrive - Washington University in St. Louis\Artiphysiology\HessianDecomp"
    plt.figure()
    plt.plot(eigenvals[::-1], alpha=0.5, lw=2, label="lanczos")
    # plt.plot(eigenvals2, alpha=0.5, lw=2, label="power_iter")
    plt.plot(vHv_vals[::-1], alpha=0.5, lw=2, label="vHv")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.title("Comparing Eigenvalue computed by Lanczos and vHv")
    plt.savefig(join(savedir, "Lanczos_vHv_cmp.png"))
    plt.show()
    #%% Analyze of isotropy of GAN space using Hessian spectrum
    feat = 64 * torch.tensor(PC1_vect).float() #torch.randn(4096).float().cuda()
    eval_col = []
    evect_col = []
    t0 = time()
    for vnorm in [0, 1, 2, 3, 4, 5]:
        evals, evecs = compute_hessian_eigenthings(G, vnorm * feat, model_squ,
                  num_eigenthings=800, mode="lanczos", use_gpu=True, )
        eval_col.append(evals)
        evect_col.append(evecs)
        print(
            "Norm %d \nEigen value: max %.3E min %.3E std %.3E" % (vnorm * 64, evals.max(), evals.min(), evals.std()))
    print(time() - t0)
    np.savez("H_norm_relation.npz", eval_col=eval_col, evect_col=evect_col, feat=feat)
    #%%
    plt.figure()
    for evals, vnorm in zip(eval_col, [0, 1, 2, 3, 4, 5]):
        plt.plot(evals[-50:]  * 1, label="norm%d"%(vnorm*64)) # / evals[-1]
    plt.legend()
    plt.xlabel("eigen id")
    plt.ylabel("eigenvals")
    plt.savefig(join(savedir, "code_norm_spectra_curv.png"))
    plt.show()