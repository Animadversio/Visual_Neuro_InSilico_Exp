"""
Toy example to see if we can calculate the Hessian of a piecewise linear function.
When the piecewise function approximates a smooth function like Gaussian.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hessian_eigenthings.lanczos import lanczos
from Hessian.lanczos_generalized import lanczos_generalized
from Hessian.GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, \
    compute_hessian_eigenthings, NNForwardHVPOperator, NNForwardHVPOperator_multiscale
from time import time
#%%
def gaussian_fun(x):
    return torch.exp(-torch.einsum("ij,ik,jk->i", x-cent, x-cent, Hess))


def quad_fun(x):
    return -torch.einsum("ij,ik,jk->i", x-cent, x-cent, Hess)
#%%
net = nn.Sequential(
    nn.Linear(4, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
)
net.cuda()
#%%
Hess = torch.diag(torch.tensor([4.0, 1.0, 0.1, 0.1]).cuda())
cent = 0.2 * torch.randn(1, 4).cuda()
#%%
X = 2 * torch.rand(5000, 4).cuda() - 1
y = torch.exp(-torch.einsum("ij,ik,jk->i", X-cent, X-cent, Hess)).unsqueeze(1)
#%%
optim = torch.optim.Adam(net.parameters(), lr=5e-3)
for i in range(100):
    optim.zero_grad()
    ypred = net(X)
    loss = F.mse_loss(ypred, y)
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print(loss.item())
#%%
x = torch.randn(1, 4).cuda()
x.requires_grad_(True)
act = net(x)
grad = torch.autograd.grad(act.sum(), x,
           create_graph=True, retain_graph=True)[0]
#%%

#%%
H = []
for i in range(grad.shape[1]):
    gradgrad = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
    H.append(gradgrad)
    # break
#%%
xx, yy = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
#%%
X_grid = torch.stack([xx.flatten(), yy.flatten(),
      cent[0,2].cpu()*torch.ones(xx.numel()), cent[0,3].cpu()*torch.ones(xx.numel())], ).T
actgrid = net(X_grid.cuda()).reshape(100, 100).detach().cpu().numpy()
#%%
plt.imshow(actgrid, cmap="jet")
plt.show()
#%% Estimating Hessian using forward HVP and Lanczos iteration.
# NNForwardHVPOperator(net, cent)
# activHVP = NNForwardHVPOperator(net, cent, EPS=1E-1,)
activHVP = NNForwardHVPOperator_multiscale(net, cent, EPS=5E-1,
                                           scalevect=(4.0, 2.0, 1.0, 0.5))
# activHVP = NNForwardHVPOperator_multiscale(gaussian_fun, cent, EPS=1E-1,
#                                            scalevect=(4.0, 2.0, 1.0, 0.5))
# activHVP = NNForwardHVPOperator(gaussian_fun, cent, EPS=1E-2,)
# activHVP.apply(1*torch.randn(4096).requires_grad_(False).cuda())
t0 = time()
eigvals, eigvects = lanczos(activHVP, num_eigenthings=2, use_gpu=True)
print(time() - t0)  # 146sec for 2000 eigens
eigvals = eigvals[::-1]
eigvects = eigvects[::-1, :]
print(eigvals)
print(eigvects)
#%%
from torch.autograd.functional import hessian

hessian(gaussian_fun, cent, create_graph=False, strict=False, vectorize=True, )

#%%
randvec = torch.randn(10, 4).cuda()
randvec = randvec / torch.norm(randvec, dim=1, keepdim=True)
randvec.requires_grad_(False)
def smooth_fun(net, eps=1E0):
    def f(x):
        output = net(eps * randvec + x)
        return output.mean()
    return f
#%%
hessian(smooth_fun(net), cent, create_graph=False, strict=False, vectorize=True,)
        #outer_jacobian_strategy='reverse-mode')