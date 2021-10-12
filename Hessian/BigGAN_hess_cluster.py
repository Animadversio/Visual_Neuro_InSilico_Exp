from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from Hessian.lanczos_generalized import lanczos_generalized
from Hessian.GAN_hvp_operator import GANHVPOperator, GANForwardHVPOperator, compute_hessian_eigenthings, get_full_hessian
import sys
import numpy as np
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite
from skimage.io import imsave
from build_montages import build_montages, color_framed_montages
import torchvision.models as tv
from torchvision.utils import make_grid
from pytorch_pretrained_biggan.model import BigGAN, BigGANConfig
from pytorch_pretrained_biggan.utils import truncated_noise_sample, save_as_images, one_hot_from_names
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
def get_BigGAN(version="biggan-deep-256"):
    cache_path = "/scratch/binxu/torch/"
    cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
    BGAN = BigGAN(cfg)
    BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    return BGAN

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

def LExpMap(refvect, tangvect, ticks=11, lims=(-1,1)):
    refvect, tangvect = refvect.reshape(1, -1), tangvect.reshape(1, -1)
    steps = np.linspace(lims[0], lims[1], ticks)[:, np.newaxis]
    interp_vects = steps @ tangvect + refvect
    return interp_vects

if sys.platform == "linux":
    sys.path.append(r"/home/binxu/PerceptualSimilarity")
    BGAN = get_BigGAN()
else:
    BGAN = BigGAN.from_pretrained("biggan-deep-256")

for param in BGAN.parameters():
    param.requires_grad_(False)
embed_mat = BGAN.embeddings.parameters().__next__().data
BGAN.cuda()

class BigGAN_wrapper():#nn.Module
    def __init__(self, BigGAN, space="class"):
        self.BigGAN = BigGAN
        self.space = space

    def visualize(self, code, scale=1.0):
        imgs = self.BigGAN.generator(code, 0.6)
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

G = BigGAN_wrapper(BGAN)

import models
ImDist = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])

# truncation = 0.4
# noise = truncated_noise_sample(batch_size=2, truncation=truncation)
# label = one_hot_from_names('diver', batch_size=2)
# noise = torch.tensor(noise, dtype=torch.float)
# label = torch.tensor(label, dtype=torch.float)
# with torch.no_grad():
#     outputs = BGAN(noise, label, truncation)
# print(outputs.shape)

# with torch.no_grad():
#     outputs = BGAN.forward(torch.randn(2,128), torch.rand(2,1000), 0.8)
# print(outputs.shape)
savedir = r"/scratch/binxu/GAN_hessian"
class_id = 1
start_cls, end_cls = 0, 1000
if len(sys.argv) > 1:
    start_cls = int(sys.argv[1])
    end_cls = int(sys.argv[2])

for class_id in range(start_cls, end_cls):
    T00 = time()
    classvec = embed_mat[:, class_id:class_id+1].cuda().T
    noisevec = torch.from_numpy(truncated_noise_sample(1, 128, 0.6)).cuda()
    ref_vect = torch.cat((noisevec, classvec, ), dim=1).detach().clone()
    mov_vect = ref_vect.detach().clone().requires_grad_(True)
    imgs1 = G.visualize(ref_vect)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H = get_full_hessian(dsim, mov_vect) # 122 sec for a 256d hessian
    eigvals, eigvects = np.linalg.eigh(H)  # 75 ms
    #%%
    noisevec.requires_grad_(True)
    classvec.requires_grad_(False)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_nois = get_full_hessian(dsim, noisevec)  # 39.3 sec to compute a Hessian.
    eigvals_nois, eigvects_nois = np.linalg.eigh(H_nois)  # 75 ms

    noisevec.requires_grad_(False)
    classvec.requires_grad_(True)
    mov_vect = torch.cat((noisevec, classvec, ), dim=1)
    imgs2 = G.visualize(mov_vect)
    dsim = ImDist(imgs1, imgs2)
    H_clas = get_full_hessian(dsim, classvec)  # 77sec to compute a Hessian.
    eigvals_clas, eigvects_clas = np.linalg.eigh(H_clas)  # 75 ms
    classvec.requires_grad_(False)

    np.savez(join(savedir, "Hess_cls%d.npz" % class_id), H=H, H_nois=H_nois, H_clas=H_clas, eigvals=eigvals,
             eigvects=eigvects, eigvals_clas=eigvals_clas, eigvects_clas=eigvects_clas, eigvals_nois=eigvals_nois,
             eigvects_nois=eigvects_nois, vect=ref_vect.cpu().numpy(),
             noisevec=noisevec.cpu().numpy(), classvec=classvec.cpu().numpy())
    #%%
    plt.figure(figsize=[7,5])
    plt.subplot(1, 2, 1)
    plt.plot(eigvals)
    plt.ylabel("eigenvalue")
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(eigvals))
    plt.ylabel("eigenvalue (log)")
    plt.suptitle("Hessian Spectrum Full Space")
    plt.savefig(join(savedir, "Hessian_cls%d.jpg"%class_id))
    #%
    plt.figure(figsize=[7,5])
    plt.subplot(1, 2, 1)
    plt.plot(eigvals_nois, label="noise")
    plt.plot(eigvals_clas, label="class")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(eigvals_nois), label="noise")
    plt.plot(np.log10(eigvals_clas), label="class")
    plt.ylabel("eigenvalue (log)")
    plt.legend()
    plt.suptitle("Hessian Spectrum Full Space")
    plt.tight_layout(pad=0.6)
    plt.savefig(join(savedir, "Hessian_sep_cls%d.jpg"%class_id))
    print("Spent %.1f sec from start" % (time() - T00))

    img_all = None
    for eigi in range(50): #eigvects.shape[1]
        interp_codes = LExpMap(ref_vect.cpu().numpy(), eigvects[:, -eigi-1], 11, (-2.5, 2.5))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 256))

    imggrid = make_grid(img_all, nrow=11)
    PILimg = ToPILImage()(imggrid)#.show()
    PILimg.save(join(savedir, "eigvect_full_cls%d.jpg"%class_id))
    #% Interpolation in the class space
    img_all = None
    for eigi in range(50): # eigvects_clas.shape[1]
        interp_class = LExpMap(classvec.cpu().numpy(), eigvects_clas[:, -eigi-1], 11, (-2.5, 2.5))
        interp_codes = np.hstack((noisevec.cpu().numpy().repeat(11, axis=0), interp_class, ))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 128))

    imggrid = make_grid(img_all, nrow=11)
    PILimg2 = ToPILImage()(imggrid)#.show()
    PILimg2.save(join(savedir, "eigvect_clas_cls%d.jpg"%class_id))
    #% Interpolation in the noise space
    img_all = None
    for eigi in range(50):#eigvects_nois.shape[1]
        interp_noise = LExpMap(noisevec.cpu().numpy(), eigvects_nois[:, -eigi-1], 11, (-4.5, 4.5))
        interp_codes = np.hstack((interp_noise, classvec.cpu().numpy().repeat(11, axis=0), ))
        img_list = G.visualize(torch.from_numpy(interp_codes).float().cuda()).cpu()
        img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
        clear_output(wait=True)
        progress_bar(eigi, 50, "ploting row of page: %d of %d" % (eigi, 128))

    imggrid = make_grid(img_all, nrow=11)
    PILimg3 = ToPILImage()(imggrid)#.show()
    PILimg3.save(join(savedir, "eigvect_nois_cls%d.jpg"%class_id))
    print("Spent %.1f sec from start" % (time() - T00))

