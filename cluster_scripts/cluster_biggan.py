
import torch
from os.path import join
from pytorch_pretrained_biggan.model import BigGAN, BigGANConfig
from pytorch_pretrained_biggan.utils import truncated_noise_sample, save_as_images, one_hot_from_names
import sys 
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

truncation = 0.4
# noise = truncated_noise_sample(batch_size=2, truncation=truncation)
# label = one_hot_from_names('diver', batch_size=2)
# noise = torch.tensor(noise, dtype=torch.float)
# label = torch.tensor(label, dtype=torch.float)
# with torch.no_grad():
#     outputs = BGAN(noise, label, truncation)
# print(outputs.shape)

with torch.no_grad():
    outputs = BGAN.forward(torch.randn(2,128), torch.rand(2,1000), 0.8)
print(outputs.shape)

class_id = 1
classvec = embed_mat[:, class_id:class_id+1].cuda().T
noisevec = torch.from_numpy(truncated_noise_sample(1, 128, 0.6)).cuda()
ref_vect = torch.cat((noisevec, classvec, ), dim=1).detach().clone()
mov_vect = ref_vect.detach().clone().requires_grad_(True)
imgs1 = G.visualize(ref_vect)
imgs2 = G.visualize(mov_vect)
dsim = ImDist(imgs1, imgs2)

%time H = get_full_hessian(dsim, mov_vect) # 122 sec for a 256d hessian


#%%





