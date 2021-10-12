#%%
import sys
sys.path.append(r"D:\Github\BigGANsAreWatching")
import torch
from BigGAN.model.BigGAN import Generator
from BigGAN.gan_load import make_big_gan, make_biggan_config, UnconditionalBigGAN
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
# from utils.utils import to_image
def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

cfg = make_biggan_config(resolution=128)
G = Generator(**cfg)
weight_path = r"D:\Github\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth"
G.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
G_u = UnconditionalBigGAN(G)
for param in G_u.parameters():
    param.requires_grad_(False)
G_u.cuda()
#%% Adapted from BigGAN Evolution's render function
def BigBiGAN_render(codes, scale=1.0, batch=20):
    sample_n = codes.shape[0]
    imgs_all = []
    csr = 0
    csr_end = 0
    while csr_end < sample_n:
        csr_end = min(csr + batch, sample_n)
        with torch.no_grad():
            imgs = G_u(torch.from_numpy(codes[csr:csr_end, :]).float().cuda())
            imgs = torch.clamp(imgs + 1.0 / 2.0, 0.0, 1.0, ) * scale
            imgs = imgs.permute(2, 3, 1, 0).cpu().numpy()
        imgs_all.extend([imgs[:, :, :, imgi] for imgi in range(imgs.shape[3])])
        csr = csr_end
    return imgs_all
#%%
if __name__ == '__main__':
    G_u.cuda()
    with torch.no_grad():
        imgs = G_u(torch.randn(40, 120).cuda())
    imgs_np = to_image(make_grid(imgs,7))
    imgs_np.show()
    #%%

