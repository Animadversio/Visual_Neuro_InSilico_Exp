"""
Native torch version of fc6 GANs. Support deployment on any machine, since the weights are publicly hosted online.
The motivation is to get rid of dependencies on Caffe framework totally.
"""
#%%
# import torch
# torch.save(G, r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\fc6\fc6GAN.pt")
# G = torch.load(r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\fc6\fc6GAN.pt")
# This is no use...since you still need network definition to use this
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
from os.path import join
from sys import platform
load_urls = False
if platform == "linux":  # CHPC cluster
    # homedir = os.path.expanduser('~')
    # netsdir = os.path.join(homedir, 'Generate_DB/nets')
    homedir = "/scratch/binxu"
    netsdir = "/scratch/binxu/torch/checkpoints"
    load_urls = True # note it will try to load from $TORCH_HOME\checkpoints\"upconvGAN_%s.pt"%"fc6"
    # ckpt_path = {"vgg16": "/scratch/binxu/torch/vgg16-397923af.pth"}
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        homedir = "D:/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
        homedir = r"C:\Users\ponce\Documents\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
        homedir = r"C:\Users\Ponce lab\Documents\Python\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
        homedir = r"C:\Users\Poncelab-ML2a\Documents\Python\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        homedir = "E:/Monkey_Data/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
        homedir = "C:/Users/zhanq/OneDrive - Washington University in St. Louis/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    else:
        load_urls = True
        homedir = os.path.expanduser('~')
        netsdir = os.path.join(homedir, 'Documents/nets')

model_urls = {"pool5" : "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA",
    "fc6": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4",
    "fc7": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw",
    "fc8": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU"}

def load_statedict_from_online(name="fc6"):
    torchhome = torch.hub._get_torch_home()
    ckpthome = join(torchhome, "checkpoints")
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, "upconvGAN_%s.pt"%name)
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None,
                                   progress=True)
    SD = torch.load(filepath)
    return SD

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

RGB_mean = torch.tensor([123.0, 117.0, 104.0])
RGB_mean = torch.reshape(RGB_mean, (1, 3, 1, 1))

class upconvGAN(nn.Module):
    def __init__(self, name="fc6", pretrained=True, shuffled=True):
        super(upconvGAN, self).__init__()
        self.name = name
        if name == "fc6" or name == "fc7":
            self.G = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ]))
            self.codelen = self.G[0].in_features
        elif name == "fc8":
            self.G = nn.Sequential(OrderedDict([
  ("defc7", nn.Linear(in_features=1000, out_features=4096, bias=True)),
  ("relu_defc7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc6", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc5", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("reshape", View((-1, 256, 4, 4))),
  ("deconv5", nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv5_1", nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv5_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv4", nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv4_1", nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv4_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv3", nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv3_1", nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv1", nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv0", nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ]))
            self.codelen = self.G[0].in_features
        elif name == "pool5":
            self.G = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ]))
            self.codelen = self.G[0].in_channels
        # load pre-trained weight from online or local folders
        if pretrained:
            if load_urls:
                SDnew = load_statedict_from_online(name)
            else:
                savepath = {"fc6": join(netsdir, r"upconv/fc6/generator_state_dict.pt"),
                            "fc7": join(netsdir, r"upconv/fc7/generator_state_dict.pt"),
                            "fc8": join(netsdir, r"upconv/fc8/generator_state_dict.pt"),
                            "pool5": join(netsdir, r"upconv/pool5/generator_state_dict.pt")}
                SD = torch.load(savepath[name])
                SDnew = OrderedDict()
                for name, W in SD.items():  # discard this inconsistency
                    name = name.replace(".1.", ".")
                    SDnew[name] = W
            self.G.load_state_dict(SDnew)
        # if shuffled:

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)
        return torch.clamp(raw[:, [2, 1, 0], :, :] + RGB_mean.to(raw.device), 0, 255.0) / 255.0 * scale

    def render(self, x, scale=1.0, B=42):  # add batch processing to avoid memory over flow for batch too large
        coden = x.shape[0]
        img_all = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < coden:
            csr_end = min(csr + B, coden)
            with torch.no_grad():
                imgs = self.visualize(torch.from_numpy(x[csr:csr_end, :]).float().cuda(), scale).permute(2,3,1,0).cpu().numpy()
            img_all.extend([imgs[:, :, :, imgi] for imgi in range(imgs.shape[3])])
            csr = csr_end
        return img_all

    def visualize_batch_np(self, codes_all_arr, scale=1.0, B=42):
        coden = codes_all_arr.shape[0]
        img_all = None
        csr = 0  # if really want efficiency, we should use minibatch processing.
        with torch.no_grad():
            while csr < coden:
                csr_end = min(csr + B, coden)
                imgs = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(), scale).cpu()
                img_all = imgs if img_all is None else torch.cat((img_all, imgs), dim=0)
                csr = csr_end
        return img_all
#%% Very useful function to visualize output
import numpy as np
from PIL import Image
from build_montages import build_montages, color_framed_montages
from IPython.display import clear_output
from hessian_eigenthings.utils import progress_bar
def visualize_np(G, code, layout=None, show=True):
    """Utility function to visualize a np code vectors.

    If it's a single vector it will show in a plt window, Or it will show a montage in a windows photo.
    G: a generator equipped with a visualize method to turn torch code into torch images.
    layout: controls the layout of the montage. (5,6) create 5 by 6 grid
    show: if False, it will return the images in 4d array only.
    """
    with torch.no_grad():
        imgs = G.visualize(torch.from_numpy(code).float().cuda()).cpu().permute([2, 3, 1, 0]).squeeze().numpy()
    if show:
        if len(imgs.shape) <4:
            plt.imshow(imgs)
            plt.show()
        else:
            img_list = [imgs[:,:,:,imgi].squeeze() for imgi in range(imgs.shape[3])]
            if layout is None:
                mtg = build_montages(img_list,(256,256),(imgs.shape[3],1))[0]
                Image.fromarray(np.uint8(mtg*255.0)).show()
            else:
                mtg = build_montages(img_list, (256, 256), layout)[0]
                Image.fromarray(np.uint8(mtg*255.0)).show()
    return imgs

#%% BigGAN wrapper for ease of usage
def loadBigGAN(version="biggan-deep-256"):
    from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
    if platform == "linux":
        cache_path = "/scratch/binxu/torch/"
        cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
        BGAN = BigGAN(cfg)
        BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    else:
        BGAN = BigGAN.from_pretrained(version)
    for param in BGAN.parameters():
        param.requires_grad_(False)
    # embed_mat = BGAN.embeddings.parameters().__next__().data
    BGAN.cuda()
    return BGAN

class BigGAN_wrapper():#nn.Module
    def __init__(self, BigGAN, space="class"):
        self.BigGAN = BigGAN
        self.space = space

    def visualize(self, code, scale=1.0, truncation=0.7):
        imgs = self.BigGAN.generator(code, truncation) # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation=0.7, B=15):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
                img_list = self.visualize(code_batch, truncation=truncation, ).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
                clear_output(wait=True)
                progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, truncation=0.7, B=15):
        img_tsr = self.visualize_batch_np(codes_all_arr, truncation=truncation, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]
#%%
import sys
if platform == "linux":
    BigBiGAN_root = r"/home/binxu/BigGANsAreWatching"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        BigBiGAN_root = r"D:\Github\BigGANsAreWatching"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        BigBiGAN_root = r"E:\Github_Projects\BigGANsAreWatching"
    else:
        BigBiGAN_root = r"D:\Github\BigGANsAreWatching"
sys.path.append(BigBiGAN_root)
# the model is on cuda from this.
def loadBigBiGAN(weightpath=None):
    from BigGAN.gan_load import UnconditionalBigGAN, make_big_gan
    # from BigGAN.model.BigGAN import Generator
    if weightpath is None:
        weightpath = join(BigBiGAN_root, "BigGAN\weights\BigBiGAN_x1.pth")
    BBGAN = make_big_gan(weightpath, resolution=128)
    # BBGAN = make_big_gan(r"E:\Github_Projects\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth", resolution=128)
    for param in BBGAN.parameters():
        param.requires_grad_(False)
    BBGAN.eval()
    return BBGAN
#%%
class BigBiGAN_wrapper():#nn.Module
    def __init__(self, BigBiGAN, ):
        self.BigGAN = BigBiGAN

    def visualize(self, code, scale=1.0, resolution=256):
        imgs = self.BigGAN(code, )
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def render(self, codes_all_arr, B=15, scale=1.0, resolution=256):
        img_tsr = None
        imgn = codes_all_arr.shape[0]
        csr = 0
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
                img_list = self.visualize(code_batch, scale=scale, resolution=resolution).cpu()
                img_tsr = img_list if img_tsr is None else torch.cat((img_tsr, img_list), dim=0)
                csr = csr_end
        return [img.permute([1,2,0]).numpy() for img in img_tsr]
#%% StyleGAN2 wrapper for ease of usage
import sys
if platform == "linux":  # CHPC cluster
    StyleGAN_root = r"/home/binxu/stylegan2-pytorch"
    ckpt_root = "/scratch/binxu/torch/StyleGANckpt"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        StyleGAN_root = r"D:\Github\stylegan2-pytorch"
        ckpt_root = join(StyleGAN_root, 'checkpoint')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        StyleGAN_root = r"E:\DL_Projects\Vision\stylegan2-pytorch"
        ckpt_root = join(StyleGAN_root, 'checkpoint')
    # elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
    #     homedir = r"C:\Users\ponce\Documents\Generator_DB_Windows"
    #     netsdir = os.path.join(homedir, 'nets')
    # elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
    #     homedir = r"C:\Users\Ponce lab\Documents\Python\Generator_DB_Windows"
    #     netsdir = os.path.join(homedir, 'nets')
    # elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
    #     homedir = r"C:\Users\Poncelab-ML2a\Documents\Python\Generator_DB_Windows"
    #     netsdir = os.path.join(homedir, 'nets')
    else:
        StyleGAN_root = r"E:\DL_Projects\Vision\stylegan2-pytorch"
        ckpt_root = join(StyleGAN_root, 'checkpoint')
sys.path.append(StyleGAN_root)

def loadStyleGAN2(ckpt_name="ffhq-512-avg-tpurun1.pt", channel_multiplier=2, n_mlp=8, latent=512, size=512,
                  device="cpu"):
    from model import Generator
    ckpt_path = join(ckpt_root, ckpt_name)
    g_ema = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    try:
        checkpoint = torch.load(ckpt_path)
    except:
        print("Checkpoint %s load failed, Available Checkpoints: "%ckpt_name, os.listdir(ckpt_path))
    g_ema.load_state_dict(checkpoint['g_ema'])
    g_ema.eval()
    for param in g_ema.parameters():
        param.requires_grad_(False)
    g_ema.cuda()
    return g_ema

class StyleGAN2_wrapper():#nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN
        truncation = 0.8  # Note these parameters could be tuned
        truncation_mean = 4096
        mean_latent = StyleGAN.mean_latent(truncation_mean)
        self.truncation = truncation
        self.mean_latent = mean_latent
        self.wspace = False

    def select_trunc(self, truncation, truncation_mean=4096):
        self.truncation = truncation
        mean_latent = self.StyleGAN.mean_latent(truncation_mean)
        self.mean_latent = mean_latent

    def select_space(self, wspace=False):
        self.wspace = wspace

    def visualize(self, code, scale=1.0, resolution=256, truncation=1, mean_latent=None, preset=True, wspace=False):
        if preset:
            imgs, _ = self.StyleGAN([code], truncation=self.truncation, truncation_latent=self.mean_latent, input_is_latent=self.wspace)
        else:
            if truncation is None:
                imgs, _ = self.StyleGAN([code], truncation=self.truncation, truncation_latent=self.mean_latent,
                                        input_is_latent=wspace)
            else:
                imgs, _ = self.StyleGAN([code], truncation=truncation, truncation_latent=mean_latent, input_is_latent=wspace)
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation, mean_latent, B=5):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                       truncation=truncation, mean_latent=mean_latent, preset=False).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
            clear_output(wait=True)
            progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, truncation=0.7, B=15):
        img_tsr = self.visualize_batch_np(codes_all_arr, truncation=self.truncation, mean_latent=self.mean_latent, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]

# G = BigGAN_wrapper(BGAN)
# # layer name translation
# # "defc7.weight", "defc7.bias", "defc6.weight", "defc6.bias", "defc5.weight", "defc5.bias".
# # "defc7.1.weight", "defc7.1.bias", "defc6.1.weight", "defc6.1.bias", "defc5.1.weight", "defc5.1.bias".
# SD = G.state_dict()
# SDnew = OrderedDict()
# for name, W in SD.items():
#     name = name.replace(".1.", ".")
#     SDnew[name] = W
# UCG.G.load_state_dict(SDnew)
#%% The first time to run this you need these modules
if __name__ == "__main__":
    import sys
    import matplotlib.pylab as plt
    sys.path.append(r"E:\Github_Projects\Visual_Neuro_InSilico_Exp")
    from torch_net_utils import load_generator, visualize
    G = load_generator(GAN="fc6")
    UCG = upconvGAN("fc6")
    #%%
    def test_consisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['deconv0'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = visualize(G, code.numpy(), mode="cpu")
        assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    test_consisitency(G, UCG)
    #%%
    G = load_generator(GAN="fc7")
    UCG = upconvGAN("fc7")
    test_consisitency(G, UCG)
    #%%
    # pool5 GAN
    def test_FCconsisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen, 6, 6))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['generated'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = G(code)['generated'][:, [2, 1, 0], :, :]
        imgorig = torch.clamp(imgorig + RGB_mean, 0, 255.0) / 255.0
        imgorig = imgorig.permute([2, 3, 1, 0]).squeeze()
        # imgorig = visualize(G, code.numpy(), mode="cpu")
        # assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    G = load_generator(GAN="pool5")
    UCG = upconvGAN("pool5")
    test_FCconsisitency(G, UCG)

#%% This can work~
# G = upconvGAN("pool5")
# G.G.load_state_dict(torch.hub.load_state_dict_from_url(r"https://drive.google.com/uc?export=download&id=1vB_tOoXL064v9D6AKwl0gTs1a7jo68y7",progress=True))