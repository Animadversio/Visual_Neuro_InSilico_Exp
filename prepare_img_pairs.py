
import torch
from torch_net_utils import load_generator
from skimage.transform import resize
from imageio import imread
import matplotlib.pylab as plt

G = load_generator("fc6")
G.requires_grad_(False)
G.cuda()

def visualize(G, code, mode="cuda"):
    """Do the De-caffe transform (Validated)
    works for a single code """
    if mode == "cpu":
        blobs = G(code)
    else:
        blobs = G(code.cuda())
    out_img = blobs['deconv0']  # get raw output image from GAN
    if mode == "cpu":
        clamp_out_img = torch.clamp(out_img + BGR_mean, 0, 255)
    else:
        clamp_out_img = torch.clamp(out_img + BGR_mean.cuda(), 0, 255)
    vis_img = clamp_out_img[:, [2, 1, 0], :, :].permute([2, 3, 1, 0]).squeeze() / 255
    return vis_img

def L1loss(target, img):
    return (img - target).abs().sum(axis=2).mean()

def L2loss(target, img):
    return (img - target).pow(2).sum(axis=2).mean()

def img_backproj(target_img, lossfun=L1loss):
    tsr_target = target_img.astype(float)/255
    rsz_target = resize(tsr_target, (256, 256), anti_aliasing=True)
    tsr_target = torch.from_numpy(rsz_target).cuda()
    # assert size of this image is 256 256
    code = np.random.randn(4096)
    code = code.reshape(-1, 4096)
    feat = torch.from_numpy(code).float().requires_grad_(True)
    feat.cuda()
    optimizer = torch.optim.Adam([feat], lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for i in range(150):
        optimizer.zero_grad()
        img = visualize(G, feat)
        #loss = (img - tsr_target).abs().sum(axis=2).mean() # This loss could be better? 
        loss = lossfun(img, tsr_target)
        loss.backward()
        optimizer.step()
        # print("step%d" % i, loss)
    print("step%d" % i, loss)
    return feat.detach(), img.detach()

sys.path.append(r"D:\Github\PerceptualSimilarity")
import models  # from PerceptualSimilarity folder
model = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=1, gpu_ids=[0])

percept_vgg = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=1, gpu_ids=[0])
# d_sim = percept_loss.forward(resz_ref_img, resz_out_img)
zcode, fitimg = img_backproj(target_img, percept_vgg.forward)

zcode, fitimg = img_backproj(target_img)
plt.imshow(fitimg.cpu().numpy())
plt.show()


