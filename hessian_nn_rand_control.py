"""
This Code try to create a shuffled control for what we found for Hessian.
"""

from GAN_utils import upconvGAN
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
G = upconvGAN("fc6")
SD = G.state_dict()
#%% Shuffle the weight matrix of each layer of GAN
shuffled_SD = {}
for name, Weight in SD.items():
    idx = torch.randperm(Weight.numel())
    W_shuf = Weight.view(-1)[idx].view(Weight.shape)
    shuffled_SD[name] = W_shuf
    # print(name, Weight.shape, Weight.mean().item(), Weight.std().item())
#%%
G_sf = upconvGAN("fc6")
G_sf.load_state_dict(shuffled_SD)
#%%
img = G_sf.visualize(torch.randn(10, 4096))
ToPILImage()(make_grid(img[:,:])).show()
#%%
W = SD['G.defc7.weight']
idx = torch.randperm(W.numel())
W_shuf = W.view(-1)[idx].view(W.shape)
#%%
with torch.no_grad():
    U, S, V = torch.svd(W.cuda())
    U_sf, S_sf, V_sf = torch.svd(W_shuf.cuda())
#%%
from os.path import join
import matplotlib.pylab as plt
savedir = r"E:\OneDrive - Washington University in St. Louis\HessNetArchit"
spect_col = []
plt.figure()
plt.plot(S.cpu())
for _ in range(10):
    idx = torch.randperm(W.numel())
    W_shuf = W.view(-1)[idx].view(W.shape).detach().clone()
    with torch.no_grad():
        U_sf, S_sf, V_sf = torch.svd(W_shuf)
    plt.plot(S_sf)
    spect_col.append(S_sf.numpy())
plt.savefig(join(savedir, "spect_shuffle_cmp_defc7.jpg"))
plt.title("Comparison of Spectrum of Shuffled Matrix and Original one ")
plt.show()
#%%
W7 = SD['G.defc7.weight']
W6 = SD['G.defc6.weight']
W5 = SD['G.defc5.weight']
W7_rnd = torch.randn([4096,4096]) * SD['G.defc7.weight'].std().item() + SD['G.defc7.weight'].mean().item() # adding
# the mean will induce
W6_rnd = torch.randn([4096,4096]) * SD['G.defc6.weight'].std().item() + SD['G.defc6.weight'].mean().item()
W5_rnd = torch.randn([4096,4096]) * SD['G.defc5.weight'].std().item() + SD['G.defc5.weight'].mean().item()
W7_rndv = torch.randn([4096,4096]) * SD['G.defc7.weight'].std().item()
W6_rndv = torch.randn([4096,4096]) * SD['G.defc6.weight'].std().item()
W5_rndv = torch.randn([4096,4096]) * SD['G.defc5.weight'].std().item()
#%%
with torch.no_grad():
    U7, S7, V7 = torch.svd(W7)
    U67, S67, V67 = torch.svd(W6 @ W7)
    U567, S567, V567 = torch.svd(W5 @ W6 @ W7)
    Ur7, Sr7, Vr7 = torch.svd(W7_rnd)
    Ur67, Sr67, Vr67 = torch.svd(W6_rnd @ W7_rnd)
    Ur567, Sr567, Vr567 = torch.svd(W5_rnd @ W6_rnd @ W7_rnd)
    Urv7, Srv7, Vrv7 = torch.svd(W7_rndv)
    Urv67, Srv67, Vrv67 = torch.svd(W6_rndv @ W7_rndv)
    Urv567, Srv567, Vrv567 = torch.svd(W5_rndv @ W6_rndv @ W7_rndv)
    
#%%
plt.figure(figsize=[14,5])
plt.subplot(1,3,1)
plt.plot(S7, label="w7")
plt.plot(S67, label="w67")
plt.plot(S567, label="w567")
plt.legend()
plt.title("Original Weights")
plt.subplot(1,3,2)
plt.plot(Sr7, label="rnd_w7 ")
plt.plot(Sr67, label="rnd_w67")
plt.plot(Sr567, label="rnd_w567")
plt.legend()
plt.title("Gaussian Random Weights (same mean, std)")
plt.subplot(1,3,3)
plt.plot(Srv7, label="rnd_v_w7")
plt.plot(Srv67, label="rnd_v_w67")
plt.plot(Srv567, label="rnd_v_w567")
plt.legend()
plt.title("Gaussian Random Weights (same std)")
plt.suptitle("Comparison of Spectrum of Original weight and Gaussian Random Weight (mean and var kept)")
plt.savefig(join(savedir, "spect_randmat_cmp_W567.jpg"))
plt.show()