#%%
figdir = "E:\OneDrive - Washington University in St. Louis\HessNetArchit"
from time import time
from collections import OrderedDict
from cycler import cycler
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
from matplotlib.pylab import plt
from GAN_utils import upconvGAN, np, join
G = upconvGAN("fc6")
#%%
def SingularValues(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)

SV = SingularValues(W.transpose((2,3,0,1)), (128,128))
#%%
import torch
code = torch.zeros((1,4096))
Fm = G.G[:7](code)
FeatMapSize = {}
FeatMapList = {}
for li in range(len(G.G)):
    layername = list(G.G.named_children())[li][0]
    Fm = G.G[:li](code)
    FeatMapSize[layername] = Fm.shape
    FeatMapList[li] = Fm.shape
#%%
SV = SingularValues(W.transpose((2,3,0,1)), (128,128))
#%%
W7 = G.G.defc7.weight.data.numpy()
W6 = G.G.defc6.weight.data.numpy()
W5 = G.G.defc5.weight.data.numpy()
#%%
W5 = G.G.deconv5.weight.data.numpy()
W5_1 = G.G.conv5_1.weight.data.numpy()
W4 = G.G.deconv4.weight.data.numpy()
W4_1 = G.G.conv4_1.weight.data.numpy()
W3 = G.G.deconv3.weight.data.numpy()
W3_1 = G.G.conv3_1.weight.data.numpy()
W2 = G.G.deconv2.weight.data.numpy()
W1 = G.G.deconv1.weight.data.numpy()
W0 = G.G.deconv0.weight.data.numpy()
#%%

T0 = time()
laynms = ["deconv5","conv5_1","deconv4","conv4_1","deconv3","conv3_1","deconv2","deconv1","deconv0",]
npWgts = [W5, W5_1, W4, W4_1, W3, W3_1, W2, W1, W0, ]
SgVals = {}
for layername, W in zip(laynms, npWgts):
    FmSz = FeatMapSize[layername]
    SV = SingularValues(W.transpose((2,3,0,1)), FmSz[-2:])
    SgVals[layername] = SV
print("%.1f sec"%(time()-T0))
#%%
fclaynms = ["defc7", "defc6", "defc5", ]
for layername, W in zip(fclaynms, [W7, W6, W5]):
    U, S, V = np.linalg.svd(W)
    SgVals[layername] = S
#%%

SV_all = OrderedDict()
for layername in fclaynms+laynms:
    SV_all[layername] = SgVals[layername]
#%%
#%%
fig,ax = plt.subplots()
colormap = plt.get_cmap('jet')
ax.set_prop_cycle(cycler(color=[colormap(k) for k in np.linspace(0, 1, 12)]))
for layername, SV in SV_all.items():
    print(layername, SV.shape)
    SVnum = np.prod(SV.shape)
    plt.plot(np.arange(SVnum)/SVnum, np.sort(SV, axis=None)[::-1])# np.sort(SV.reshape(-1)))
plt.ylabel("SV")
plt.xlabel("Singular Value Id (Ratio of All SV that layer)")
plt.title("Singular Value per Layer in FC6 GAN")
plt.legend(fclaynms+laynms)
plt.savefig(join(figdir, "SV_per_layer_ratio.png"))
plt.savefig(join(figdir, "SV_per_layer_ratio.pdf"))
plt.show()
#%%
fig,ax = plt.subplots()
colormap = plt.get_cmap('jet')
ax.set_prop_cycle(cycler(color=[colormap(k) for k in np.linspace(0, 1, 12)]))
for layername, SV in SV_all.items():
    print(layername, SV.shape)
    plt.plot(np.sort(SV, axis=None)[::-1])
plt.ylabel("SV")
plt.xlabel("Singular Value Id")
plt.title("Singular Value per Layer in FC6 GAN")
plt.legend(fclaynms+laynms)
plt.savefig(join(figdir, "SV_per_layer.png"))
plt.savefig(join(figdir, "SV_per_layer.pdf"))
plt.show()
#%%
fig,ax = plt.subplots()
colormap = plt.get_cmap('jet')
ax.set_prop_cycle(cycler(color=[colormap(k) for k in np.linspace(0, 1, 12)]))
for layername, SV in SV_all.items():
    print(layername, SV.shape)
    plt.plot(np.log10(np.sort(SV, axis=None)[::-1]))
plt.ylabel("log_10(SV)")
plt.xlabel("Singular Value Id")
plt.title("Singular Value per Layer in FC6 GAN")
plt.legend(fclaynms+laynms)
plt.savefig(join(figdir, "SV_log_per_layer.png"))
plt.savefig(join(figdir, "SV_log_per_layer.pdf"))
plt.show()
#%%
fig,ax = plt.subplots()
colormap = plt.get_cmap('jet')
ax.set_prop_cycle(cycler(color=[colormap(k) for k in np.linspace(0, 1, 12)]))
for layername, SV in SV_all.items():
    print(layername, SV.shape)
    SVnum = np.prod(SV.shape)
    plt.plot(np.arange(SVnum) / SVnum, np.log10(np.sort(SV, axis=None)[::-1]))
plt.ylabel("log_10(SV)")
plt.xlabel("Singular Value Id (Ratio of All SV that layer)")
plt.title("Singular Value per Layer in FC6 GAN")
plt.legend(fclaynms+laynms)
plt.savefig(join(figdir, "SV_log_per_layer_ratio.png"))
plt.savefig(join(figdir, "SV_log_per_layer_ratio.pdf"))
plt.show()
#%%
for layername, SV in SV_all.items():
    print(layername, SV.shape, "Max %.1f 95%% %.1f 50%% %.1f Min %.1e"%( SV.max(), np.percentile(SV, 95), np.percentile(SV, 50), SV.min(), ))
#%%
np.savez(join(figdir,"SV_per_layer.npz"),**SV_all)
#%%
data = np.load(join(figdir,"SV_per_layer.npz"),allow_pickle=True)
# (deconv5): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# (relu_deconv5): LeakyReLU(negative_slope=0.3, inplace=True)
# (conv5_1): ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (relu_conv5_1): LeakyReLU(negative_slope=0.3, inplace=True)
# (deconv4): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# (relu_deconv4): LeakyReLU(negative_slope=0.3, inplace=True)
# (conv4_1): ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (relu_conv4_1): LeakyReLU(negative_slope=0.3, inplace=True)
# (deconv3): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# (relu_deconv3): LeakyReLU(negative_slope=0.3, inplace=True)
# (conv3_1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# (relu_conv3_1): LeakyReLU(negative_slope=0.3, inplace=True)
# (deconv2): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# (relu_deconv2): LeakyReLU(negative_slope=0.3, inplace=True)
# (deconv1): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
# (relu_deconv1): LeakyReLU(negative_slope=0.3, inplace=True)
# (deconv0): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))