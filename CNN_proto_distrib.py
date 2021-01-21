from GAN_utils import upconvGAN
G = upconvGAN("fc6")
savedir = r"E:\Cluster_Backup\manif_allchan\resnet50_linf_8_.layer3.Bottleneck2_manifold-"
#%%
import numpy as np
from os.path import join
import matplotlib.pylab as plt
netname = "resnet50_linf_8"
layers = []


data = np.load(join(savedir, "Manifold_set_.layer3.Bottleneck2_995_7_7_rf_fit.npz"))
manifcentvec = 255*data['Perturb_vec'][0:1, :]
img = G.render(manifcentvec)
plt.imshow(img[0])
plt.show()
