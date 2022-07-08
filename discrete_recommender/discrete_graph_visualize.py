import sklearn.datasets
import pandas as pd
import numpy as np
import torch
from time import time
from os.path import join
import umap
import umap.plot
import matplotlib.pylab as plt
savedir = r"E:\OneDrive - Harvard University\CNN-PCs"
feattsr_penult = torch.load(join(savedir, "resnet50_swsl_INvalid_penult_feattsrs.pt"))
feattsrs = torch.load(join(savedir, "resnet50_swsl_INvalid_feattsrs.pt"))
#%%
t0 = time()
mapper = umap.UMAP(metric="cosine", n_neighbors=40).fit(feattsr_penult) #
print("%.2f sec "%(time() - t0))
#%%
ax = umap.plot.points(mapper, values=feattsr_penult[:, 2000],
                      width=2500, height=2500, cmap="Blues")
plt.show()
#%%
ax = umap.plot.points(mapper, values=feattsrs['.layer3.Bottleneck5'][:, 1000],
                      width=2500, height=2500, cmap="Blues")
plt.show()
#%%
ax = umap.plot.points(mapper, values=feattsr_penult[:, 0], theme='fire', width=1200,
    height=1200,)
plt.show()
#%%
umap.plot.connectivity(mapper, edge_bundling='hammer') # show_points=True,
plt.show()




#%% Sketches
# mnist = sklearn.datasets.fetch_openml('mnist_784')
# fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')
pendigits = sklearn.datasets.load_digits()
mapper = umap.UMAP().fit(pendigits.data)

#%%
# umap.plot.points(mapper)
umap.plot.points(mapper, labels=pendigits.target, width=1200,
    height=1200,)
plt.show()
#%%
umap.plot.points(mapper, values=pendigits.data.mean(axis=1), theme='fire')
plt.show()
#%%
umap.plot.points(mapper, labels=pendigits.target, color_key_cmap='Paired', background='black')
#%%
