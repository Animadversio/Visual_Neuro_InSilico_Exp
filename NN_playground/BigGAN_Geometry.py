"""This Script is devoted to analyze the geometry (distance) between the classes of BigGAN in the Embedding space"""
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, one_hot_from_int, truncated_noise_sample, convert_to_images)
import torch
from os.path import join
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
homedir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Generator_Testing"
#%%
model = BigGAN.from_pretrained('biggan-deep-256')
EmbedVectors = model.embeddings.weight.detach().numpy()
#%%

EbdVecNorms = norm(EmbedVectors, axis=0)
plt.figure()
plt.hist(EbdVecNorms, bins=30)
plt.title("Norm of 128d embed vector of BigGAN-Deep256")
plt.xlabel("Norm")
plt.savefig(join(homedir, "EmbedVectNormDist.png"))
plt.show()
#%% Mean Vector
meanEmbed = EmbedVectors.mean(axis=1)  # norm(meanEmbed) = 0.017733473
#%%
imgs_mean = BigGAN_embed_render(meanEmbed[np.newaxis, :], scale=1.0)
plt.figure(figsize=[4,4])
plt.imshow(imgs_mean[0])
plt.title("Mean Class Embedding")
plt.axis("image")
plt.axis("off")
plt.savefig(join(homedir, "class_mean_img.png"))
plt.show()
#%%
imgs_zeros = BigGAN_embed_render(np.zeros((1, 128)), scale=1.0)
plt.figure(figsize=[4,4])
plt.imshow(imgs_zeros[0])
plt.title("Zeros Vector visualized")
plt.axis("image")
plt.axis("off")
plt.savefig(join(homedir, "zero_vect_img.png"))
plt.show()

#%%
from numpy.linalg import svd
U, s, V = svd(EmbedVectors, full_matrices=False)
#%%
plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.plot(np.cumsum(s**2) / sum(s**2) * 100)
plt.ylabel("Explained Var by first nth PC")
plt.xlabel("PC number")
plt.subplot(122)
plt.plot(s**2 / sum(s**2) * 100)
plt.ylabel("Explained Var by nth PC")
plt.xlabel("PC number")
plt.suptitle("PCA Variance in the 128d Class Embedding")
plt.tight_layout()
plt.savefig(join(homedir, "explained_var_embed256.png"))
plt.show()
#%% Visualize PCA 123 components
plt.figure(figsize=[10, 10])
plt.subplot(221)
plt.scatter(s[0] * V[0, :], s[1] * V[1, :], 25, "blue", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.subplot(222)
plt.scatter(s[2] * V[2, :], s[1] * V[1, :], 25, "blue", alpha=0.6)
plt.ylabel("PC2")
plt.xlabel("PC3")
plt.axis("equal")
plt.subplot(224)
plt.scatter(s[2] * V[2, :], s[0] * V[0, :], 25, "blue", alpha=0.6)
plt.xlabel("PC3")
plt.ylabel("PC1")
plt.axis("equal")
plt.savefig(join(homedir, "PCA_DimenRed_ClassEmbed.png"))
plt.show()
#%%
from scipy.spatial import distance_matrix
dist_mat = distance_matrix(EmbedVectors.T, EmbedVectors.T)
# Clustering based on the distance matrix
#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000)
# tsne.fit(dist_mat)
vect_embedded = tsne.fit_transform(dist_mat)
#%%
plt.figure(figsize=[10, 10])
plt.scatter(vect_embedded[:, 0], vect_embedded[:, 1], 25, "blue", alpha=0.6)
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.axis("equal")
plt.savefig(join(homedir, "tSNE_DimenRed_ClassEmbed.png"))
plt.show()
#%%
plt.figure(figsize=[10, 10])
plt.pcolor(dist_mat)
plt.xlabel("ImageNet Class #")
plt.ylabel("ImageNet Class #")
plt.title("Distance Matrix between the 128d embed vectors")
plt.colormaps()
plt.colorbar()
plt.axis("image")
plt.savefig(join(homedir, "DistanceMatrix.png"))
#%%
plt.figure()
plt.hist(dist_mat[dist_mat!=0], bins=50)
plt.title("Distribution of Distance Matrix of BigGAN-Deep256")
plt.xlabel("L2 Distance")
plt.savefig(join(homedir, "EmbedVectDistanceDist.png"))
plt.show()
#%%
#%%
from nltk.corpus import wordnet as wn
import pickle, urllib
ImageNet_Classname = pickle.load(urllib.request.urlopen(
    'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )
#%% Visualize Images for each class
truncation = 0.7
class_vector = one_hot_from_int(list(range(1000)), batch_size=1000)
sample_n = class_vector.shape[0]
# noise_vector = np.zeros((sample_n, 128))
numbering = list(range(1000))
#%%
noise_vec = truncated_noise_sample(batch_size=1, dim_z=128, truncation=0.7, seed=None)
#%%
batch = 10
csr = 0
csr_end = 0
while csr_end < sample_n:
    csr_end = min(csr + batch, sample_n)
    # imgs = BigGAN_render(class_vector[csr:csr_end, :], noise_vector[csr:csr_end, :], truncation)
    imgs = BigGAN_render(class_vector[csr:csr_end, :], noise_vec, truncation)
    for number, img in zip(numbering[csr:csr_end], imgs):
        img.save(join(homedir, "Repr_imgs_wnoise3", "%03d_0.7noise0.7trunc.png" % number))
    csr = csr_end
#%%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=0.2)
#%%
fig, ax = plt.subplots(figsize=[30, 30])
ax.scatter(vect_embedded[:, 0], vect_embedded[:, 1])
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
for number in range(1000):
    path = join(homedir, "Repr_imgs_wnoise", "%03d_0.7noise0.7trunc.png" % number)
    ab = AnnotationBbox(getImage(path), (vect_embedded[number, 0], vect_embedded[number, 1]), frameon=False)
    ax.add_artist(ab)
fig.savefig(join(homedir, "0.7wnoise_sample_tSNE_Embed.png"))
plt.show()
#%%
fig, ax = plt.subplots(figsize=[30, 30])
ax.scatter(s[0] * V[0, :], s[1] * V[1, :])
plt.xlabel("PC1")
plt.ylabel("PC2")
for number in range(1000):
    path = join(homedir, "Repr_imgs_wnoise", "%03d_0.7noise0.7trunc.png" % number)
    ab = AnnotationBbox(getImage(path), (s[0] * V[0, number], s[1] * V[1, number]), frameon=False)
    ax.add_artist(ab)
fig.savefig(join(homedir, "0.7wnoise_sample_PC12_Embed.png"))
plt.show()
#%%
fig, ax = plt.subplots(figsize=[30, 30])
ax.scatter(s[1] * V[1, :], s[2] * V[2, :], )
plt.xlabel("PC2")
plt.ylabel("PC3")
for number in range(1000):
    path = join(homedir, "Repr_imgs_wnoise", "%03d_0.7noise0.7trunc.png" % number)
    ab = AnnotationBbox(getImage(path), (s[1] * V[1, number], s[2] * V[2, number]), frameon=False)
    ax.add_artist(ab)
fig.savefig(join(homedir, "0.7wnoise_sample_PC23_Embed.png"))
plt.show()