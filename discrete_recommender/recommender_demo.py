"""Use kernel regression with deep net kernels to predict firing rate and to optimize peak firing rate
images in a diverse set.
"""
from insilico_Exp_torch import TorchScorer
from layer_hook_utils import featureFetcher, get_module_names, get_layer_names, register_hook_by_module_names
#%
from tqdm import tqdm
import torch
from imageio import imread
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader  #, ImageDataset
from torchvision.transforms import Compose, Resize, Normalize, ToPILImage, ToTensor
#%%
RGB_mean = torch.tensor([0.485, 0.456, 0.406])#.view(1,-1,1,1).cuda()
RGB_std  = torch.tensor([0.229, 0.224, 0.225])#.view(1,-1,1,1).cuda()
preprocess = Compose([ToTensor(),
                      Resize(256, ),
                      Normalize(RGB_mean, RGB_std),
                      ])
dataset = ImageFolder(r"E:\Datasets\kay-shared1000", transform=preprocess)
loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
#%% Self supervised representation
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
#%%
fetcher = featureFetcher(model, device="cuda")
fetcher.record(".Linearfc", return_input=True, ingraph=False)
#%%
model.eval().cuda()
feat_col = []
for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
    with torch.no_grad():
        model(imgtsr.cuda())
    feats = fetcher[".Linearfc"][0].cpu()
    feat_col.append(feats)

feattsr = torch.cat(tuple(feat_col), dim=0)
#%%
normfeat = feattsr / feattsr.norm(dim=1, keepdim=True)
cosmat = normfeat @ normfeat.T
#%% Analyze and visualize the kernel matrix
import numpy as np
import matplotlib.pylab as plt
plt.imshow(cosmat)
plt.show()
#%%
plt.figure()
plt.hist(cosmat.flatten().numpy(), bins=300, histtype='step')
plt.show()
#%% Spectrum of cosine matrix and (normalized) features
eva, evc = np.linalg.eigh(cosmat)
assert np.allclose(evc @ np.diag(eva) @ evc.T, cosmat)
Un, Sn, Vhn = np.linalg.svd(normfeat, full_matrices=False)
assert np.allclose(Un @ np.diag(Sn) @ Vhn, normfeat, rtol=1E-6, atol=1E-6)
U, S, Vh = np.linalg.svd(feattsr, full_matrices=False)
# assert np.allclose(U @ np.diag(S) @ Vh, feattsr, rtol=1E-6)
#%% Visualize the cosine matrix
plt.figure()
plt.plot(np.log(eva[::-1])-np.log(max(eva)), label="Cosine Matrix Eigval")
# plt.show()
plt.plot(np.log(Sn)-np.log(max(Sn)), label="Normed Feature singular val")
# plt.show()
plt.plot(np.log(S)-np.log(max(S)), label="Normed Feature singular val")
plt.legend()
plt.show()
#%%
def topK(vector, K, max=True):
    if max:
        ind = np.argpartition(vector, -K)[-K:]
        vals = vector[ind]
        rel_ids = np.argsort(-vals, )  # descending order
        return ind[rel_ids], vals[rel_ids]
    else:
        ind = np.argpartition(vector, K)[:K]
        vals = vector[ind]
        rel_ids = np.argsort(vals, ) # ascending order
        return ind[rel_ids], vals[rel_ids]


def top_k_idvals(distmat, k, query_idx):
    ind = np.argpartition(distmat[query_idx, ], -(k+1))[-(k+1):-1]  # exclude the closest which is usually itself
    vals = distmat[query_idx, ind]
    return ind, distmat[query_idx, ind]


def show_imgtsr(imgtsr, nrow=5, denorm=True):
    RGB_mean = torch.tensor([0.485, 0.456, 0.406])
    RGB_std = torch.tensor([0.229, 0.224, 0.225])
    if denorm:
        denorm_imgtsr = Normalize(-RGB_mean / RGB_std, 1 / RGB_std)(imgtsr)
    else:
        denorm_imgtsr = imgtsr
    pilimg = ToPILImage()((make_grid(denorm_imgtsr, nrow=nrow)))
    pilimg.show()
    return pilimg

#%%
from sklearn.kernel_ridge import KernelRidge
class DiscreteOptimizer:
    def __init__(self, kermat, alpha=0.1):#imgN
        imgN = kermat.shape[0]
        self.imgN = imgN
        self.model = KernelRidge(alpha=alpha, kernel="precomputed")
        self.Nrep = np.zeros((imgN,), dtype=np.int)
        self.predscore = np.zeros((imgN,), dtype=np.float)
        self.kermat = kermat # np.zeros(imgN, imgN)
        self.score_col = [[] for i in range(imgN)]
        self.score_sum = np.zeros((imgN,), dtype=np.float)
        self.score_mean = np.zeros((imgN,), dtype=np.float)
        self.score_std = np.zeros((imgN,), dtype=np.float)

    def update(self, idxs, scores):
        # Update repeat time
        self.Nrep[idxs] = self.Nrep[idxs] + 1
        self.score_sum[idxs] += scores
        for i, idx in enumerate(idxs):
            self.score_col[idx].append(scores[i])
            self.score_std[idx] = np.std(self.score_col[idx], ddof=1)

        self.score_mean = self.score_sum / self.Nrep
        known_idx = np.nonzero(self.Nrep)[0]
        known_scores = self.score_mean[known_idx]
        self.model.fit(self.kermat[:, known_idx][known_idx, :], known_scores)
        self.predscore = self.model.predict(self.kermat[:, known_idx])

    # def predict(self, idx=None):
    #     self.model

    def propose(self, randsamp=5, size=40, MaxRep=5, maximize=True):
        if maximize:
            sortidxs = np.argsort(-self.predscore)
        else:
            sortidxs = np.argsort(self.predscore)
        # randidxs = np.random.randint(self.imgN, size=randsamp)
        sampprob = 1 / (self.Nrep + 1)
        oversampled = sampprob >= MaxRep
        sampprob[oversampled] = 1E-8
        sampprob /= sampprob.sum()
        randidxs = np.random.choice(self.imgN, size=randsamp, replace=False, p=sampprob)
        return np.hstack((sortidxs[:size], randidxs))

#%%
#%%
init_idx = np.random.randint(1000)
# top_k_idvals(cosmat, 10, init_idx)
inds, distvals = topK(cosmat[init_idx], 15, True)  # closest 15 images
NNimgtsr = torch.stack([dataset[idx][0] for idx in inds])
show_imgtsr(NNimgtsr, nrow=5)
inds, distvals = topK(cosmat[init_idx], 15, False)  # farthest 15 images
NNimgtsr = torch.stack([dataset[idx][0] for idx in inds])
show_imgtsr(NNimgtsr, nrow=5)
#%%
scorer = TorchScorer("resnet50_linf_8", )
scorer.select_unit(("resnet50_linf_8", ".layer3.Bottleneck5", 100, 5, 5))
kermat = cosmat.numpy()-0.15
Doptim = DiscreteOptimizer(kermat)
#%%
popsize = 20
rndsize = 20
init_idxs = np.random.randint(0, 1000, size=40)
gen_idxs = init_idxs
for geni in range(100):
    imgtsr = torch.stack([dataset[idx][0] for idx in gen_idxs])
    scores = scorer.score_tsr(imgtsr)
    Doptim.update(gen_idxs, scores)
    gen_idxs = Doptim.propose(randsamp=rndsize, size=popsize)
    print("Gen %d Recorded %.3f+-%.3f Explore %.3f+-%.3f\tPred %.3f+-%.3f Explore %.3f+-%.3f"%
          (geni, scores[:popsize].mean(), scores[:popsize].std(), scores[popsize:].mean(), scores[popsize:].std(),
          Doptim.predscore[gen_idxs[:popsize]].mean(), Doptim.predscore[gen_idxs[:popsize]].std(),
          Doptim.predscore[gen_idxs[popsize:]].mean(), Doptim.predscore[gen_idxs[popsize:]].std()))

#%%
kermat = cosmat.numpy()
KRmodel = KernelRidge(alpha=1.0, kernel="precomputed")
#%%
init_idxs = np.random.randint(0, 1000, size=40)
init_imgtsr = torch.stack([dataset[idx][0] for idx in init_idxs])
scores = scorer.score_tsr(init_imgtsr)
KRmodel.fit(kermat[init_idxs, :][:, init_idxs], scores)
#%%
pred_score = KRmodel.predict(kermat[:, init_idxs])
#%%
ranked_idx = np.argsort(pred_score)
pred_score[ranked_idx[-20:]]
pred_score[ranked_idx[:20]]
#%%
torch.stack([dataset[idx][0] for idx in ranked_idx[-40:]])
scores_next = scorer.score_tsr(init_imgtsr)
#%%
# score_full = np.ones((1000,)) * np.nan
# score_full[init_idxs] = scores
#%%
import numpy.ma as ma
print(ma.corrcoef(ma.masked_invalid(Doptim.score_mean), ma.masked_invalid(Doptim.predscore)))
#%%
# KRmodel.fit(cosmat.numpy(), score_full)
plt.scatter(Doptim.score_mean, Doptim.predscore)
plt.show()
#%%
pred_max_idx = np.argsort(Doptim.predscore)[-20:]
imgtsr = torch.stack([dataset[idx][0] for idx in pred_max_idx])
show_imgtsr(imgtsr, nrow=5)
print("Actual activ %.3f+-%.3f; Predict activ %.3f+-%.3f"%
      (Doptim.score_mean[pred_max_idx].mean(), Doptim.score_mean[pred_max_idx].std(),
       Doptim.predscore[pred_max_idx].mean(), Doptim.predscore[pred_max_idx].std(), ))
#%%
pred_min_idx = np.argsort(Doptim.predscore)[:20]
imgtsr = torch.stack([dataset[idx][0] for idx in pred_min_idx])
show_imgtsr(imgtsr, nrow=5)
print("Actual activ %.3f+-%.3f; Predict activ %.3f+-%.3f"%
      (Doptim.score_mean[pred_min_idx].mean(), Doptim.score_mean[pred_min_idx].std(),
       Doptim.predscore[pred_min_idx].mean(), Doptim.predscore[pred_min_idx].std(), ))
#%%
nannum = np.isnan(Doptim.score_mean).sum()
actu_max_idx = np.argsort(Doptim.score_mean)[-20-nannum:-nannum]
imgtsr = torch.stack([dataset[idx][0] for idx in actu_max_idx])
show_imgtsr(imgtsr, nrow=5)
print("Actual activ %.3f+-%.3f; Predict activ %.3f+-%.3f"%
      (Doptim.score_mean[actu_max_idx].mean(), Doptim.score_mean[actu_max_idx].std(),
       Doptim.predscore[actu_max_idx].mean(), Doptim.predscore[actu_max_idx].std(), ))
#%%
nannum = np.isnan(Doptim.score_mean).sum()
actu_min_idx = np.argsort(-Doptim.score_mean)[-20-nannum:-nannum]
imgtsr = torch.stack([dataset[idx][0] for idx in actu_min_idx])
show_imgtsr(imgtsr, nrow=5)
print("Actual activ %.3f+-%.3f; Predict activ %.3f+-%.3f"%
      (Doptim.score_mean[actu_min_idx].mean(), Doptim.score_mean[actu_min_idx].std(),
       Doptim.predscore[actu_min_idx].mean(), Doptim.predscore[actu_min_idx].std(), ))

#%% Direct fitting and generalization experiment
scorer = TorchScorer("resnet50_linf_8", )
scorer.select_unit(("resnet50_linf_8", ".layer3.Bottleneck0", 5, 5, 5))
#%%
score_col = []
for ibatch, (imgtsr, label) in tqdm(enumerate(loader)):
    scores = scorer.score_tsr(imgtsr)
    score_col.append(scores)
score_all = np.concatenate(tuple(score_col), axis=0)
#%%
Nimg = 1000
Ntrain = 800
train_idx = np.random.choice(Nimg, size=Ntrain, replace=False)
test_idx = np.setdiff1d(np.arange(Nimg), train_idx)
#%% Generalization poorly based on a single layer representation similarity matrix.
Doptim = DiscreteOptimizer(kermat, alpha=0.3)
Doptim.update(train_idx, score_all[train_idx])
print("Training set corr %.3f Testing set corr %.3f All corr %.3f"%(
        np.corrcoef(Doptim.predscore[train_idx], score_all[train_idx])[0,1],
        np.corrcoef(Doptim.predscore[test_idx], score_all[test_idx])[0,1],
        np.corrcoef(Doptim.predscore, score_all)[0,1]))
#%%
KRmodel.fit(kermat[train_idx, :][:, train_idx], score_all[train_idx])

