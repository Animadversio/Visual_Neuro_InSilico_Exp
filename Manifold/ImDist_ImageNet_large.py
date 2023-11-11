import matplotlib.pyplot as plt
import torch
from lpips import LPIPS
from NN_sparseness.visualize_sparse_inv_example import *
from NN_sparseness.sparse_invariance_lib import *
from NN_sparseness.sparse_plot_utils import scatter_density_grid
from torchvision.transforms.functional import rotate, affine, resize, center_crop
import seaborn as sns
from stats_utils import saveallforms

savedir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\ImageNet_imdist"

figdir = r"E:\OneDrive - Harvard University\Manifold_Sparseness\actlevel_tolerence_evol"
# INdataset = create_imagenet_valid_dataset(normalize=True)
INdataset = create_imagenet_valid_dataset(normalize=False)
#%%
D = LPIPS(net="squeeze").cuda()
D.requires_grad_(False)
#%%F
loader1 = DataLoader(INdataset, num_workers=8, shuffle=False, batch_size=64)
loader2 = DataLoader(INdataset, num_workers=8, shuffle=False, batch_size=64)
#%%

disttmp = D.forward_distmat(imgtsr1.cuda(), batch_size=24).flatten(start_dim=1)
distmatvals = disttmp[torch.triu_indices(*disttmp.shape, 1).unbind()]

#%%
distmat_all = []
for i, (imgtsr1, _) in enumerate(tqdm(loader1)):
    distmat_row = []
    # raise Exception("Stop")
    for imgtsr2, _ in tqdm(loader2):
        with torch.no_grad():
            distmat = D.forward_distmat(imgtsr1.cuda(), imgtsr2.cuda(), batch_size=24).cpu()
        torch.cuda.empty_cache()
        distmat_row.append(distmat)
    distmat_all.append(distmat_row)
    if i > 15:
        raise Exception
#%
savedir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\ImageNet_imdist"
# torch.save(distmat_all, join(savedir,"INet_imgdistmat.pt"))
torch.save(distmat_all, join(savedir,"INet_imgdistmat_nonnorm.pt")) # this version is used in the Manfiold paper
#%%
distmat_all = torch.load(join(savedir,"INet_imgdistmat.pt"))
#%%
dist_mat = []
for i in range(len(distmat_all)):
    distrow = torch.concat(distmat_all[i], dim=1).flatten(start_dim=1)
    dist_mat.append(distrow)
#%%
dist_mat = torch.concat(dist_mat, dim=0)
#%%
savedir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\ImageNet_imdist"
torch.save(dist_mat, join(savedir, "INet_imgdistmat_cat.pt"))

#%%
dist_mat = torch.load(join(savedir, "INet_imgdistmat_cat.pt"))
#%%
dist_mat_sort, dist_mat_idx = torch.sort(dist_mat, dim=1)
#%%
distmat_sq = dist_mat[:, :dist_mat.shape[0]]
distmatvals = distmat_sq[torch.triu_indices(1088, 1088, 1).unbind()]
#%%
NN_dist = dist_mat_sort[:,1]
NNmean = torch.mean(NN_dist)
NNstd = torch.std(NN_dist)
NNqtl = torch.quantile(NN_dist, torch.tensor([0.05,0.95]))
print(f"{NNmean:.3f}+-{NNstd:.3f} 5%-95% quantile [{NNqtl[0]:.3f},{NNqtl[1]:.3f}]")
#%%
#%%
figdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\ImageNet_imdist"
#%%
figh = plt.figure(figsize=[5,5])
sns.histplot(dist_mat_sort[:,1], bins=100, kde=True)
# sns.kdeplot(dist_mat_sort[:,1], )
YLIM = plt.ylim()
plt.vlines(NNmean, *YLIM, color="k", lw=2, label="mean")
plt.vlines([NNmean+NNstd, NNmean-NNstd], *YLIM, color="r", lw=2, label="mean+-std")
plt.legend()
plt.xlabel("LPIPS distance (SqueezeNet)")
plt.title("Distribution of distance to nearest neighbor\nImageNet validation")
saveallforms(figdir, "INet_NNdist_hist", )
plt.show()