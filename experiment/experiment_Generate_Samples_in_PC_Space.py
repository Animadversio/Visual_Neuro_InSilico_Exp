'''
Experimental Code to generate samples in selected PC space from an experiment
depending on `utils` for code loading things
'''
from  scipy.io import loadmat
import os
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from utils_old import generator
import utils_old
from GAN_utils import upconvGAN
G = upconvGAN("fc6").cuda()
#%%
'''
Input the experimental backup folder containing the mat codes files.
'''
backup_dir = r"N:\Stimuli\2021-ProjectPFC\2021-Evolutions\2021-06-30-Caos-01\2021-06-30-10-52-27"

newimg_dir = os.path.join(backup_dir,"PC_imgs")

#%%
os.makedirs(newimg_dir, exist_ok=True)
print("Save new images to folder %s", newimg_dir)
#%%
print("Loading the codes from experiment folder %s", backup_dir)
codes_all, generations = utils_old.load_codes_mat(backup_dir)
generations = np.array(generations)
print("Shape of code", codes_all.shape)
#%%
final_gen_norms = np.linalg.norm(codes_all[generations==max(generations), :], axis=1)
final_gen_norm = final_gen_norms.mean()
print("Average norm of the last generation samples %.2f" % final_gen_norm)

sphere_norm = final_gen_norm
print("Set sphere norm to the last generations norm!")
#%%
print("Computing PCs")
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1
# %% Spherical interpolation

PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
# sphere_norm = 300
print("Generating images on PC1, PC2, PC3 sphere (rad = %d)" % sphere_norm)
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ PC_vectors[0:3, :]
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        # img = generator.visualize(code_vec)
        # img_list.append(img.copy())
        img = G.render(code_vec)[0]
        img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (sphere_norm, PC2_ang_step * j, PC3_ang_step* k)), img)

fig1 = utils_old.visualize_img_list(img_list)

# %% Spherical interpolation
PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
# sphere_norm = 300
print("Generating images on PC1, PC49, PC50 sphere (rad = %d)" % sphere_norm)
PC_nums = [0, 48, 49]  # can tune here to change the selected PC to generate images
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ PC_vectors[PC_nums, :]
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        # img = generator.visualize(code_vec)
        # img_list.append(img.copy())
        img = G.render(code_vec)[0]
        img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_PC%d_%d_PC%d_%d.jpg" % (sphere_norm,
                                                                            PC_nums[1] + 1, PC2_ang_step * j,
                                                                            PC_nums[2] + 1, PC3_ang_step * k)), img)
fig2 = utils_old.visualize_img_list(img_list)

# %% Spherical interpolation
PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
# sphere_norm = 300
print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d)" % sphere_norm)
# Random select and orthogonalize the vectors to form the sphere
rand_vec2 = np.random.randn(2, 4096)
rand_vec2 = rand_vec2 - (rand_vec2 @ PC_vectors.T) @ PC_vectors
rand_vec2 = rand_vec2 / np.sqrt((rand_vec2**2).sum(axis=1))[:, np.newaxis]
vectors = np.concatenate((PC_vectors[0:1, :], rand_vec2))
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ vectors
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        # img = generator.visualize(code_vec)
        # img_list.append(img.copy())
        img = G.render(code_vec)[0]
        img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_RND1_%d_RND2_%d.jpg" % (sphere_norm, PC2_ang_step * j, PC3_ang_step * k)), img)
fig3 = utils_old.visualize_img_list(img_list)

fig1.savefig(os.path.join(backup_dir,"Norm%d_PC12_Montage.png"%sphere_norm))
fig2.savefig(os.path.join(backup_dir,"Norm%d_PC4950_Montage.png"%sphere_norm))
fig3.savefig(os.path.join(backup_dir,"Norm%d_RND12_Montage.png"%sphere_norm))
np.savez(os.path.join(newimg_dir, "PC_vector_data.npz"), PC_vecs=PC_vectors, rand_vec2=rand_vec2,
         sphere_norm=sphere_norm, PC2_ang_step=PC2_ang_step, PC3_ang_step=PC3_ang_step)