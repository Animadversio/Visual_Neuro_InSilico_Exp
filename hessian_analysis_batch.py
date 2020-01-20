
import sys
import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from caffenet import *
from hessian import hessian
#%%
import utils
from utils import generator
from insilico_Exp import CNNmodel
#%%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
#%%
hess_dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Artiphysiology\Hessian"
output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"
#%%
unit_arr = [
            ('caffe-net', 'conv1', 10, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc6', 2),
            ('caffe-net', 'fc6', 3),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc7', 2),
            ('caffe-net', 'fc8', 1),
            ('caffe-net', 'fc8', 10),
            ]
unit = ('caffe-net', 'fc8', 1)
CNNmodel = CNNmodel(unit[0])  # 'caffe-net'
CNNmodel.select_unit(unit)
data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
z = data["z"]
G = data["grad"]
Heig = data["heig"]
Heigvec = data["heigvec"]
os.makedirs((join(hess_dir, "%s_%s_%d"%unit)),exist_ok=True)

eigen_idx = -2
eigen_val = Heig[eigen_idx]

stepsize = 5
RNG = np.arange(-20, 21)
img_list = perturb_images_line(z, Heigvec[:, eigen_idx], PC2_step=stepsize, RNG=RNG)
scores = CNNmodel.score(img_list)
figh, ax_tune = visualize_img_and_tuning(img_list, scores, DS_num=4)
figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E"%(unit, eigen_idx, eigen_val), fontsize=16)
#figh.show()
figh.savefig(join(hess_dir,"%s_%s_%d"%unit,"Tuning_eigid_%d"%eigen_idx))

stepsize = 5
RNG = np.arange(-20, 21)
img_list = perturb_images_arc(z, Heigvec[:, eigen_idx], PC2_ang_step=stepsize, RNG=RNG)
scores = CNNmodel.score(img_list)
figh, ax_tune = visualize_img_and_tuning(img_list, scores, DS_num=4)
ax_tune.set_xlabel("Code Angular Distance (deg)")
figh.suptitle("%s\nEigen Vector No. %d, Eigen Value %.3E"%(unit, eigen_idx, eigen_val),fontsize=16)
#figh.show()
figh.savefig(join(hess_dir,"%s_%s_%d"%unit,"Ang_Tuning_eigid_%d"%eigen_idx))
