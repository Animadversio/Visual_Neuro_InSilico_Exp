from hessian_axis_visualize import vis_eigen_frame, vis_eigen_action, vis_distance_curve, vis_eigen_explore
from hessian_analysis_tools import scan_hess_npz, average_H, compute_hess_corr, plot_consistentcy_mat, \
    plot_consistency_hist, plot_consistency_example, plot_spectra
from GAN_utils import loadBigGAN, loadBigBiGAN, loadStyleGAN2, BigGAN_wrapper, BigBiGAN_wrapper, StyleGAN2_wrapper, \
    loadStyleGAN, StyleGAN_wrapper, upconvGAN, PGGAN_wrapper, loadPGGAN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
rootdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"
"""Note the loading and visualization is fully deterministic, reproducible."""
#%%

