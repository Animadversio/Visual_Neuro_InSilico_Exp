"""
This script summarize and the Hessian computation for StyleGAN2
Analyze the geometry of the BigGAN manifold. How the metric tensor relates to the coordinate.
"""
import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from time import time
from os.path import join
from imageio import imwrite
from build_montages import build_montages, color_framed_montages
SGdir = r"E:\Cluster_Backup\StyleGAN2"
Hdir = "E:\Cluster_Backup\StyleGAN2\stylegan2-cat-config-f"
#%%
