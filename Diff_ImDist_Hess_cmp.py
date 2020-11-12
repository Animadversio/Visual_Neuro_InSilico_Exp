
from GAN_hessian_compute import hessian_compute
# from hessian_analysis_tools import scan_hess_npz, plot_spectra, average_H, compute_hess_corr, plot_consistency_example
# from hessian_axis_visualize import vis_eigen_explore, vis_eigen_action, vis_eigen_action_row, vis_eigen_explore_row
from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper, loadBigGAN, BigGAN_wrapper
import matplotlib.pylab as plt
import matplotlib

import lpips
ImDist = lpips.LPIPS(net="squeeze").cuda()
# L1

# L2

# SSIM

# mSSIM

#%% 


