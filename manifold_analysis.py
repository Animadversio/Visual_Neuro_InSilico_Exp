# This script is used to produce fitting and confidence interval for results in python.
#
#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

# x = np.array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
# y = np.array([ 4.70192769,  4.46826356,  4.57021389,  4.29240134,  3.88155125,
#                3.78382253,  3.65454727,  3.86379487,  4.16428541,  4.06079909])
# def func(x,c0, c1):
#     return c0 * np.exp(-x) + c1*x
# pars, pcov = curve_fit(func, x, y, p0=[4.96, 2.11])
#
# alpha = 0.05 # 95% confidence interval
# n = len(y)    # number of data points
# p = len(pars) # number of parameters
# dof = max(0, n-p) # number of degrees of freedom
# tval = t.ppf(1.0 - alpha / 2.0, dof) # student-t value for the dof and confidence level
# for i, p,var in zip(range(n), pars, np.diag(pcov)):
#     sigma = var**0.5
#     print('c{0}: {1} [{2}  {3}]'.format(i, p,
#                                   p - sigma*tval,
#                                   p + sigma*tval))

#%%
from numpy import cos, sin, exp, pi, meshgrid

def KentFunc(Xin, theta, phi, psi, kappa, beta, A):
    # Assume theta_z, phi_z are column vectors ([0,2 pi]), theta, phi, psi are
    # rotational scaler ([0,2 pi])
    theta_z, phi_z = Xin[:, 0], Xin[:, 1]
    Z = np.array([cos(theta_z) * cos(phi_z), sin(theta_z) * cos(phi_z), sin(phi_z)]).T  # M by 3 finally
    coord = SO3(theta, phi, psi)
    mu1 = coord[:, 0:1] # col vector
    # mu23 = coord[:, 1:3] # 2 col vectors, 3 by 2
    mu2 = coord[:, 1:2]  # 2 col vectors, 3 by 2
    mu3 = coord[:, 2:3]  # 2 col vectors, 3 by 2
    fval = A * exp(kappa * Z @ mu1 + beta * ((Z @ mu2) ** 2 - (Z @ mu3) ** 2))
    return fval[:, 0]

def SO3(theta, phi, psi):
    orig = np.array([[cos(theta)*cos(phi),  sin(theta)*cos(phi), sin(phi)],
                     [-sin(theta)       ,   cos(theta)        ,         0],
                     [cos(theta)*sin(phi), sin(theta)*sin(phi), -cos(phi)]]).T
    Rot23 = np.array([[1,         0 ,        0],
                      [0,   cos(psi), sin(psi)],
                      [0,  -sin(psi), cos(psi)]])
    coord = orig @ Rot23
    return coord
#%%
ang_step = 18
theta_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
fval = KentFunc(Xin, *[0, 0, pi/2, 0.1, 0.1, 1])
param, pcov = curve_fit(KentFunc, Xin, fval, p0=[-1, 1, pi/2, 0.1, 0.1, 0.2])
# Note python fitting will treat each data point as separate. He cannot estimate CI like matlab.

#%%
ang_step = 18
theta_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
fval = KentFunc(Xin, *[1, 0, pi/2, 0.1, 0.1, 0.1]) + np.random.randn(Xin.shape[0]) * 0.01
param, pcov = curve_fit(KentFunc, Xin, fval,
                        p0=[0, 0, pi/2, 0.1, 0.1, 0.1],
                        bounds=([-pi, -pi/2,  0, -np.inf,      0,      0],
                                [ pi,  pi/2,  pi, np.inf, np.inf, np.inf]))

print(param)
print(pcov)

#%%
import numpy as np
import lmfit

x = np.linspace(0.3, 10, 100)
np.random.seed(0)
y = 1/(0.1*x) + 2 + 0.1*np.random.randn(x.size)
pars = lmfit.Parameters()
pars.add_many(('a', 0.1), ('b', 1))
# pars.add_many(theta, phi, psi, kappa, beta, A)
#%%
theta_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_arr = np.arange(-90,90.1,ang_step) / 180 * pi
phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
fval = KentFunc(Xin, *[1, 0, pi/2, 0.1, 0.1, 0.1]) + np.random.randn(Xin.shape[0]) * 0.1
pars = lmfit.Parameters()
pars.add_many(("theta", 0, True, -pi, pi),
              ("phi", 0, True, -pi/2, pi/2),
              ("psi", pi/2, True, 0, pi),
              ("kappa", 0.1, True, -np.inf, np.inf),
              ("beta", 0.1, True, 0, np.inf),
              ("A", 0.2, True, 0, np.inf))
def residual(p):
    return KentFunc(Xin, p["theta"], p["phi"], p["psi"], p["kappa"], p["beta"], p["A"]) - fval
mini = lmfit.Minimizer(residual, pars)
result = mini.minimize()
print(lmfit.fit_report(result.params))
ci = lmfit.conf_interval(mini, result)
lmfit.printfuncs.report_ci(ci)
#%%
param, pcov = curve_fit(KentFunc, Xin, fval,
                        p0=[0, 0, pi/2, 0.1, 0.1, 0.1],
                        bounds=([-pi, -pi/2,  0, -np.inf,      0,      0],
                                [ pi,  pi/2,  pi, np.inf, np.inf, np.inf]))
np.diag(pcov)
n = len(fval)    # number of data points
p = len(pars) # number of parameters
dof = max(0, n-p)
tval = t.ppf(1.0 - 0.05 / 2.0, dof) # student-t value for the dof and confidence level
for i, p, var in zip(range(n), param, np.diag(pcov)):
    sigma = var**0.5
    print('c{0}: {1} [{2}  {3}]'.format(i, p,
                                  p - sigma*tval,
                                  p + sigma*tval))
#%%
def fit_Kent(theta_arr, phi_arr, act_map):
    phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
    Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
    fval = act_map.flatten()
    param, pcov = curve_fit(KentFunc, Xin, fval,
                            p0=[0, 0, pi / 2, 0.1, 0.1, 0.1],
                            bounds=([-pi, -pi / 2, 0, -np.inf, 0, 0],
                                    [pi, pi / 2, pi, np.inf, np.inf, np.inf]))
    sigmas = np.diag(pcov) ** 0.5
    return param, sigmas

#%% Load up data from in silico exp
ang_step = 9
theta_arr = np.arange(-90, 90.1, ang_step) / 180 * pi
phi_arr = np.arange(-90, 90.1, ang_step) / 180 * pi
#%% time
from time import time
t0 = time()
from os import listdir
from os.path import join, exists
result_dir = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Artiphysiology\Manifold"
netname = "caffe-net"
layers = ["conv1""conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"]
param_col = []
sigma_col = []
for i in range(len(layers)):
    lay_col = []
    lay_sgm = []
    savepath = join(result_dir, "%s_%s_manifold" % (netname, layers[i]))
    for ch_i in range(50):
        ch_col = []
        ch_sgm = []
        data = np.load(join(savepath, "score_map_chan%d.npz"%ch_i))
        score_sum = data['score_sum']
        subsp_axis = [(1, 2), (24, 25), (48, 49), "RND"]
        for subsp_j in range(len(subsp_axis)):
            tunemap = score_sum[subsp_j, :, :]
            param, sigmas = fit_Kent(theta_arr, phi_arr, tunemap)
            param_name = ["theta", "phi", "psi", "kappa", "beta", "A"]
            # for par, sgm, name in zip(param, sigmas, param_name):
            #     print(name, ": {}+-{}".format(par, sgm))
            ch_col.append(param.copy())
            ch_sgm.append(sigmas.copy())
        lay_col.append(ch_col.copy())
        lay_sgm.append(ch_sgm.copy())
        print(time()-t0,"s passed. ")
    param_col.append(lay_col.copy())
    sigma_col.append(lay_sgm.copy())


#%%

param_col_arr = np.array(param_col)
sigma_col_arr = np.array(sigma_col)
np.savez(join(result_dir,"KentFit.npz"), param_col=param_col_arr, sigma_col=sigma_col_arr, subsp_axis=subsp_axis, layers=layers)