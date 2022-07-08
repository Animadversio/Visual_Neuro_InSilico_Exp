#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
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


def KentFunc_bsl(Xin, theta, phi, psi, kappa, beta, A, bsl):
    # Assume theta_z, phi_z are column vectors ([0,2 pi]), theta, phi, psi are
    # rotational scaler ([0,2 pi])
    theta_z, phi_z = Xin[:, 0], Xin[:, 1]
    Z = np.array([cos(theta_z) * cos(phi_z), sin(theta_z) * cos(phi_z), sin(phi_z)]).T  # M by 3 finally
    coord = SO3(theta, phi, psi)
    mu1 = coord[:, 0:1] # col vector
    # mu23 = coord[:, 1:3] # 2 col vectors, 3 by 2
    mu2 = coord[:, 1:2]  # 2 col vectors, 3 by 2
    mu3 = coord[:, 2:3]  # 2 col vectors, 3 by 2
    fval = A * exp(kappa * Z @ mu1 + beta * ((Z @ mu2) ** 2 - (Z @ mu3) ** 2)) + bsl
    return fval[:, 0]


def SO3(theta, phi, psi):
    orig = np.array([[cos(theta)*cos(phi),  sin(theta)*cos(phi), sin(phi)],
                     [-sin(theta)       ,   cos(theta)        ,         0],
                     [cos(theta)*sin(phi), sin(theta)*sin(phi), -cos(phi)]]).T
    Rot23 = np.array([[1,          0,        0],
                      [0,   cos(psi), sin(psi)],
                      [0,  -sin(psi), cos(psi)]])
    coord = orig @ Rot23
    return coord
#%
def fit_Kent(theta_arr, phi_arr, act_map):
    phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
    Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
    fval = act_map.flatten()
    try:  # avoid fitting failure to crash the whole thing.
        param, pcov = curve_fit(KentFunc, Xin, fval,
                                p0=[0, 0, pi / 2, 0.1, 0.1, 1],
                                bounds=([-pi, -pi / 2, 0, 0, 0, 0],
                                        [pi, pi / 2, pi, np.inf, np.inf, np.inf]))
        sigmas = np.diag(pcov) ** 0.5
        return param, sigmas
    except RuntimeError as err:
        print(type(err))
        print(err)
        return np.ones(6)*np.nan, np.ones(6)*np.nan

def fit_Kent_bsl(theta_arr, phi_arr, act_map):
    """ Fit Kent function with baseline
    : param = [theta, phi, psi, kappa, beta, A, bsl]
    """
    phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
    Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
    fval = act_map.flatten()
    try:  # avoid fitting failure to crash the whole thing.
        # param, pcov = curve_fit(KentFunc_bsl, Xin, fval,
        #                         p0=[0, 0, pi / 2, 0.1, 0.1, 1, 0.001],
        #                         bounds=([-pi, -pi / 2, 0, 0, 0, 0, 0],
        #                                 [pi, pi / 2, pi, np.inf, np.inf, np.inf, np.inf]))
        param, pcov = curve_fit(KentFunc_bsl, Xin, fval,
                                p0=[0, 0, pi / 2, 0.1, 0.1, 1, 0.001],
                                bounds=([-pi/2, -pi / 2, 0, 0, 0, 0, 0],
                                        [pi/2, pi / 2, pi, np.inf, np.inf, np.inf, np.inf]))
        sigmas = np.diag(pcov) ** 0.5
        return param, sigmas
    except RuntimeError as err:
        print(type(err))
        print(err)
        return np.ones(7)*np.nan, np.ones(7)*np.nan
#%
def fit_stats(theta_arr, phi_arr, act_map, param, func=KentFunc):
    """Generate fitting statistics from scipy's curve fitting"""
    phi_grid, theta_grid = meshgrid(phi_arr, theta_arr)
    Xin = np.array([theta_grid.flatten(), phi_grid.flatten()]).T
    fval = act_map.flatten()
    fpred = func(Xin, *param)  # KentFunc
    res = fval - fpred
    rsquare = 1 - (res**2).mean() / fval.var()
    return res.reshape(act_map.shape), rsquare


def fit_Kent_Stats(theta_arr, phi_arr, act_map, func=KentFunc_bsl):
    """ Combine function fitting and getting statistics """
    if func is KentFunc:
        param, sigmas = fit_Kent(theta_arr, phi_arr, act_map)
    elif func is KentFunc_bsl:
        param, sigmas = fit_Kent_bsl(theta_arr, phi_arr, act_map)
    else:
        raise ValueError('Unknown fitting function')
    if np.any(np.isnan(param)):
        res, R2 = np.ones_like(act_map)*np.nan, np.nan
    else:
        res, R2 = fit_stats(theta_arr, phi_arr, act_map, param, func=func)
    return param, sigmas, res, R2
