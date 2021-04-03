#!/usr/bin/env python3

import sqlite3
from matplotlib import cm
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

from params import *
import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True

plt.rc('font', family='serif', size='12')

with open(fname+'clusters.dat', 'rb') as f:
    clusters = pickle.load(f)



#%%
##########################
# RA VS DEC DENSITY PLOT #
##########################
# import pandas as pd
# df = pd.read_csv('FoF\\processing\\datasets\\cosmos2015_dataset.csv')
# columns = ['ra', 'dec', 'ID', 'zphot', 'redshift', 'z_lower', 'z_upper+', 'RMag', 'Class', 'log_Stellar_Mass', 'log_SFR', 'LR'] # renaming columns
# df.columns = columns
# df = df[(df['redshift'] <= 2.5)] 
# df = df[df['RMag'] >= -50]

# fig, ax = plt.subplots(figsize=(10,8))
# h = ax.hist2d(df['ra'], df['dec'], bins=(100,80), cmap='viridis')
# cbar = fig.colorbar(h[3], ax=ax)
# plt.gca().invert_xaxis()

# ax.minorticks_on()
# ax.ticklabel_format(style='sci', axis='y')
# ax.tick_params(which='major', direction='inout')
# ax.tick_params(which='both', right=True, top=True)

# ax.set_xlabel(r'$\alpha$ (deg)')
# ax.set_ylabel(r'$\delta$ (deg)')
# cbar.ax.set_ylabel('N')
# plt.tight_layout()
# plt.show()


#%%
################################
# ABSOLUTE MAGNITUDE HISTOGRAM # (add redshift distribution?)
################################

# mag_lim_df = df[df['RMag'] <= -19.5]

# fig, ax = plt.subplots(figsize=(12,10), sharex=True, sharey=True)
# main_ax = plt.subplot2grid((3,3), (1,0), rowspan=3, colspan=2, fig=fig)
# hist_ax = plt.subplot2grid((3,3), (1,2), rowspan=3, fig=fig, sharey=main_ax)
# z_ax = plt.subplot2grid((3,3), (0,0), colspan=2, fig=fig, sharex=main_ax)

# h = main_ax.hist2d(df['redshift'], df['RMag'], bins=(60,40), cmap='Blues')
# main_ax.axhline(-19.5, color='r', linestyle='--', label='Magnitude Cut')

# hist_ax.hist(df['RMag'], bins=40, orientation='horizontal', color='steelblue')
# hist_ax.axhline(-19.5, color='r', linestyle='--')
# # hist_ax.hist(mag_lim_df['RMag'], bins=40, orientation='horizontal', histtype='step', color='r')

# z_ax.hist(df['redshift'], bins=20, color='steelblue')
# z_ax.hist(mag_lim_df['redshift'], bins=20, histtype='step', color='k')

# main_ax.set_xlabel('Redshift')
# main_ax.set_ylabel(r'M$_R$ [mag]')

# hist_ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# hist_ax.get_yaxis().set_visible(False)
# hist_ax.set_xlabel(r'N [${10^3}$]')

# z_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# z_ax.get_xaxis().set_visible(False)
# z_ax.set_ylabel(r'N [$10^3$]')

# main_ax.minorticks_on()
# main_ax.tick_params(which='major', direction='inout')
# main_ax.tick_params(which='both', right=True, top=True)
# main_ax.invert_yaxis()

# plt.subplots_adjust(wspace= 0.1, hspace=0.1)

# handles, _ = main_ax.get_legend_handles_labels()
# main_ax.legend(frameon=False, labels=['Magnitude Cut'], handles=handles[1:], loc='lower right')
# plt.tight_layout()
# plt.show()


# No. of galaxies (N) distribution
# fig, ax = plt.subplots(figsize=(8,8))
# ax.hist(df['redshift'], bins=20)
# ax.set_xlabel('Redshift')
# ax.set_ylabel('Counts (x$\mathrm{10^3}$)')
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# plt.show()


# Magnitude comparison between redshifts
# df_first_bin = df[(df['redshift'] <= 1.0) & (df['redshift'] >= 0.5)]
# df_second_bin = df[(df['redshift'] <= 1.5) & (df['redshift'] > 1.0)]
# df_third_bin = df[(df['redshift'] <= 2.0) & (df['redshift'] >1.5)]
# df_fourth_bin = df[(df['redshift'] <= 2.5) & (df['redshift'] > 2.0)]
# df_fifth_bin = df[(df['redshift'] <= 3.0) & (df['redshift'] > 2.5)]

# fig, ax = plt.subplots(figsize=(10,8))
# n, bins, patches = ax.hist(df_first_bin['RMag'], bins=30, alpha=0.4, label='$0.5 \leq z \leq 1$')
# ax.hist(df_second_bin['RMag'], bins=30, alpha=0.4, label='$1 < z \leq 1.5$')
# ax.hist(df_third_bin['RMag'], bins=30, alpha=0.4, label='$1.5 < z \leq 2$')
# ax.hist(df_fourth_bin['RMag'], bins=30, alpha=0.4, label='$2 < z \leq 2.5$')
# ax.hist(df_fifth_bin['RMag'], bins=30, alpha=0.4, label='$2.5 < z \leq 3$')
# ax.axvline(-19.5, color='k', linestyle='--')
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(
#     lambda y, pos: '%.0f' % (y * 1e-3)))
# ax.set_ylabel('Counts (x$10^3$)')
# ax.set_xlabel('R Absolute Magnitude (mag)')
# ax.set_xlim(-28, -8)
# plt.legend(frameon=False)
# plt.show()


#%%
############################
# biweight scale estimator #
############################
# from astropy.stats import biweight_scale
# from analysis.methods import redshift_to_velocity

# cluster_vel_disp = np.array([c.vel_disp.value for c in clusters]) # to check against

# # get cluster galaxies velocity
# galaxies_z = np.array([c.galaxies[:,2] for c in clusters], dtype=object)

# biweight_vel_disp = np.zeros(len(cluster_vel_disp))
# for i,c in enumerate(galaxies_z): # for each row
#     galaxies_vel = redshift_to_velocity(c, np.mean(c))

#     # calculate velocity dispersion for each cluster
#     galaxies_vel_disp = biweight_scale(galaxies_vel).value
#     biweight_vel_disp[i] = galaxies_vel_disp

# masses = np.array([c.cluster_mass.value for c in clusters])

# # compare biweight vs standard
# # plot biweight vs virial mass
# fig, ax = plt.subplots(figsize=(8,10), sharex=True)
# main_ax = plt.subplot2grid((3,1),(1,0), fig=fig, rowspan=2)
# res_ax = plt.subplot2grid((3,1),(0,0), fig=fig, sharex=main_ax)

# biweight_vel_disp = biweight_vel_disp/1000
# main_ax.scatter(np.log10(masses), biweight_vel_disp, s=5, alpha=0.7, color='steelblue')

# def func(x,a,b):
#     return a*x+b

# popt, pcov = curve_fit(func, np.log10(masses), biweight_vel_disp)
# perr = np.sqrt(np.diag(pcov))

# residuals = biweight_vel_disp - func(np.log10(masses), *popt)
# ss_res = sum(residuals**2)
# ss_total = sum((biweight_vel_disp - np.mean(biweight_vel_disp))**2)
# r2 = 1-(ss_res/ss_total)
# print(popt, perr, r2)

# X = np.linspace(np.min(main_ax.get_xlim()),15.6)
# main_ax.plot(X, func(X, *popt), 'r--', label=f'y = {popt[0]:.2f}x + {popt[1]:.2f}')

# difference = biweight_vel_disp/(np.sqrt(cluster_vel_disp)/1000)
# res_ax.scatter(np.log10(masses), difference, s=5, alpha=0.7, color='steelblue')
# res_ax.axhline(1, color='red', ls='--')

# main_ax.minorticks_on()
# main_ax.tick_params(which='major', direction='inout')
# main_ax.tick_params(which='both', right=True, top=True)
# # main_ax.ticklabel_format(style='sci')

# res_ax.set_ylabel(r'$\sigma_{v,bi}\, / \, \sigma_{v}$')
# # res_ax.set_ylim(1-abs(max(difference)), 1+abs(max(difference)))
# res_ax.get_xaxis().set_visible(False)

# main_ax.set_xlabel(r'$M_V$ log[$h^{-1}\,$ M/M$_{\odot}$]')
# main_ax.set_ylabel(r'$\sigma_{v,bi}$ [$10^3$ km$\,$s$^{-1}$]')
# plt.subplots_adjust(hspace= 0.05)

# main_ax.legend()
# plt.tight_layout()
# plt.show()


#%%
#################################
# VELOCITY DISPERSION VS RADIUS # (STILL WIP)
#################################



#%%
##########################
# PROPERTIES CORRELATION #
##########################
# clusters = np.array([c for c in virial_clusters if c.richness>=10])
# c_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.projected_radius.value, c.total_absmag, c.richness, c.bcg_absMag] for c in clusters])

# fig, ax = plt.subplots(figsize=(10,8))
# ax1 = plt.subplot2grid((1,2),(0,0), fig=fig)
# ax2 = plt.subplot2grid((1,2),(0,1), fig=fig, sharey=ax1)

# redshift distribution
# c_z = np.array([c.z for c in clusters])
# ax.hist(c_params[:,-3], bins=20, histtype='stepfilled', color='steelblue')
# ax.scatter(c_z, (c_params[:,3]), s=5, alpha=0.75)
# ax.set_xlim(0.5,2.5)

# scatter plots
# ax.scatter(c_params[:,1], (c_params[:,-3]), s=8, alpha=0.5, color='steelblue')
# ax2.scatter(c_params[:,-3], np.log10(c_params[:,0]), s=8, alpha=0.5, color='steelblue')
# ax.scatter(np.log10(c_params[:,0]), np.log10(c_params[:,3]), s=8, alpha=0.5, color='tab:blue')
# ax.invert_yaxis()

# def func(x,a,b):
#     return a*x+b

# popt, pcov = curve_fit(func, c_params[:,-2], c_params[:,-3])
# perr = np.sqrt(np.diag(pcov))

# residuals = c_params[:,-3] - func(c_params[:,-2], *popt)
# ss_res = sum(residuals**2)
# ss_total = sum((c_params[:,-3] - np.mean(c_params[:,-3]))**2)
# r2 = 1-(ss_res/ss_total)
# print(popt, perr, r2)

# X = np.linspace(np.min(ax.get_xlim()),np.max(ax.get_xlim()))
# ax.plot(X, func(X, *popt), 'r--', label=f'y = {popt[0]:.2f}x + {popt[1]:.1f}')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.tick_params(which='both', right=True, top=True)
# ax.ticklabel_format(style='sci')

# # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.1f' % (y * 1e-6)))
# ax.set_ylabel('N')
# # ax.set_xlabel('Richness')
# ax.set_xlabel(r'$M_R$ [mag]')
# # ax.set_ylabel(r'$M_V$, log($h^{-1}\,$ M/M$_{\odot}$)')
# # ax.get_yaxis().set_visible(False)

# plt.legend(frameon=False)
# plt.tight_layout()
# plt.show()


# z_err
# galaxies_arr = np.loadtxt('FoF\\analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 

# lower_diff = galaxies_arr[:,2] - galaxies_arr[:,3]
# upper_diff = abs(galaxies_arr[:,2] - galaxies_arr[:,4])

# # element-wise max value
# z_err = np.minimum(lower_diff, upper_diff)

# fig, ax = plt.subplots(figsize=(10,8))
# plt.scatter(galaxies_arr[:,2], z_err, s=5, alpha=0.7, color='steelblue')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.tick_params(which='both', right=True, top=True)

# ax.set_ylabel('$\sigma_z$')
# ax.set_xlabel('z')

# plt.tight_layout()
# plt.show()

#%%
# --------- number density of clusters against redshift
# from scipy.integrate import quad
# from scipy.optimize import curve_fit
# import matplotlib.ticker as ticker

# galaxies_arr = np.loadtxt('FoF\\analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 
# clusters = np.array(clusters)
# n_density = []

# bin clusters into redshift bins dz
# n = 10 # number of bins
# z_arr = np.array([c.z for c in clusters])
# z_arr = galaxies_arr[:,2]
# bins = np.linspace(min(z_arr), max(z_arr), n)
# digitized = np.digitize(z_arr, bins)

# for i in range(1, len(bins)):
#     bin_clusters = galaxies_arr[np.where(digitized==i)]
#     N = len(bin_clusters) # number of clusters in redshift bin, dN/dz
    
#     # number_density = N/V where V is differential comoving volume dv/dz
#     temp = lambda x: cosmo.differential_comoving_volume(x).value
#     solid_angle = (np.pi*(1.8*u.deg)**2).to(u.sr)
#     V = ((quad(temp, bins[i-1], bins[i])[0])*u.Mpc**3/u.sr)*solid_angle

#     # dv_dz = cosmo.differential_comoving_volume(bins[i-1])*(1.51*u.deg)**2
#     ndensity = (N/V).to(u.Mpc**-3)
#     n_density.append(ndensity.value)


# fig, ax = plt.subplots(figsize=(8,8))

# X_data = [(bins[i]+bins[i+1])/2 for i in range(0,len(bins)-1)]

# ax.scatter(X_data, n_density, color='steelblue', alpha=0.9, s=10)

# # curve fit
# def func(x,a,b,c,d):
#     # return a*np.exp(-b*x)+c
#     return a*x**3+b*x**2+c*x+d
# popt, pcov = curve_fit(func, X_data, n_density)

# xdata = np.linspace(min(X_data), max(z_arr), 100)
# ax.plot(xdata, func(xdata, *popt), 'r--')
# ax.set_ylim(max(n_density)/10, max(n_density)*2)

# ax.set_xlim(0.5,2.5)
# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.tick_params(which='both', right=True, top=True)
# ax.ticklabel_format(style='sci')
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.1f' % (y * 1e4)))
# ax.set_ylabel(r'galaxy number density [$10^{-4}\,$ ($h^{-1}\,$ Mpc)$^{3}$]')
# ax.set_xlabel('z')
# plt.tight_layout()
# plt.show()

#%%
def compare_masses(projected, virial):
    virial = np.array([[c.cluster_mass.value,c.mass_err.value] for c in virial])
    projected = np.array([[c.cluster_mass.value, c.mass_err.value] for c in projected])

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x=np.log10(virial[:,0]), y=np.log10(projected[:,0]), s=5, alpha=0.75)

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')

    def func(x,a,b):
        return a*x+b

    popt, pcov = curve_fit(func, np.log10(virial[:,0]), np.log10(projected[:,0]))
    perr = np.sqrt(np.diag(pcov))

    residuals = func(np.log10(virial[:,0]), *popt) - np.log10(projected[:,0])
    ss_res = sum(residuals**2)
    ss_total = sum((np.log10(projected[:,0]) - np.mean(np.log10(projected[:,0])))**2)
    r2 = 1-(ss_res/ss_total)
    print(popt, perr, r2)

    X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()))
    ax.plot(X, func(X, *popt), 'r--', label=f'y = {popt[0]:.2f}x + {popt[1]:.2f}')
    ax.set_aspect('equal')

    ax.minorticks_on()
    ax.tick_params(which='major', direction='inout')
    ax.tick_params(which='both', right=True, top=True)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('$M_V$ log($h^{-1}\,$ M/M$_{\odot}$)')
    ax.set_ylabel('$M_{PM}$ log($h^{-1}\,$ M/M$_{\odot}$)')
    
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


#%%
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from helper import lowess

def plot_masses(clusters, x, y): # adapt this into plot_quantities, with y vs x
    
    cluster_arr = np.array([[c.z, c.cluster_mass.value, c.mass_err.value] for c in clusters])
    
    z_arr = cluster_arr[:,0]
    z_err = np.array([c.z_unc for c in clusters])
    z_err = abs(z_err-z_arr[:,np.newaxis])

    masses = cluster_arr[:,1]
    mass_err = cluster_arr[:,2]
    log_mass_err = 0.434*mass_err/masses

    fig, ax = plt.subplots(figsize=(10,8))

    # raw points
    ax.scatter(z_arr, np.log10(masses), s=5, alpha=0.9, label='Virial Mass', color='steelblue')

    # LOWESS fit
    yest = lowess(z_arr, np.log10(masses), iter=10)
    z_arr, yest = zip(*sorted(zip(z_arr,yest), key=lambda x:x[0]))
    ax.plot(z_arr, yest, 'r--', label='LOWESS Fit')

    # bootstrap data
    with NumpyRNGContext(1):
        bootresult = bootstrap(cluster_arr[:,:2], bootnum=1000)
    boot_yest = np.zeros(bootresult.shape)
    for i,b in enumerate(bootresult):
        y = lowess(b[:,0], np.log10(b[:,1]))
        b[:,0], y = zip(*sorted(zip(b[:,0],y), key=lambda x:x[0]))
        boot_yest[i,:,0] = b[:,0]
        boot_yest[i,:,1] = y

    # confidence interval
    yest_high = np.percentile(boot_yest, q=95, axis=0, keepdims=True)
    yest_low = np.percentile(boot_yest, q=5, axis=0, keepdims=True)
    f = ax.fill_between(z_arr, yest_low[0][:,1], yest_high[0][:,1], color='grey', alpha=0.5)
    
    ax.set_xlabel('z')
    ax.set_ylabel(r'M$_V$, log($h^{-1}\,$ M/M$_{\odot}$)')
    ax.minorticks_on()
    ax.ticklabel_format(style='sci', axis='y')
    ax.tick_params(which='major', direction='inout')
    ax.tick_params(which='both', right=True, top=True)
    # ax.yaxis.major.formatter._useMathText = True

    ax.set_xlim(0.5,2.5)
    # ax.set_ylim(14.6,15.8)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


