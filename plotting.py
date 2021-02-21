import pickle
import sqlite3
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import optimize
from scipy.stats import linregress

from params import *

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True

checking = True # change to False if finalizing plots

if not checking:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)


# with open(fname+'clusters.dat', 'rb') as f:
#     virial_clusters = pickle.load(f)


# -- plotting clusters for manual checking (do not use, find somewhere else to put this)
def check_plots(clusters):

    coords = np.array([[c.ra, c.dec] for c in clusters])
    clusters = np.array(clusters)
    z_arr = np.array([c.z for c in clusters])        

    bins = np.linspace(0.5,2.53,83) # bins of 0.03 width
    # bins = np.arange(0.5,2.53,0.00666)
    digitized = np.digitize(z_arr, bins)

    for i in range(1,len(bins)):
        binned_data = clusters[np.where(digitized==i)]
        binned_data = sorted(binned_data, key=lambda x: x.ra)
        logging.info(f'Number of clusters in bin {i}: {len(binned_data)}')

        if len(binned_data): # plot clusters for checking
            fig = plt.figure(figsize=(10,8))
            logging.info(f'Plotting bin {i}. Clusters with binned redshift {bins[i]}')
            # plt.hist2d(x=arr[:,0], y=arr[:,1], bins=(100,80), cmap=plt.cm.Reds)

            for center in binned_data:
                plt.scatter(center.galaxies[:,0], center.galaxies[:,1], s=5)
                plt.scatter(center.ra, center.dec, s=10)
                plt.axis([min(coords[:,0]), max(coords[:,0]), min(coords[:,1]), max(coords[:,1])])
                logging.info('Plotting ' + center.__str__())
            plt.show()


# check_plots(virial_clusters)


#%%
##########################
# RA VS DEC DENSITY PLOT #
##########################
# df = pd.read_csv('datasets\\cosmos2015_dataset.csv')
# columns = ['ra', 'dec', 'ID', 'zphot', 'redshift', 'z_lower', 'z_upper+', 'RMag', 'Class', 'log_Stellar_Mass', 'log_SFR', 'LR'] # renaming columns
# df.columns = columns
# df = df[(df['redshift'] <= 2.5)] 
# df = df[df['RMag'] >= -50]

# fig, ax = plt.subplots(figsize=(10,8))
# h = ax.hist2d(df['ra'], df['dec'], bins=(100,80))
# cbar = fig.colorbar(h[3], ax=ax)

# ax.set_xlabel('Right Ascension (deg)')
# ax.set_ylabel('Declination (deg)')
# cbar.ax.set_ylabel('Counts')
# plt.show()

#%%
################################
# ABSOLUTE MAGNITUDE HISTOGRAM # (add redshift distribution?)
################################

# fig, ax = plt.subplots(figsize=(12,8), sharey=True)
# main_ax = plt.subplot2grid((3,3), (0,0), rowspan=3, colspan=2, fig=fig)
# hist_ax = plt.subplot2grid((3,3), (0,2), rowspan=3, fig=fig, sharey=main_ax)

# h = main_ax.hist2d(mag_lim_df['redshift'], mag_lim_df['RMag'], bins=(60,40), cmap='Blues')
# # # cbar = fig.colorbar(h[3], ax=ax)
# main_ax.axhline(-19.5, color='r', linestyle='--')
# hist_ax.hist(mag_lim_df['RMag'], bins=40, orientation='horizontal')
# hist_ax.axhline(-19.5, color='r', linestyle='--')

# plt.gca().invert_yaxis()
# # cbar.ax.set_ylabel('Counts')
# main_ax.set_xlabel('Redshift')
# main_ax.set_ylabel('$\mathrm{M_R}$')
# hist_ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# hist_ax.get_yaxis().set_visible(False)
# hist_ax.set_xlabel('Counts (x$\mathrm{10^3}$)')

# plt.subplots_adjust(wspace= 0.1)
# plt.show()


# No. of galaxies (N) distribution
# fig, ax = plt.subplots(figsize=(10,8))
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
####################
# MASS VS REDSHIFT #
####################

# fig, ax = plt.subplots(figsize=(12,8))
# p = ax.scatter(bcg_arr[:,2], np.log10(masses), s=8, color='tab:blue', alpha=0.75, label='Cluster Mass')

# ax.set_title('Estimated Virial Mass against Redshift')
# ax.set_xlabel('Redshift')
# ax.set_ylabel('log($\mathrm{M/M_\odot}$)')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.ticklabel_format(style='sci', axis='y')
# ax.yaxis.major.formatter._useMathText = True

# ax.set_xlim(0.5,2.5)
# ax.set_ylim(15.8, 16.8)
# # ax.legend(frameon=False)
# plt.show()

#%%
#################################
# VIRIAL MASS VS PROJECTED MASS #
#################################

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(x=np.log10(virial), y=np.log10(projected), s=10, alpha=0.75)
# lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
#         np.max([ax.get_xlim(), ax.get_ylim()])]

# m, c, r, _, _, = linregress(np.log10(virial), np.log10(projected))
# print(m,c,r)
# ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')
# X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()))
# ax.plot(X, m*X+c, 'r--', label='y = {m}x + {c}, $R^2={r2}$'.format(m=round(m,3), c=round(c,2), r2=round(r,3)))
# ax.set_aspect('equal')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e1)))
# # ax.ticklabel_format(style='sci')

# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_xlabel('Virial mass log($\mathrm{M/M_\odot}$)')
# ax.set_ylabel('Projected mass log($\mathrm{M/M_\odot}$)')
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.legend(frameon=False)

# plt.show()

#%%
##################################
# RICHNESS VS ABSOLUTE MAGNITUDE #
##################################

# R_arr = np.array([c.richness for c in virial_clusters])

# def get_lum(c):
#     member_galaxies = c.galaxies
#     lum = member_galaxies[member_galaxies[:,3].argsort()][2,3]
#     return lum

# lum_arr = np.array([get_lum(c) for c in virial_clusters])

# X = np.linspace(min(R_arr),max(R_arr),100)
# m2,c2,r2,_,_ = linregress(R_arr, lum_arr)
# print(m2, c2, r2)

# fig, ax = plt.subplots(figsize=(10,8))
# plt.gca().invert_yaxis()

# ax.scatter(R_arr, lum_arr, s=8, alpha=0.5, color='tab:blue')
# ax.plot(X, m2*X+c2, 'r--', label='Regression Best Fit, $y = {m}x+{c}, R^2 = {r2}$'.format(m=round(m2,4), c=round(c2,1), r2=round(r2,3)))

# ax.set_xlabel('Richness R')
# ax.set_ylabel('2nd Brightest Galaxy Absolute Magnitude (mag)')

# plt.legend(frameon=False)
# plt.show()


#%%
#################################
# VELOCITY DISPERSION VS RADIUS # (STILL WIP)
#################################

# from FoF_algorithm import linear_to_angular_dist
# from mass_estimator import redshift_to_velocity
# from astropy.cosmology import LambdaCDM
# from astropy.coordinates import SkyCoord
# import astropy.units as u

# cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology

# # bcg_df['Doppler_vel'] = redshift_to_velocity(bcg_df['redshift']).to('km/s').value
# member_df['Doppler_vel'] = redshift_to_velocity(member_df['redshift'])/(1+)
# member_df = member_df.sort_values('cluster_id'.values)

# arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)

# for g in group_n:
#     center = bcg_arr[bcg_arr[:,-1]==g]
#     cluster_members = arr[arr[:,-1]==g]

#     cluster_members_coord = SkyCoord(ra=cluster_members[:,0]*u.degree, dec=cluster_members[:,1]*u.degree)
#     cluster_center_coord = SkyCoord(ra=center[:,0]*u.degree, dec=center[:,1]*u.degree)
#     separation = cluster_center_coord.separation(cluster_members_coord)
#     distance = (cosmo.angular_diameter_distance(z=center[2]) * separation).to(u.Mpc, u.dimensionless_angles())

#     cluster_velocities = cluster_members[:,-1]-center[:,-1]
#     mean_cluster_velocity = np.mean(cluster_velocities)
#     # normalized_velocity = velocities/mean_cluster_velocity
#     velocity_dispersion = sum((velocities-mean_cluster_velocity)**2)/(len(velocities)-1)

#     # form velocity dispersion profile for radius
#     n = 10
#     bins = np.linspace(min(distance), max(distance), n)
#     digitized = np.digitize(distance, bins)
#     dispersion_curve = []

#     for i in range(1,len(bins)+1):
#         bin_galaxies = cluster_velocities[np.where(digitized==i)]
#         if len(bin_galaxies) > 1:
#             mean_bin_velocity = np.mean(bin_galaxies)
#             bin_dispersion = sum((bin_galaxies-mean_bin_velocity)**2)/(len(bin_galaxies)-1)
#         dispersion_curve.append(np.sqrt(bin_dispersion))
#     dispersion_curve = np.array(dispersion_curve)
#     assert len(dispersion_curve) == n

# #     # plotting 2 plots
#     fig, ax = plt.subplots(figsize=(8,12), sharex=True)
# #     bottom_ax = plt.subplot2grid((2,1), (1,0), fig=fig)
# #     top_ax = plt.subplot2grid((2,1), (0,0), fig=fig, sharex=bottom_ax)

#     ax.scatter(distance.value, cluster_velocities/1000, s=5)

#     R_binned = np.linspace(0, max(distance.value), n)
#     # ax.plot(R_binned, (mean_cluster_velocity + 3*dispersion_curve)/1000, '-')
# #     bottom_ax.plot(R_binned, (mean_cluster_velocity - 3*dispersion_curve)/1000, '-')
# #     top_ax.plot(R_binned, dispersion_curve/1000, '.')

#     ax.set_xlabel('R (Mpc)')
#     ax.set_ylabel('v (x10^3 km/s)')

# #     top_ax.set_ylabel('sigma (x10^3 km/s)')
# #     top_ax.set_ylim(0.2,None)
# #     top_ax.get_xaxis().set_visible(False)
# #     plt.subplots_adjust(hspace=0.1)

#     plt.show()

#%%
#################################
# REDSHIFT EVOLUTION # (STILL WIP)
#################################

# parameters to compare:
    # mass vs vel_disp: strong linear relationship
    # mass vs lum4: high scatter
    # mass vs abs mag: 
    # mass vs richness: scatter
    # mass vs radius: linear relationship

    # vel_disp vs radius: scatter
    # vel disp vs richness: scatter
    # vel_disp vs lum4: scatter

    # lum4 vs radius: slight linear relationship
    # richness vs radius: weird power trend?
    # lum4 vs richness: new clusters linear relationship

# globally, old clusters have higher mass, higher radius, lower richness (less galaxies at high z), higher luminosity (malmquist bias)

# old_clusters = [c for c in virial_clusters if c.z > 1.5]
# new_clusters = [c for c in virial_clusters if c.z <= 1.5]

# new_c_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.virial_radius.value, c.total_luminosity, c.richness, c.bcg_brightness] for c in new_clusters])
# old_c_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.virial_radius.value, c.total_luminosity, c.richness, c.bcg_brightness] for c in old_clusters])

# fig, ax = plt.subplots(figsize=(10,8))
# ax.scatter(new_c_params[:,-2], new_c_params[:,-3], s=8, alpha=0.5, color='tab:blue', label='Young')
# ax.scatter(old_c_params[:,-2], old_c_params[:,-3], s=8, alpha=0.5, color='tab:red', label='Old')

# ax.plot(X, m2*X+c2, 'r--', label='Regression Best Fit, $y = {m}x+{c}, R^2 = {r2}$'.format(m=round(m2,4), c=round(c2,1), r2=round(r2,3)))

# ax.set_xlabel('')
# ax.set_ylabel('')

# plt.legend(frameon=False)
# plt.show()
