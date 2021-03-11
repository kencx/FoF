#!/usr/bin/env python3

import sqlite3
from matplotlib import cm
import matplotlib.ticker as ticker

from scipy import optimize
from scipy.stats import linregress

from params import *

checking = True # change to False if finalizing plots

if not checking:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


with open(fname+'clusters.dat', 'rb') as f:
    virial_clusters = pickle.load(f)


# -- plotting clusters for manual checking (do not use, find somewhere else to put this)
def check_plots(clusters):

    coords = np.array([[c.ra, c.dec] for c in clusters])
    clusters = np.array(clusters)
    z_arr = np.array([c.z for c in clusters])        

    # bins = np.linspace(0.5,2.53,83) # bins of 0.03 width
    bins = np.arange(0.5,2.53,0.00666)
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

# print(len(virial_clusters))
# richness_arr = [c.richness for c in virial_clusters if c.flag_poor == False]
# z_arr = [c.z for c in virial_clusters if c.flag_poor == False]
# plt.figure()
# plt.scatter(z_arr, richness_arr, s=5, alpha=0.7)
# plt.hist(richness_arr, bins=30)
# plt.show()


#%%
##########################
# RA VS DEC DENSITY PLOT #
##########################
import pandas as pd
df = pd.read_csv('FoF\\processing\\datasets\\cosmos2015_dataset.csv')
columns = ['ra', 'dec', 'ID', 'zphot', 'redshift', 'z_lower', 'z_upper+', 'RMag', 'Class', 'log_Stellar_Mass', 'log_SFR', 'LR'] # renaming columns
df.columns = columns
df = df[(df['redshift'] <= 2.5)] 
df = df[df['RMag'] >= -50]

# fig, ax = plt.subplots(figsize=(10,8))
# h = ax.hist2d(df['ra'], df['dec'], bins=(100,80))
# cbar = fig.colorbar(h[3], ax=ax)
# plt.gca().invert_xaxis()
# # plt.tight_layout()

# ax.set_xlabel(r'$\mathrm{\alpha}$ (deg)')
# ax.set_ylabel('$\mathrm{\delta}$ (deg)')
# cbar.ax.set_ylabel('N')
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
# main_ax.axhline(-19.5, color='r', linestyle='--')

# hist_ax.hist(df['RMag'], bins=40, orientation='horizontal', color='tab:blue')
# hist_ax.axhline(-19.5, color='r', linestyle='--')
# # hist_ax.hist(mag_lim_df['RMag'], bins=40, orientation='horizontal', histtype='step', color='r')

# z_ax.hist(df['redshift'], bins=20, color='tab:blue')
# z_ax.hist(mag_lim_df['redshift'], bins=20, histtype='step', color='k')

# main_ax.set_xlabel('Redshift')
# main_ax.set_ylabel('$\mathrm{M_R}$ (mag)')

# hist_ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# hist_ax.get_yaxis().set_visible(False)
# hist_ax.set_xlabel('Counts (x$\mathrm{10^3}$)')

# z_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
# z_ax.get_xaxis().set_visible(False)
# z_ax.set_ylabel('Counts (x$\mathrm{10^3}$)')

# main_ax.invert_yaxis()
# plt.subplots_adjust(wspace= 0.1, hspace=0.1)
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
#################################
# VELOCITY DISPERSION VS RADIUS # (STILL WIP)
#################################


#%%
##########################
# PROPERTIES CORRELATION #
##########################

# clusters = [c for c in virial_clusters]
# c_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.virial_radius.value, c.total_luminosity, c.richness, c.bcg_absMag, c.r200(cosmo, 200).value] for c in clusters])

# fig, ax = plt.subplots(figsize=(10,8))

# redshift distribution
# c_z = np.array([c.z for c in clusters])
# plt.hist(c_z, bins=20, histtype='step')

# scatter plots
# ax.scatter(c_params[:,2], np.log10(c_params[:,3]), s=8, alpha=0.5, color='tab:blue')
# ax.scatter(np.log10(c_params[:,-3]), np.log10(c_params[:,0]), s=8, alpha=0.5, color='tab:blue')
# plt.gca().invert_xaxis()

# regression fits
# ax.plot(X, m2*X+c2, 'r--', label='Regression Best Fit, $y = {m}x+{c}, R^2 = {r2}$'.format(m=round(m2,4), c=round(c2,1), r2=round(r2,3)))

# ax.set_xlabel('')
# ax.set_ylabel('')

# plt.legend(frameon=False)
# plt.show()


#%%
# old_clusters = [c for c in virial_clusters if c.z > 1.5]
# new_clusters = [c for c in virial_clusters if c.z <= 1.5]

# old_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.virial_radius.value, c.total_luminosity, c.richness, c.bcg_absMag, c.r200(cosmo, 200).value] for c in old_clusters])
# new_params = np.array([[c.cluster_mass.value, c.vel_disp.value, c.virial_radius.value, c.total_luminosity, c.richness, c.bcg_absMag, c.r200(cosmo, 200).value] for c in new_clusters])

# # histograms
# plt.hist((old_params[:,-3]), bins=15, histtype='step', label='z > 1.5')
# plt.hist((new_params[:,-3]), bins=10, histtype='step', label='z < 1.5')

# plt.legend(frameon=False)
# plt.show()