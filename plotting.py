import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from scipy.stats import linregress
from scipy import optimize

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)

# import data from DB

# RA against DEC



# --------------------------- Magnitude Histogram ----------------------------
# df = pd.read_csv('datasets\\cosmos2015_dataset.csv')
# columns = ['ra', 'dec', 'ID', 'zphot', 'redshift', 'z_lower', 'z_upper+', 'RMag', 'Class', 'log_Stellar_Mass', 'log_SFR', 'LR'] # renaming columns
# df.columns = columns

# df = df[(df['redshift'] <= 2.5)] # select for redshift 0<z<2
# mag_lim_df = df[df['RMag'] >= -50]

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



# Redshift distribution

# No. of galaxies (N) distribution




# ---------------------- Mass against redshift -----------------------------
# bcg_df = pd.read_csv('derived_datasets\\filtered_bcg.csv')
# bcg_arr = bcg_df.sort_values('cluster_id').values
# masses = np.loadtxt('derived_datasets\\test_virial_masses.txt')

# k, = np.where(bcg_arr[:,2] <= 2.5)
# bcg_arr = bcg_arr[k,:]
# masses = masses[k]

# bins = np.arange(0.5,2.0+0.2,0.2)
# digitized = np.digitize(bcg_arr[:,2], bins, right=True)
# mass_curve = []

# for i in range(1,len(bins)+1):
#         bin_mass = masses[np.where(digitized==i)]
#         if len(bin_mass) > 1:
#                 median_bin_mass = np.mean(bin_mass)
#                 mass_curve.append(median_bin_mass)
#         elif len(bin_mass) == 1:
#                 mass_curve.append(bin_mass[0])
#         else:
#                 mass_curve.append(0)

# mass_curve = np.array(mass_curve)
# assert len(mass_curve) == len(bins)

# fig, ax = plt.subplots(figsize=(12,8))
# p = ax.scatter(bcg_arr[:,2], np.log10(masses), s=8, alpha=0.75, label='Cluster Mass')
# ax.scatter(bcg_arr[:,2], np.log10(corrected_mass), s=8, alpha=0.75, label='Corrected Mass')
# ax.plot(bins, np.log10(mass_curve), 'r--', label='Median log($\mathrm{M/M_\odot}$)')
# cbar = fig.colorbar(p)


# ax.set_title('Mass against redshift')
# ax.set_xlabel('Redshift')
# ax.set_ylabel('log($\mathrm{M/M_\odot}$)')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.ticklabel_format(style='sci', axis='y')
# ax.yaxis.major.formatter._useMathText = True

# ax.set_xlim(0.5,2.0)
# ax.set_ylim(14.5,15.8)
# ax.legend(frameon=False)
# plt.show()



# ------------------ Virial mass against projected mass --------------------------------
# projected = np.loadtxt('derived_datasets\\filtered_projected_masses.txt')
# virial = np.loadtxt('derived_datasets\\filtered_virial_masses.txt')
# virial = np.loadtxt('corrected_masses.txt')

# projected = projected[projected < 1e17]
# virial = virial[virial < 1e17]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(x=np.log10(virial), y=np.log10(projected), s=10, alpha=0.75)
# lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
#         np.max([ax.get_xlim(), ax.get_ylim()])]

# m, c, r, _, _, = linregress(np.log10(virial), np.log10(projected))
# print(m,c,r)
# ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')
# X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()))
# ax.plot(X, m*X+c, 'r--', label='y = {m}x + {c}'.format(m=round(m,2), c=round(c,2)))
# ax.set_aspect('equal')

# ax.minorticks_on()
# ax.tick_params(which='major', direction='inout')
# ax.ticklabel_format(style='sci', axis='y')

# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_xlabel('Virial mass log($\mathrm{M/M_\odot}$)')
# ax.set_ylabel('Projected mass log($\mathrm{M/M_\odot}$)')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()

# plt.show()


# --------- velocity against radius -----------------
# from FoF_algorithm import linear_to_angular_dist
# from mass_estimator import redshift_to_velocity
from data_processing import split_df_into_groups
# from astropy.cosmology import WMAP5 as cosmo
# from astropy.coordinates import SkyCoord
# import astropy.units as u

# bcg_df = pd.read_csv('derived_datasets\\filtered_bcg.csv')
# member_df = pd.read_csv('derived_datasets\\filtered_members.csv')

# bcg_df['Doppler_vel'] = redshift_to_velocity(bcg_df['redshift']).to('km/s').value
# member_df['Doppler_vel'] = redshift_to_velocity(member_df['redshift']).to('km/s').value

# bcg_arr = bcg_df.values
# arr, group_n = split_df_into_groups(member_df, 'cluster_id', -2)

# for g in group_n:
#     center = bcg_arr[bcg_arr[:,-2]==g]
#     cluster_members = arr[arr[:,-2]==g]

#     cluster_member_points = SkyCoord(ra=cluster_members[:,0]*u.degree, dec=cluster_members[:,1]*u.degree)
#     center_point = SkyCoord(ra=center[:,0]*u.degree, dec=center[:,1]*u.degree)
#     separation = center_point.separation(cluster_member_points)
#     distance = (cosmo.angular_diameter_distance(z=cluster_members[:,2]) * separation).to(u.Mpc, u.dimensionless_angles())

#     cluster_velocities = cluster_members[:,-1]-center[:,-1]
#     mean_cluster_velocity = np.mean(cluster_velocities)
#     # normalized_velocity = velocities/mean_cluster_velocity
#     # velocity_dispersion = sum((velocities-mean_cluster_velocity)**2)/(len(velocities)-1)

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

#     # plotting 2 plots
#     fig, ax = plt.subplots(figsize=(8,12), sharex=True)
#     bottom_ax = plt.subplot2grid((2,1), (1,0), fig=fig)
#     top_ax = plt.subplot2grid((2,1), (0,0), fig=fig, sharex=bottom_ax)

#     bottom_ax.scatter(distance.value, cluster_velocities/1000, s=5)

#     R_binned = np.linspace(0, max(distance.value), n)
#     bottom_ax.plot(R_binned, (mean_cluster_velocity + 3*dispersion_curve)/1000, '-')
#     bottom_ax.plot(R_binned, (mean_cluster_velocity - 3*dispersion_curve)/1000, '-')
#     top_ax.plot(R_binned, dispersion_curve/1000, '.')

#     bottom_ax.set_xlabel('R (Mpc)')
#     bottom_ax.set_ylabel('v (x10^3 km/s)')

#     top_ax.set_ylabel('sigma (x10^3 km/s)')
#     top_ax.set_ylim(0.2,None)
#     top_ax.get_xaxis().set_visible(False)
#     plt.subplots_adjust(hspace=0.1)

#     plt.show()


# ------ richness against mass
# bcg_df = pd.read_csv('derived_datasets\\filtered_bcg.csv').sort_values('cluster_id')
# member_df = pd.read_csv('derived_datasets\\filtered_members.csv')
# virial = np.loadtxt('derived_datasets\\test_virial_masses.txt')

# bcg_df = bcg_df[bcg_df['redshift'] <= 2.5]
# k, = np.where(bcg_df['redshift'] <= 2.5)
# # virial = virial[k]

# bcg_arr = bcg_df.values
# arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
# total_lum = np.zeros(len(group_n))

# for i, g in enumerate(group_n):
#     center = bcg_arr[bcg_arr[:,-1]==g]
#     cluster_members = arr[arr[:,-1]==g]
#     lum2 = cluster_members[cluster_members[:,3].argsort()][2,3]
#     total_lum[i] = lum2

# X = np.linspace(min(bcg_df['total_N']),max(bcg_df['total_N']),100)
# m,c,r,_,_ = linregress(bcg_df['total_N'], virial)

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(bcg_df['total_N'], virial, s=8)
# ax.plot(X, m*X+c, '--')
# plt.show()