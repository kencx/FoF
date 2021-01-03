import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.cosmology import WMAP5 as cosmo
from scipy.spatial import cKDTree
from FoF_algorithm import cluster_search, linear_to_angular_dist, redshift_to_velocity
from virial_mass_estimator import split_df_into_groups, virial_mass_estimator


'''
look at bookmarked catalog matching to improve accuracy of cluster_finder_checker? IMPORTANT
'''

# --- perform FoF on sample
# df = pd.read_csv('datasets/cosmos2015_filtered.csv')
# df = df.loc[:, ['ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR', 'NUMBER']]

# select for within redshift of catalogue, remove outliers
# df = df.loc[(df['PHOTOZ'] <= 0.990) & (df['PHOTOZ'] >= 0.100)]
# galaxy_arr = np.asarray(df)
# galaxy_arr = galaxy_arr[galaxy_arr[:,2].argsort()] # sort by redshift
# galaxy_arr = np.hstack((galaxy_arr, np.zeros((galaxy_arr.shape[0], 1)))) # add column for galaxy velocity
# galaxy_arr[:,-1] = redshift_to_velocity(galaxy_arr[:,2]).to('km/s').value
# luminous_galaxy_data = galaxy_arr[galaxy_arr[:,3] <= -20.5] # filter for galaxies brighter than -20.5

# --- cluster search 
# bcg_df, _ = cluster_search(galaxy_arr, luminous_galaxy_data, max_velocity=2000, linking_length_factor=0.4, virial_radius=2, check_map=False, export=True, fname='test')


# ---
bcg_df = pd.read_csv('0.1-0.9_bcg (interloper removed).csv')
member_galaxy_df = pd.read_csv('0.1-0.9_members (interloper removed).csv')

bcg_arr = bcg_df.sort_values('cluster_redshift').values
arr, group_n = split_df_into_groups(member_galaxy_df, 'cluster_id')

# total N histogram
# plt.hist(bcg_arr[:,5], bins='auto')
# plt.show()

# redshift vs total N
# plt.plot(bcg_arr[:,-1], bcg_arr[:,5], '.')
# plt.show()

# plt.plot(bcg_df['cluster_id'], bcg_df['redshift'], '.')
# plt.show()


# --- masses
# masses = np.zeros(group_n.shape)
# vel_dispersion = np.zeros(group_n.shape)

# for g in group_n:
#     cluster = arr[arr[:,-1]==g]
#     center = bcg_arr[int(g),:]
#     mass, vel_disp, rad = virial_mass_estimator(cluster[:,:3], center[-1])
#     masses[int(g)] = mass.value
#     vel_dispersion[int(g)] = vel_disp.value

# np.savetxt(fname='cluster_masses.txt', X=masses)

# masses = np.loadtxt(fname='cluster_masses.txt')
# # plt.hist(masses, bins='auto', histtype='step')
# plt.plot(masses, bcg_arr[:,5],'.')
# # plt.plot(bcg_arr[:,2], vel_dispersion, '.')
# # plt.yscale('log')
# plt.show()


# ---- import catalogue table
conn = sqlite3.connect('galaxy_clusters.db')

deep_field_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, R
    FROM deep_field 
    WHERE (redshift > 0.1) and (redshift < 0.990)
    ORDER BY redshift
    ''', conn)
deep_field_arr = deep_field_df.values

cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM cosmic_web
    WHERE (redshift > 0.1) and (redshift < 0.990)
    ORDER BY redshift
    ''', conn)
cosmic_web_arr = cosmic_web_df.values

# group_data = pd.read_csv('datasets/xray_group_catalog_tbl.csv')
# group_data = group_data.loc[:,['group_id', 'ra', 'dec', 'redshift', 'nmem']]
# group_arr = np.asarray(group_data.loc[:,['ra', 'dec', 'redshift', 'nmem']])
# group_arr = group_arr[group_arr[:,2].argsort()] # sort by redshift

conn.close()


def compare_clusters(candidates, catalogue):

    count = 0
    # catalogue_richness = catalogue[:,4]
    # candidate_richess = candidates[:,4]

    can = SkyCoord(candidates[:,0]*u.degree, candidates[:,1]*u.degree)
    catalog = SkyCoord(catalogue[:,0]*u.degree, catalogue[:,1]*u.degree)
    # max_sep = linear_to_angular_dist(0.5, candidates[:,2])
    d_A = cosmo.angular_diameter_distance(z=candidates[:,2])

    idx, d2d, _ = can.match_to_catalog_sky(catalog)
    actual_d2d = (d2d*d_A).to(u.Mpc, u.dimensionless_angles())

    sep_constraint = actual_d2d < 0.5*u.Mpc

    can_matches = candidates[sep_constraint]
    catalog_matches = catalogue[idx[sep_constraint]]

    # print(sum(can_matches[:,2]==catalog_matches[:,2]))

    print(len(np.array(can_matches))/len(can))

    # for i, c in enumerate(catalogue):

    #     redshift_constraint = 0.05
    #     redshift_cut = can[abs(candidates[:,2] - c[2]) <= redshift_constraint]
    #     max_sep = linear_to_angular_dist(0.1, c[2])

    #     if len(redshift_cut):
    #         idx, d2d, _ = redshift_cut.match_to_catalog_sky(catalog)
    #         d2d_min = min(d2d)
    #         sep_constraint = d2d < max_sep

    #         if len(sep_constraint):
    #             can_matches = redshift_cut[sep_constraint]
    #         # catalog_matches = catalog[idx[sep_constraint]]

    #             if (can_matches):
    #                 count += 1

    # print(str((count/len(catalogue))*100) + '% matched')

#     fig = plt.figure()
#     plt.scatter(candidate_richness, catalogue_richness)
#     plt.show() 

compare_clusters(bcg_arr, cosmic_web_arr)