import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.cosmology import WMAP5 as cosmo
from scipy.spatial import cKDTree
from FoF_algorithm import linear_to_angular_dist, redshift_to_velocity
from mass_estimator import virial_mass_estimator

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s -  %(levelname)s -  %(message)s")

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
bcg_df = pd.read_csv('derived_datasets\\filtered_bcg.csv')
member_galaxy_df = pd.read_csv('derived_datasets\\filtered_members.csv')

bcg_arr = bcg_df.values
# arr, group_n = split_df_into_groups(member_galaxy_df, 'cluster_id')


# ---- import catalogue table
conn = sqlite3.connect('galaxy_clusters.db')

deep_field_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, R
    FROM deep_field 
    WHERE Ngal>=20
    ORDER BY redshift
    ''', conn)
deep_field_arr = deep_field_df.values

cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM cosmic_web_bcg
    WHERE Ngal>=20
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
    catalogue_richness = np.zeros(len(candidates))
    candidate_richness = np.zeros(len(candidates))

    min_redshift = min(candidates[:,2])
    max_redshift = max(catalogue[:,2])
    logging.debug('Minimum redshift: {min}; Maximum redshift: {max}'.format(min=min_redshift, max=max_redshift))
    
    candidates = candidates[(candidates[:,2] >= 0.5)&(candidates[:,2]<= max_redshift)]
    catalogue = catalogue[(catalogue[:,2] >= 0.5)&(catalogue[:,2]<= max_redshift)]
    logging.debug('No. of candidates: {n_cand} and No. of catalogues: {n_catalogue}'.format(n_cand=len(candidates), n_catalogue=len(catalogue)))

    can = SkyCoord(candidates[:,0]*u.degree, candidates[:,1]*u.degree)

    for i, c in enumerate(catalogue): # for each cluster in the catalogue, find candidate clusters that are within 0.05z and 0.1Mpc

        redshift_constraint = 0.05
        redshift_cut = can[abs(candidates[:,2] - c[2]) <= redshift_constraint]
        candidates_rich = candidates[abs(candidates[:,2] - c[2]) <= redshift_constraint][:,5]
        max_sep = linear_to_angular_dist(0.5, c[2])
        c = SkyCoord(c[0]*u.degree, c[1]*u.degree)

        if len(redshift_cut):
            idx, d2d, _ = c.match_to_catalog_sky(redshift_cut)
            if d2d < max_sep:
                can_matches = redshift_cut[idx]
                if (can_matches):
                    count += 1
                    catalogue_richness[i] = catalogue[i,3]
                    candidate_richness[i] = candidates_rich[idx]

    print('{n}/{total} matched. {number} % of candidates matched'.format(n=count, total=len(catalogue), number=(count/len(catalogue))*100))

    fig = plt.figure()
    plt.scatter(candidate_richness, catalogue_richness)
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.show() 

compare_clusters(bcg_arr, deep_field_arr)