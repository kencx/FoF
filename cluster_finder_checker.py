import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fof_kdtree import cluster_search, linear_to_angular_dist, redshift_to_velocity

def sort_by_redshift(arr):
    return arr[arr[:,2].argsort()]

def df_to_arr_sorted(df):
    return sort_by_redshift(np.asarray(df))


# --- perform FoF on sample
# df = pd.read_csv('datasets/cosmos2015_filtered.csv')
# df = df.loc[:, ['ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR', 'NUMBER']]

# # select for within redshift of catalogue, remove outliers
# df = df.loc[(df['PHOTOZ'] <= 0.990) & (df['PHOTOZ'] >= 0.100)]
# galaxy_arr = np.asarray(df)
# galaxy_arr = galaxy_arr[galaxy_arr[:,2].argsort()] # sort by redshift
# galaxy_arr = np.hstack((galaxy_arr, np.zeros((galaxy_arr.shape[0], 1)))) # add column for galaxy velocity
# galaxy_arr[:,-1] = redshift_to_velocity(galaxy_arr[:,2]).to('km/s').value
# luminous_galaxy_data = galaxy_arr[galaxy_arr[:,3] <= -20.5] # filter for galaxies brighter than -20.5

# cluster search 
# bcg_df, _ = cluster_search(galaxy_arr, luminous_galaxy_data, max_velocity=2000, linking_length_factor=0.4, virial_radius=2, check_map=False, export=True, fname='')
bcg_df = pd.read_csv('bcg_df.csv')
bcg_arr = np.asarray(bcg_df.loc[:,['ra','dec','redshift', 'cluster_redshift', 'total_N']])
bcg_arr = bcg_arr[bcg_arr[:,2].argsort()] # sort by redshift


# import catalogue table
conn = sqlite3.connect('galaxy_clusters.db')

deep_field_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, R, cluster_id
    FROM deep_field 
    WHERE (redshift > 0.1) and (redshift < 0.990)
    ''', conn)
deep_field_arr = df_to_arr_sorted(deep_field_df)

cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM cosmic_web
    WHERE (redshift > 0.1) and (redshift < 0.990)
    ''', conn)
cosmic_web_arr = df_to_arr_sorted(cosmic_web_df)


# group_data = pd.read_csv('datasets/xray_group_catalog_tbl.csv')
# group_data = group_data.loc[:,['group_id', 'ra', 'dec', 'redshift', 'nmem']]
# group_arr = np.asarray(group_data.loc[:,['ra', 'dec', 'redshift', 'nmem']])
# group_arr = group_arr[group_arr[:,2].argsort()] # sort by redshift


def compare_clusters(candidates, catalogue):

    count = 0
    catalogue_richness = catalogue[:,4]
    candidate_richness = np.zeros((catalogue_richness.shape))

    for i, c in enumerate(catalogue):
        
        # search a radius of 1Mpc and within redshift constraint
        redshift_constraint = 0.05
        redshift_cutoff = candidates[abs(candidates[:,3] - c[2]) <= redshift_constraint]

        if len(redshift_cutoff):
            candidate_tree = cKDTree(redshift_cutoff[:,:2]) # create binary search tree for candidate sample
            centers_idx = candidate_tree.query_ball_point(c[:2], linear_to_angular_dist(0.5, c[2]).value)
            centers = redshift_cutoff[centers_idx]
            if len(centers) >= 1:
                count += 1 # add to count if constraints satisfied
                sort_richness = centers[(centers[:,-1]).argsort()]
                candidate_richness[i] = sort_richness[0,-1]
    
    print((count/len(catalogue))*100) # good algorithm if >80%

    fig = plt.figure()
    plt.scatter(candidate_richness, catalogue_richness)
    plt.show() # good algorithm if x=y trend


compare_clusters(bcg_arr, deep_field_arr)
