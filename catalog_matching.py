import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Distance, match_coordinates_sky
from FoF_algorithm import linear_to_angular_dist, redshift_to_velocity

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s -  %(levelname)s -  %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


# ---
richness = 25
D = 4
fname = 'derived_datasets\\R{r}_D{d}\\'.format(r=richness, d=D)

bcg_df = pd.read_csv(fname+'\\filtered_bcg.csv')
member_galaxy_df = pd.read_csv(fname+'\\filtered_members.csv')

bcg_arr = bcg_df.values
# arr, group_n = split_df_into_groups(member_galaxy_df, 'cluster_id')


# ---- import catalogue table
conn = sqlite3.connect('galaxy_clusters.db')

deep_field_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, R
    FROM deep_field 
    WHERE Ngal>=25 AND redshift>=0.5
    ORDER BY redshift
    ''', conn)
deep_field_arr = deep_field_df.values

cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM cosmic_web_bcg
    WHERE Ngal>=25 AND redshift>=0.5
    ORDER BY redshift
    ''', conn)
cosmic_web_arr = cosmic_web_df.values

ultra_deep_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, R
    FROM ultra_deep
    WHERE R>=25 AND redshift>=0.5
    ORDER BY redshift
    ''', conn)
ultra_deep_arr = ultra_deep_df.values

conn.close()


def compare_clusters(cluster_sample, cluster_catalog):
    '''
    Performs a comparison of cluster centers between a cluster sample and a published cluster catalog survey.
    The centers are matched if they lie within 0.5 h^-1 Mpc and 0.04*(1+z) of each other.
    The richness of both surveys are also compared and plotted.

    Clusters within a redshift range (to be determined) are compared. It is important to account for only clusters within a stipulated redshift range as the comparison should be fair.

    Parameters
    ----------
    cluster_sample (array-like):
    Sample of clusters to be matched.

    cluster_catalog (array-like): 
    Clusters in catalog survey to be matched against.

    Returns
    -------
    Prints the percentage of matching clusters based on the total number of clusters in the published survey.
    Plots a richness scatter plot of clusters in the sample against catalog.
    '''

    count = 0
    catalogue_richness = np.zeros(len(cluster_sample))
    sample_richness = np.zeros(len(cluster_sample))

    min_redshift, max_redshift = min(cluster_sample[:,2]), max(cluster_catalog[:,2])
    logging.debug('Minimum redshift: {min}; Maximum redshift: {max}'.format(min=min_redshift, max=max_redshift))
    
    cluster_sample = cluster_sample[(cluster_sample[:,2] >= min_redshift)&(cluster_sample[:,2]<= max_redshift)]
    cluster_catalog = cluster_catalog[(cluster_catalog[:,2] >= min_redshift)&(cluster_catalog[:,2]<= max_redshift)]
    logging.debug('No. of galaxies in sample: {n_cand} and No. of galaxies in catalog: {n_catalog}'.format(n_cand=len(cluster_sample), n_catalog=len(cluster_catalog)))

    sample_coords = SkyCoord(cluster_sample[:,0]*u.degree, cluster_sample[:,1]*u.degree)

    for i, c in enumerate(cluster_catalog): # for each cluster in the catalog, find matching sample clusters

        z_cut = 0.04*(1+c[2])
        z_cut_coords = sample_coords[abs(cluster_sample[:,2] - c[2]) <= z_cut]
        sample_cut = cluster_sample[abs(cluster_sample[:,2] - c[2]) <= z_cut][:,6]

        max_sep = linear_to_angular_dist(0.5*u.Mpc/u.littleh, c[2])
        c = SkyCoord(c[0]*u.degree, c[1]*u.degree)

        if len(z_cut_coords):
            idx, d2d, _ = match_coordinates_sky(c, z_cut_coords)
            if d2d < max_sep:
                sample_matches = z_cut_coords[idx]
                if (sample_matches):
                    count += 1
                    catalogue_richness[i] = cluster_catalog[i,-1]
                    sample_richness[i] = sample_cut[idx]

    print('{n}/{total} matched. {number} % of candidates matched'.format(n=count, total=len(cluster_catalog), number=(count/len(cluster_catalog))*100))

    fig, ax= plt.subplots()
    ax.scatter(sample_richness, catalogue_richness, s=8, alpha=0.75)
    # ax.set_aspect('equal')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show() 

compare_clusters(bcg_arr, ultra_deep_arr)