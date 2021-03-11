#!/usr/bin/env python3

import sqlite3
import pandas as pd
from astropy.coordinates import SkyCoord, Distance, match_coordinates_sky

from params import *
from analysis.methods import linear_to_angular_dist

import logging
logging.basicConfig(filename=fname+'sensitivity_test.txt', level=logging.DEBUG, format='%(levelname)s: %(asctime)s %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

checking = True # change to False if finalizing plots

if not checking:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)

with open(fname+'clusters.dat', 'rb') as f:
    virial_clusters = pickle.load(f)


#####################
# IMPORT CATALOGUES #
#####################

conn = sqlite3.connect('FoF\\processing\\datasets\\galaxy_clusters.db')

deep_field_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM deep_field 
    WHERE Ngal>=? AND redshift>=0.5
    ORDER BY redshift
    ''', conn, params=(richness,))
deep_field_arr = deep_field_df.values

cosmic_web_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal
    FROM cosmic_web_bcg
    WHERE Ngal>=? AND redshift>=0.5
    ORDER BY redshift
    ''', conn, params=(richness,))
cosmic_web_arr = cosmic_web_df.values

ultra_deep_df = pd.read_sql_query('''
    SELECT ra, dec, redshift, R
    FROM ultra_deep
    WHERE R>=? AND redshift>=0.5
    ORDER BY redshift
    ''', conn, params=(richness,))
ultra_deep_arr = ultra_deep_df.values

conn.close()

xray_df = pd.read_csv('FoF\\processing\\datasets\\xray_group_catalog_tbl.csv')
xray_df = xray_df[['ra','dec','redshift','nmem']]
xray_df = xray_df[(xray_df['redshift']>=0.5) & (xray_df['nmem']>=richness)]
xray_arr = xray_df.values


def compare_clusters(sample, catalog, richness_plot=True):
    '''
    Performs a comparison of cluster centers between a cluster sample and a published cluster catalog survey.
    The centers are matched if they lie within 0.5 h^-1 Mpc and 0.04*(1+z) of each other.
    The richness of both surveys are also compared and plotted.

    Clusters within a redshift range (to be determined) are compared. It is important to account for only clusters within a stipulated redshift range as the comparison should be fair.

    Parameters
    ----------
    sample (array-like):
    Sample of clusters to be matched.

    catalog (array-like): 
    Clusters in catalog survey to be matched against.

    Returns
    -------
    Prints the percentage of matching clusters based on the total number of clusters in the published survey.
    Plots a richness scatter plot of clusters in the sample against catalog.
    '''


    shape = len(catalog)
    sample_matched = np.zeros((shape, 4))
    catalog_matched = np.zeros((shape, 4))

    sample = np.array([[c.ra, c.dec, c.z, c.richness] for c in sample])
    z_arr = sample[:,-2]

    min_redshift, max_redshift = min(z_arr), max(z_arr)
    logging.debug(f'Minimum redshift: {min_redshift}; Maximum redshift: {max_redshift}')
    
    # filter appropriate redshift range for sample and catalog
    sample = sample[(sample[:,2] >= min_redshift) & (sample[:,2]<= max_redshift)]
    catalog = catalog[(catalog[:,2] >= min_redshift) & (catalog[:,2]<= max_redshift)]
    logging.debug(f'No. of galaxies in sample: {len(sample)} and No. of galaxies in catalog: {len(catalog)}')

    for i, c in enumerate(catalog): # for each cluster in the catalog, find matching sample clusters
        
        # select sample clusters in redshift bin
        z_cut = 0.04*(1+c[2])
        mask = abs(sample[:,2] - c[2]) <= z_cut
        z_cut_sample = sample[mask]
        sample_coord = SkyCoord(z_cut_sample[:,0]*u.deg, z_cut_sample[:,1]*u.deg)

        # convert proper distance comoving distance
        max_sep = linear_to_angular_dist(0.5*u.Mpc/u.littleh, c[2]) 
        max_sep = (max_sep*cosmo.comoving_transverse_distance(c[2])).to(u.Mpc, u.dimensionless_angles())
        max_sep = linear_to_angular_dist(max_sep, c[2]) # convert to angular comoving distance
        catalog_coord = SkyCoord(c[0]*u.degree, c[1]*u.degree)

        if len(sample_coord):
            idx, d2d, _ = match_coordinates_sky(catalog_coord, sample_coord)
            if d2d < max_sep:
                match = sample_coord[idx]
                if (match):
                    catalog_matched[i,:] = c
                    sample_matched[i,:] = sample[idx,:]
                    
    sample_matched = sample_matched[np.all(sample_matched != 0, axis=1)]
    catalog_matched = catalog_matched[np.all(catalog_matched != 0, axis=1)]
    count = len(sample_matched)
    logging.debug(f'{count}/{len(catalog)} matched. {count/len(catalog)*100} % of candidates matched')

    sample_richness = sample_matched[:,-1]
    catalog_richness = catalog_matched[:,-1]
    
    if richness_plot:
        fig, ax= plt.subplots(figsize=(10,8))

        ax.scatter(sample_richness, catalog_richness, s=8, alpha=0.75)
        if np.max([ax.get_xlim(), ax.get_ylim()]) > 200:
            lims = [0 , 200]
        else:
            lims = [0, np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')
        ax.set_aspect('equal')
        ax.set_title('CES Ultra Deep Catalogue')
        ax.set_xlabel('Sample Richness')
        ax.set_ylabel('Catalog Richness')
        plt.show() 

    return sample_matched


if __name__ == "__main__":
    xray_matched = compare_clusters(virial_clusters, xray_arr, richness_plot=True)
    cosmic_web_matched = compare_clusters(virial_clusters, cosmic_web_arr, richness_plot=True)
    ultra_deep_matched = compare_clusters(virial_clusters, ultra_deep_arr, richness_plot=True)
    deep_field_matched = compare_clusters(virial_clusters, deep_field_arr, richness_plot=True)
    
    
    # ultra_deep_matched = ultra_deep_matched[ultra_deep_matched>0].reshape(-1,4)
    # cosmic_web_matched = cosmic_web_matched[cosmic_web_matched>0].reshape(-1,4)
    # res = (ultra_deep_matched[:, None] == cosmic_web_matched).all(-1).any(-1)
    # print(sum(res)) # number of clusters matched in both catalogs
