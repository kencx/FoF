import sqlite3
import pandas as pd
from astropy.coordinates import SkyCoord, Distance, match_coordinates_sky

from params import *
from analysis.methods import linear_to_angular_dist

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
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
    SELECT ra, dec, redshift, Ngal, R
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

# xray_df = pd.read_csv('datasets\\xray_group_catalog_tbl.csv')
# xray_df = xray_df[['ra','dec','redshift','nmem']]
# xray_df = xray_df[(xray_df['redshift']>=0.5) & (xray_df['nmem']>=15)]
# xray_arr = xray_df.values


def compare_clusters(cluster_sample, cluster_catalog, richness_plot=True):
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
    cluster_matched = np.zeros((len(cluster_sample), 3))

    sample_c = np.array([c.richness for c in cluster_sample])
    coords = np.array([[c.ra, c.dec, c.z] for c in cluster_sample])
    z_arr = np.array([c.z for c in cluster_sample])

    min_redshift, max_redshift = min(z_arr), max(z_arr)
    print(f'Minimum redshift: {min_redshift}; Maximum redshift: {max_redshift}')
    
    coords = coords[(coords[:,2] >= min_redshift) & (coords[:,2]<= max_redshift)]
    cluster_catalog = cluster_catalog[(cluster_catalog[:,2] >= min_redshift) & (cluster_catalog[:,2]<= max_redshift)]
    logging.debug(f'No. of galaxies in sample: {len(cluster_sample)} and No. of galaxies in catalog: {len(cluster_catalog)}')

    sample_coords = SkyCoord(coords[:,0]*u.degree, coords[:,1]*u.degree)

    for i, c in enumerate(cluster_catalog): # for each cluster in the catalog, find matching sample clusters

        z_cut = 0.04*(1+c[2])
        mask = abs(coords[:,2] - c[2]) <= z_cut
        z_cut_coords = coords[mask]
        # z_cut_coords = np.column_stack((z_cut_coords[:,0]*u.deg, z_cut_coords[:,1]*u.deg))
        z_cut_coords = SkyCoord(z_cut_coords[:,0]*u.deg, z_cut_coords[:,1]*u.deg)
        sample_cut = sample_c[mask]

        max_sep = linear_to_angular_dist(0.5*u.Mpc/u.littleh, c[2])
        c_coord = SkyCoord(c[0]*u.degree, c[1]*u.degree)

        if len(z_cut_coords):
            idx, d2d, _ = match_coordinates_sky(c_coord, z_cut_coords)
            if d2d < max_sep:
                sample_matches = z_cut_coords[idx]
                if (sample_matches):
                    count += 1
                    cluster_matched[i,:] = coords[idx,:]
                    catalogue_richness[i] = cluster_catalog[i,-1]
                    sample_richness[i] = sample_cut[idx]

    print('{n}/{total} matched. {number} % of candidates matched'.format(n=count, total=len(cluster_catalog), number=(count/len(cluster_catalog))*100))
    
    if richness_plot:
        fig, ax= plt.subplots(figsize=(10,8))

        ax.scatter(sample_richness, catalogue_richness, s=8, alpha=0.75)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')
        ax.set_aspect('equal')
        ax.set_title('CES Ultra Deep Catalogue')
        ax.set_xlabel('Sample Richness')
        ax.set_ylabel('Catalog Richness')
        plt.show() 

    return cluster_matched



if __name__ == "__main__":
    # xray_matched = compare_clusters(virial_clusters, xray_arr, richness_plot=False)
    cosmic_web_matched = compare_clusters(virial_clusters, cosmic_web_arr, richness_plot=True)
    ultra_deep_matched = compare_clusters(virial_clusters, ultra_deep_arr, richness_plot=True)
    deep_field_matched = compare_clusters(virial_clusters, deep_field_arr, richness_plot=True)
    ultra_deep_matched = ultra_deep_matched[ultra_deep_matched>0].reshape(-1,3)
    cosmic_web_matched = cosmic_web_matched[cosmic_web_matched>0].reshape(-1,3)

    res = (ultra_deep_matched[:, None] == cosmic_web_matched).all(-1).any(-1)
    print(sum(res)) # number of clusters matched in both catalogs
