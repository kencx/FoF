#!/usr/bin/env python3

import sqlite3
import pandas as pd
from astropy.coordinates import SkyCoord, Distance, match_coordinates_sky

from params import *
from analysis.methods import linear_to_angular_dist, redshift_to_velocity

import logging

logging.basicConfig(
    filename=fname + "sensitivity_test.txt",
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s %(message)s",
)
logging.getLogger("matplotlib.font_manager").disabled = True
logger = logging.getLogger()


def compare_clusters(sample, catalog, matching_distance, richness_plot=True):
    """
    Performs a comparison of cluster centers between a cluster sample and a published cluster catalog survey.
    The centers are matched if they lie within 0.5 h^-1 Mpc and z +- 0.1 of each other.
    The richness of both surveys are also compared and plotted.

    Clusters within a redshift range (to be determined) are compared. It is important to account for only clusters within a stipulated redshift range as the comparison should be fair.

    Parameters
    ----------
    sample (ndarray):
    Sample of clusters to be matched.

    catalog (ndarray):
    Clusters in catalog survey to be matched against.

    matching_distance (float):
    Maximum threshold distance between matching cluster centers

    richness_plot (bool):
    If True, show richness plot. Default is True

    Returns
    -------
    Logs the percentage of matching clusters based on the total number of clusters in the published survey.
    Plots a richness scatter plot of clusters in the sample against catalog.

    sample_matched (ndarray):
    Sample clusters that were matched.

    catalog_matching_idx (ndarray):
    Boolean array where 1 == matched clusters, 0 == unmatched clusters

    """
    sample = np.array([[c.ra, c.dec, c.z, c.richness] for c in sample])
    z_arr = sample[:, -2]

    min_redshift, max_redshift = min(z_arr), max(z_arr)
    logging.info(f"Minimum redshift: {min_redshift}; Maximum redshift: {max_redshift}")

    # filter appropriate redshift range for sample and catalog
    sample = sample[
        (sample[:, 2] >= min_redshift - 0.1) & (sample[:, 2] <= max_redshift + 0.1)
    ]
    catalog = catalog[
        (catalog[:, 2] >= min_redshift - 0.1) & (catalog[:, 2] <= max_redshift + 0.1)
    ]
    logging.info(
        f"No. of galaxies in sample: {len(sample)} and No. of galaxies in catalog: {len(catalog)}"
    )

    shape = len(catalog)
    sample_matched = np.zeros((shape, 4))
    catalog_matching_idx = np.zeros((shape))

    for i, c in enumerate(
        catalog
    ):  # for each cluster in the catalog, find matching sample clusters

        # select sample clusters in redshift bin
        z_cut = 0.1
        mask = abs(sample[:, 2] - c[2]) <= z_cut
        z_cut_sample = sample[mask]
        sample_coord = SkyCoord(z_cut_sample[:, 0] * u.deg, z_cut_sample[:, 1] * u.deg)

        if len(sample_coord):
            # convert proper distance comoving distance
            max_sep = linear_to_angular_dist(matching_distance, c[2])
            max_sep = (max_sep * cosmo.comoving_transverse_distance(c[2])).to(
                u.Mpc, u.dimensionless_angles()
            )
            max_sep = linear_to_angular_dist(
                max_sep, c[2]
            )  # convert to angular comoving distance
            catalog_coord = SkyCoord(c[0] * u.degree, c[1] * u.degree)

            idx, d2d, _ = match_coordinates_sky(
                catalog_coord, sample_coord
            )  # find nearest match, idx is index of sample
            if d2d <= max_sep:
                match = sample_coord[idx]
                if match:
                    catalog_matching_idx[i] = 1
                    sample_matched[i, :] = z_cut_sample[idx, :]

    sample_matched = sample_matched[np.all(sample_matched != 0, axis=1)]
    count = len(sample_matched)
    logging.info(
        f"{count}/{len(catalog)} matched. {count/len(catalog)*100} % of candidates matched"
    )

    sample_richness = sample_matched[:, -1]
    catalog_richness = catalog[catalog_matching_idx == 1][:, -1]

    if richness_plot:
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(sample_richness, catalog_richness, s=8, alpha=0.75)
        if np.max([ax.get_xlim(), ax.get_ylim()]) > 200:
            lims = [0, 200]
        else:
            lims = [0, np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.plot(lims, lims, "k--", alpha=0.75, label="y=x")
        ax.set_aspect("equal")
        ax.set_xlabel("Sample Richness")
        ax.set_ylabel("Catalog Richness")
        plt.show()

    return sample_matched, catalog_matching_idx


if __name__ == "__main__":

    #####################
    # IMPORT CATALOGUES #
    #####################

    conn = sqlite3.connect("FoF\\processing\\datasets\\galaxy_clusters.db")

    deep_field_df = pd.read_sql_query(
        """
        SELECT ra, dec, redshift, Ngal
        FROM deep_field
        WHERE Ngal>=? AND redshift>=0.5 AND (ra BETWEEN ? AND ?) AND (dec BETWEEN ? AND ?)
        ORDER BY redshift
        """,
        conn,
        params=(richness, lims[0], lims[1], lims[2], lims[3]),
    )
    deep_field_arr = deep_field_df.values

    cosmic_web_df = pd.read_sql_query(
        """
        SELECT ra, dec, redshift, Ngal, environment
        FROM cosmic_web_bcg
        WHERE Ngal>=? AND redshift>=0.5 AND (ra BETWEEN ? AND ?) AND (dec BETWEEN ? AND ?)
        ORDER BY redshift
        """,
        conn,
        params=(richness, lims[0], lims[1], lims[2], lims[3]),
    )
    cosmic_web_arr = cosmic_web_df.values
    cosmic_web_arr[:, 4] = np.array([i.strip() for i in cosmic_web_arr[:, 4]])
    cosmic_web_arr = cosmic_web_arr[
        (cosmic_web_arr[:, 4] == "cluster") | (cosmic_web_arr[:, 4] == "filament")
    ]

    ultra_deep_df = pd.read_sql_query(
        """
        SELECT ra, dec, redshift, R
        FROM ultra_deep
        WHERE R>=? AND redshift>=0.5 AND (ra BETWEEN ? AND ?) AND (dec BETWEEN ? AND ?)
        ORDER BY redshift
        """,
        conn,
        params=(richness, lims[0], lims[1], lims[2], lims[3]),
    )
    ultra_deep_arr = ultra_deep_df.values

    xray_df = pd.read_sql_query(
        """
        SELECT ra, dec, REDSHIFT, NMEM, LX, RA_MMGGS_ZBEST, DEC_MMGGS_ZBEST, M200C
        FROM xgroups
        WHERE NMEM>=? AND REDSHIFT>=0.5 AND (RA_MMGGS_ZBEST BETWEEN ? AND ?) AND (DEC_MMGGS_ZBEST BETWEEN ? AND ?)
        ORDER BY REDSHIFT
        """,
        conn,
        params=(richness, lims[0], lims[1], lims[2], lims[3]),
    )

    lensing_df = pd.read_sql_query(
        """
        SELECT RAdeg, DEdeg, z, Rich
        FROM lensing
        WHERE Rich>=? AND z>=0.5 AND (RAdeg BETWEEN ? AND ?) AND (DEdeg BETWEEN ? AND ?)
        ORDER BY z
        """,
        conn,
        params=(richness, lims[0], lims[1], lims[2], lims[3]),
    )
    lensing_arr = lensing_df.values

    conn.close()

    xray_mmggs = xray_df[
        ["RA_MMGGS_ZBEST", "DEC_MMGGS_ZBEST", "REDSHIFT", "NMEM", "M200C", "LX"]
    ]
    xray_mmggs_arr = xray_mmggs.values

    with open(fname + "clusters.dat", "rb") as f:
        virial_clusters = pickle.load(f)

    # uncomment when analysing unmatched clusters
    # logger.setLevel(logging.CRITICAL)

    xray_matched, xray_catalog_idx = compare_clusters(
        virial_clusters,
        xray_mmggs_arr[:, :4],
        0.5 * u.Mpc / u.littleh,
        richness_plot=False,
    )
    cosmic_web_matched, web_catalog_idx = compare_clusters(
        virial_clusters,
        cosmic_web_arr[:, :4],
        0.5 * u.Mpc / u.littleh,
        richness_plot=False,
    )
    ultra_deep_matched, ultra_catalog_idx = compare_clusters(
        virial_clusters, ultra_deep_arr, 0.5 * u.Mpc / u.littleh, richness_plot=False
    )
    deep_field_matched, deep_catalog_idx = compare_clusters(
        virial_clusters, deep_field_arr, 0.5 * u.Mpc / u.littleh, richness_plot=False
    )
    lensing_matched, lensing_catalog_idx = compare_clusters(
        virial_clusters, lensing_arr, 0.5 * u.Mpc / u.littleh, richness_plot=False
    )

    # candidates = [c for c in candidates if c.flag_poor == True]
    # candidates_arr = np.array([[c.ra,c.dec,c.z] for c in candidates])
    # radec_arr = np.array([[c.ra,c.dec,c.z] for c in virial_clusters])
    # lims = [min(radec_arr[:,0]), max(radec_arr[:,0]), min(radec_arr[:,1]), max(radec_arr[:,1])]

    # plt.rc('font', family='serif', size=14)

    # -- xray analysis
    # unmatched_xray = xray_mmggs_arr[xray_catalog_idx==0]
    # matched_xray = xray_mmggs_arr[xray_catalog_idx==1]

    # --- cosmic web analysis
    # unmatched objects
    # unmatched_web = cosmic_web_arr[web_catalog_idx==0]
    # unmatched_clusters = unmatched_web[unmatched_web[:,4]=='cluster']
    # unmatched_filaments = unmatched_web[unmatched_web[:,4]=='filament']

    # matched objects
    # matched_web = cosmic_web_arr[web_catalog_idx==1]
    # matched_clusters = matched_web[matched_web[:,4]=='cluster']
    # matched_filaments = matched_web[matched_web[:,4]=='filament']

    # -- deep field
    # unmatched_deep = deep_field_arr[deep_catalog_idx==0]
    # matched_deep = deep_field_arr[deep_catalog_idx==1]

    # -- ultra deep
    # ultra_deep_arr = ultra_deep_arr[ultra_deep_arr[:,2]<=max(radec_arr[:,2])+0.1]
    # unmatched_ultra = ultra_deep_arr[ultra_catalog_idx==0]
    # matched_ultra = ultra_deep_arr[ultra_catalog_idx==1]

    # -- lensing
    # unmatched_lensing = lensing_arr[lensing_catalog_idx==0]
    # matched_lensing = lensing_arr[lensing_catalog_idx==1]

    # --- plotting
    # plt.hist2d(xray_mmggs_arr[:,0], xray_mmggs_arr[:,1], bins=(80,80))
    # candidates_arr = candidates_arr[(candidates_arr[:,2]<=1.18)]
    # plt.scatter(candidates_arr[:,0], candidates_arr[:,1], color='magenta', s=5, alpha=0.5)
    # radec_arr = radec_arr[(radec_arr[:,2]<=1.18) & (radec_arr[:,2]>=0.4660)]
    # plt.scatter(radec_arr[:,0], radec_arr[:,1], color='grey', s=5, alpha=0.9)

    # plt.scatter(matched_deep[:,0], matched_deep[:,1], s=5, alpha=0.5, color='lime')
    # plt.scatter(matched_xray[:,0], matched_xray[:,1], color='green', s=5, alpha=0.5)
    # plt.scatter(matched_web[:,0], matched_web[:,1], color='cyan', s=5, alpha=0.5)
    # plt.scatter(matched_ultra[:,0], matched_ultra[:,1], color='blue', s=5, alpha=0.5)
    # plt.scatter(matched_lensing[:,0], matched_lensing[:,1], color='red', s=5, alpha=0.5)
    # plt.xlim(lims[0], lims[1])
    # plt.ylim(lims[2], lims[3])
    # plt.show()
