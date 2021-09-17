#!/usr/bin/env python3

import pickle
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from astropy import units as u
from grispy import GriSPy
from cluster import Cluster
from analysis.methods import (
    linear_to_angular_dist,
    mean_separation,
    redshift_to_velocity,
)

import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger("matplotlib.font_manager").disabled = True


def plot_clusters(clusters, redshift_range, flagging=True):
    # plt.rc('font', family='serif', size='12')

    clusters = [c for c in clusters if c.flag_poor == False]
    coords = np.array([[c.ra, c.dec] for c in clusters])
    clusters = np.array(clusters)
    z_arr = np.array([c.z for c in clusters])
    lims = [
        np.min(coords[:, 0]),
        np.max(coords[:, 0]),
        np.min(coords[:, 1]),
        np.max(coords[:, 1]),
    ]

    bins = np.arange(redshift_range[0], redshift_range[1], 0.00666)
    digitized = np.digitize(z_arr, bins)

    for i in range(0, len(bins)):
        binned_data = clusters[np.where(digitized == i)]
        binned_data = sorted(binned_data, key=lambda x: x.ra)
        logging.info(f"Number of clusters in bin {i}: {len(binned_data)}")

        if len(binned_data):  # plot clusters for checking
            logging.info(
                f"Plotting bin {i}/{len(bins)}. Clusters with binned redshift: {bins[i]}"
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.minorticks_on()
            # ax.tick_params(top=True, right=True, labeltop=True, labelright=True, which='both', direction='inout')
            ax.tick_params(top=True, right=True, which="both", direction="inout")
            plt.title(f"z = {bins[i]:.2f}")

            for i, center in enumerate(binned_data):
                plt.scatter(center.galaxies[:, 0], center.galaxies[:, 1], s=5)
                plt.scatter(center.ra, center.dec, s=10)
                circ = plt.Circle(
                    (center.ra, center.dec),
                    radius=0.14169,
                    color="r",
                    fill=False,
                    ls="--",
                    alpha=0.7,
                )
                ax.add_patch(circ)
                plt.axis(lims)
                logging.info(f"{i}. Plotting " + center.__str__())
            plt.tight_layout()
            plt.show()

            if flagging:
                print(
                    "Scan the plotted clusters for flagging. Clusters that are poor, merging or on the outer edges should be flagged."
                )
                flagged = input("Choose the clusters to flag: ")  # format: 1,2,4

                if flagged != "-":
                    idx_list = list(map(int, flagged.strip().split(",")))
                    flagged_clusters = np.array(binned_data)[idx_list]

                    # if clusters are deemed poor, self.flag_poor = True
                    for c in flagged_clusters:
                        c.flag_poor = True

                    print([c.flag_poor for c in binned_data])

    if flagging:
        with open(fname + "cleaned_candidates_flagged.dat", "wb") as f:
            pickle.dump(clusters, f)


def lowess(x, y, f=2.0 / 3.0, iter=3):
    """
    THIS FUNCTION IS FROM agramfort @ https://gist.github.com/agramfort/850437

    lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)],
                ]
            )
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


# --- helper functions
def setdiff2d(cluster1, cluster2):
    """
    Find common rows between two 2d arrays and return the rows of cluster1 not present in cluster2
    """
    gal1 = cluster1[:, 6]  # use gal_id to extract unique galaxies
    gal2 = cluster2[:, 6]
    mask = np.isin(gal1, gal2, invert=True)
    S = cluster1[mask, :]
    return S


def find_number_count(center, galaxies, distance=0.5 * u.Mpc / u.littleh):
    """
    Computes number count of the center of interest, within a distance d.

    Number count, N(d) is the number of galaxies surrounding the center of interest.

    Note: Uses haversine metric so angular coordinates must be used.

    Parameters
    ----------
    center: ndarray, shape (1, 3) or cluster.Cluster.
        Center of interest

    galaxies: ndarray, shape (n, 2)
        Sample of galaxies to search.

    distance: float with units of [Mpc/littleh], default: 0.5*u.Mpc/u.littleh
        distance, d.

    Returns
    -------
    len(n_points): int
        N(d), Number of points around center of interest, within distance d.

    """
    if isinstance(center, Cluster):
        coords = [center.ra, center.dec]
        z = center.z
    else:
        coords = center[:2]
        z = center[2]

    N_gsp = GriSPy(galaxies[:, :2], metric="haversine")
    distance = linear_to_angular_dist(distance, z).value  # convert to angular distance
    n_dist, n_idx = N_gsp.bubble_neighbors(
        np.array([coords]), distance_upper_bound=distance
    )
    n_points = galaxies[tuple(n_idx)]

    return len(n_points)


def center_overdensity(
    center, galaxy_data, max_velocity
):  # calculate overdensity of cluster

    # select 300 random points (RA and dec)
    n = 300
    ra_random = np.random.uniform(
        low=min(galaxy_data[:, 0]), high=max(galaxy_data[:, 0]), size=n
    )
    dec_random = np.random.uniform(
        low=min(galaxy_data[:, 1]), high=max(galaxy_data[:, 1]), size=n
    )
    points = np.vstack((ra_random, dec_random)).T
    assert points.shape == (n, 2)

    # select all galaxies within max velocity
    velocity_bin = galaxy_data[
        abs(redshift_to_velocity(galaxy_data[:, 2], center.z)) <= max_velocity
    ]

    virial_gsp = GriSPy(velocity_bin[:, :2], metric="haversine")
    max_dist = linear_to_angular_dist(0.5 * u.Mpc / u.littleh, center.z).value
    v_dist, v_idx = virial_gsp.bubble_neighbors(points, distance_upper_bound=max_dist)

    # find the N(0.5) mean and rms for the 300 points
    N_arr = np.array([len(idx) for idx in v_idx])

    N_mean = np.mean(N_arr)
    N_rms = np.sqrt(np.mean(N_arr ** 2))  # rms
    D = (center.N - N_mean) / N_rms
    return D


def export(clusters, name):
    from astropy.io import fits

    N = len(clusters[0].properties)
    export_arr = np.zeros((len(clusters), N))

    for i, c in enumerate(clusters):
        export_arr[i] = c.properties

    hdu = fits.PrimaryHDU(export_arr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname + name + ".fits")

    return export_arr
