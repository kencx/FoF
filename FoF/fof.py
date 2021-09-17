#!/usr/bin/env python3

import pandas as pd
from grispy import GriSPy
from astropy.coordinates import SkyCoord

from params import *
from cluster import Cluster
from analysis.methods import (
    linear_to_angular_dist,
    mean_separation,
    redshift_to_velocity,
)

import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger("matplotlib.font_manager").disabled = True

from helper import plot_clusters, setdiff2d, find_number_count, center_overdensity


# --- search algorithms ---
def candidate_center_search(
    galaxy_data, max_velocity, fname, path="FoF\\analysis\\derived_datasets\\"
):
    """
    This search selects galaxies of interest that would be considered as cluster centers. This speeds up the following FoF step by limiting the number of galaxies to search around.
    Galaxies of interest are selected by their number count, N(d) >= 8 where d = 0.5
    2 arrays are returned and saved into the default path.

    Parameters
    ----------
    galaxy_data: ndarray, shape (n, 6)
        Galaxy data with properties: ['ra', 'dec', 'z', 'abs mag', 'id', 'LR']

    max_velocity: float
        Largest velocity a galaxy can have with respect to the cluster center in km/s

    fname: str
        Custom filename for reference

    path: str
        Directory to save files to

    Returns
    -------
    galaxy_data: ndarray, shape (n, 7)
        Galaxy data with additional column of N(0.5). N(0.5) is the number of galaxies within 0.5h^-1 Mpc of the galaxy.

    candidate centers: ndarray, shape (m, 7)
        m < n sample of galaxy_data with galaxies of N(0.5) >= 8 and sorted by decreasing N(0.5).

    """

    if isinstance(galaxy_data, pd.DataFrame):
        galaxy_arr = galaxy_data.values
    print(f"No. of galaxies to search through: {len(galaxy_arr)}")

    # add additional N(0.5) column to galaxy_data
    main_arr = np.column_stack([galaxy_arr, np.zeros(galaxy_arr.shape[0])])
    assert galaxy_arr.shape[1] + 1 == main_arr.shape[1], "Galaxy array is wrong shape."

    # count N(0.5) for each galaxy
    for i, galaxy in enumerate(main_arr):

        vel_bin = main_arr[
            abs(redshift_to_velocity(galaxy_arr[:, 2], galaxy[2])) <= max_velocity
        ]  # select galaxies within appropriate redshift range

        if len(vel_bin) >= 8:
            main_arr[i, -1] = find_number_count(
                galaxy, vel_bin[:, :2], 0.5 * u.Mpc / u.littleh
            )  # galaxies within 0.5 h^-1 Mpc

    candidate_centers = main_arr[
        main_arr[:, -1] >= 8
    ]  # select for galaxies with N(0.5) >= 8
    candidate_centers = candidate_centers[
        candidate_centers[:, -1].argsort()[::-1]
    ]  # sort by N(0.5) in desc
    assert candidate_centers[0, -1] == max(
        candidate_centers[:, -1]
    ), "Galaxy array is not sorted by N(0.5)"

    print(
        f"Total Number of Galaxies in sample: {len(main_arr)}, Number of candidate centers: {len(candidate_centers)}"
    )

    np.savetxt(path + fname + "_galaxy_data.csv", main_arr, delimiter=",")
    np.savetxt(
        path + fname + "_candidate_centers.csv", candidate_centers, delimiter=","
    )

    return main_arr, candidate_centers


def FoF(
    galaxy_data,
    candidate_centers,
    richness,
    overdensity,
    max_velocity=2000 * u.km / u.s,
    linking_length_factor=0.1,
    virial_radius=1.5 * u.Mpc / u.littleh,
):
    """
    The Friends-of-Friends algorithm is a clustering algorithm used to identify groups of particles. In this instance, FoF is used to identify clusters of galaxies.

    FoF uses a linking length, l, whereby galaxies within a distance l from another galaxy are linked directly (as friends) and galaxies within a distance l from its friends are linked indirectly (as friends of friends). This network of particles are considered a cluster.
    After locating all candidate clusters, overlapping clusters are merged, with preference towards the center with larger N(d) and abs magnitude.
    A new cluster center is then defined as the brightess galaxy within 0.5 Mpc away from the current center.
    Finally, a cluster is only initialized if it has met the threshold richness and overdensity.

    The algorithm is sped up with:
    - numpy vectorization
    - grispy nearest neighbor implementation, which uses cell techniques to efficiently locate neighbors. This is preferred as it allows the use of the haversine metric for spherical coordinates.

    Parameters
    ----------
    galaxy_data: ndarray, shape (n,7)
        Galaxy data with compulsory properties: ['ra', 'dec', 'z', 'abs mag', 'id', 'LR', 'N']

    candidate_centers: ndarray, shape (m,7)
        Array of candidate centers with compulsory properties: ['ra', 'dec', 'z', 'abs mag', 'id', 'LR', 'N']

    max_velocity: float, units [km/s]
        Default value: 2000 km/s

    linking_length_factor: float
        Default value: 0.1

    virial_radius: float, units [Mpc/littleh]
        Default value: 1.5 hMpc

    richness: integer

    overdensity: float

    Returns
    -------
    candidates: list of cluster.Cluster object

    """
    candidates = []
    # sep_arr = [] # tracks change in linking length with redshift

    # tracker identifies galaxies that have been included in another cluster previously to speed up algorithm.
    # candidate_centers was sorted by N(0.5) before to ensure larger clusters are prioritized
    tracker = np.ones(len(candidate_centers))

    # identify cluster candidates
    for i, center in enumerate(
        candidate_centers
    ):  # each row is a candidate center to search around

        if tracker[i]:
            velocity_bin = galaxy_data[
                abs(redshift_to_velocity(galaxy_data[:, 2], center[2])) <= max_velocity
            ]  # select galaxies within max velocity

            virial_gsp = GriSPy(velocity_bin[:, :2], metric="haversine")

            # given virial radius is in proper distances, we convert to comoving distance to account for cosmological expansion.
            ang_virial_radius = linear_to_angular_dist(virial_radius, center[2]).to(
                "rad"
            )  # convert proper virial radius to angular separation
            max_dist = (
                ang_virial_radius * cosmo.comoving_transverse_distance(center[2])
            ).to(
                u.Mpc, u.dimensionless_angles()
            )  # convert to comoving distance
            max_dist = linear_to_angular_dist(
                max_dist, center[2]
            ).value  # convert comoving distance to angular separation

            virial_dist, virial_idx = virial_gsp.bubble_neighbors(
                np.array([center[:2]]), distance_upper_bound=max_dist
            )  # center must be a ndarray of (n,2)
            virial_points = velocity_bin[
                tuple(virial_idx)
            ]  # convert to tuple for deprecation warning

            if (
                len(virial_points) >= 12
            ):  # reject if <12 galaxies within virial radius (to save time)
                mean_sep = mean_separation(
                    len(virial_points),
                    center[2],
                    max_dist * u.degree,
                    max_velocity,
                    survey_area=1.7,
                )  # Mpc
                linking_length = (
                    linking_length_factor * mean_sep
                )  # determine transverse LL from local mean separation
                # sep_arr.append([linking_length.value, center[2]])
                linking_length = linear_to_angular_dist(
                    linking_length, center[2]
                ).value  # fix linking length here

                f_gsp = GriSPy(virial_points[:, :2], metric="haversine")
                f_dist, f_idx = f_gsp.bubble_neighbors(
                    np.array([center[:2]]), distance_upper_bound=linking_length
                )  # select galaxies within linking length
                f_points = virial_points[tuple(f_idx)]

                member_galaxies = f_points
                fof_dist, fof_idx = f_gsp.bubble_neighbors(
                    f_points[:, :2], distance_upper_bound=linking_length
                )  # select all galaxies within 2 linking lengths

                for idx in fof_idx:
                    fof_points = virial_points[idx]

                    # ensure no repeated points in cluster
                    mask = np.isin(
                        fof_points, member_galaxies, invert=True
                    )  # filter for points not already accounted for
                    vec_mask = np.isin(mask.sum(axis=1), center.shape[0])
                    fof_points = fof_points[vec_mask].reshape(
                        (-1, center.shape[0])
                    )  # points of 2 linking lengths (FoF)

                    if len(fof_points):
                        member_galaxies = np.concatenate(
                            (member_galaxies, fof_points)
                        )  # merge all FoF points within 2 linking lengths

                if len(member_galaxies) >= richness:  # must have >= richness
                    c = Cluster(center, member_galaxies)
                    candidates.append(c)

                    if not i % 100:
                        logging.info(f"{i} " + c.__str__())

                    # locate centers within member_galaxies (centers of interest)
                    member_gal_id = member_galaxies[:, 4]
                    luminous_gal_id = candidate_centers[:, 4]
                    coi, _, coi_idx = np.intersect1d(
                        member_gal_id, luminous_gal_id, return_indices=True
                    )

                    # update tracker to 0 for these points
                    for i in coi_idx:
                        tracker[i] = 0

            # if len(candidates) >= 100: # for quick testing
            #     break

    # plot_clusters(candidates, flagging=False) # for quick check of clusters

    # tracks mean separation across redshift
    # sep_arr = np.array(sep_arr)
    # plt.plot(sep_arr[:,1], sep_arr[:,0], '.')
    # plt.show()

    # perform overlap removal and merger
    print("Performing overlap removal")
    candidate_clusters = np.array(
        [[c.ra, c.dec, c.z, c.gal_id] for c in candidates]
    )  # get specific attributes from candidate center sample
    candidates = np.array(candidates)
    merged_candidates = candidates.copy()
    gal_id_space = candidate_clusters[:, 3]

    for center in candidates:

        # identity overlapping centers (centers lying within virial radius of current cluster)
        velocity_bin = candidate_clusters[
            abs(redshift_to_velocity(candidate_clusters[:, 2], center.z))
            <= max_velocity
        ]  # select galaxies within max velocity

        center_gsp = GriSPy(velocity_bin[:, :2], metric="haversine")
        c_coords = [center.ra, center.dec]
        max_dist = linear_to_angular_dist(
            virial_radius, center.z
        ).value  # convert virial radius to angular distance
        c_dist, c_idx = center_gsp.bubble_neighbors(
            np.array([c_coords]), distance_upper_bound=max_dist
        )  # center must be a ndarray of (n,2)
        c_points = velocity_bin[tuple(c_idx)]

        # merge each overlapping cluster
        if len(c_points):
            for c in c_points:
                c = candidates[gal_id_space == c[-1]][0]

                if center.gal_id == c.gal_id:  # if same center, ignore
                    continue

                # modify the cluster's galaxies in merged_candidates array
                if len(c.galaxies) and len(
                    center.galaxies
                ):  # check both clusters are not empty
                    S = setdiff2d(
                        c.galaxies, center.galaxies
                    )  # identify overlapping galaxies
                    if len(S):
                        new_c = merged_candidates[gal_id_space == c.gal_id][
                            0
                        ]  # c from merged_candidates
                        new_center = merged_candidates[gal_id_space == center.gal_id][
                            0
                        ]  # center from merged_candidates

                        c_galaxies, center_galaxies = c.remove_overlap(center)
                        new_c.galaxies = c_galaxies
                        new_center.galaxies = center_galaxies

    merged_candidates = np.array(
        [c for c in merged_candidates if c.richness >= richness]
    )  # select only clusters >= richness
    if len(merged_candidates) >= len(candidates):
        logging.warning("No candidates were merged!")

    bcg_clusters = merged_candidates.copy()

    # replace candidate center with brightest galaxy in cluster
    print("Searching for BCGs")
    merged_candidates = sorted(
        merged_candidates, key=lambda x: x.N, reverse=True
    )  # sort by N

    for center in merged_candidates:
        bcg_space_gal_id = np.array([c.gal_id for c in bcg_clusters])

        # identify galaxies within 0.25*virial radius
        cluster_gsp = GriSPy(
            center.galaxies[:, :2], metric="haversine"
        )  # for galaxies within a cluster
        c_coords = [center.ra, center.dec]
        max_dist = 0.25 * (
            linear_to_angular_dist(virial_radius, center.z).value
        )  # convert virial radius to angular distance
        c_dist, c_idx = cluster_gsp.bubble_neighbors(
            np.array([c_coords]), distance_upper_bound=max_dist
        )  # center must be a ndarray of (n,2)
        bcg_arr = center.galaxies[tuple(c_idx)]

        if len(bcg_arr) and len(
            center.galaxies
        ):  # check for galaxies within 0.25*virial radius

            mag_sort = bcg_arr[
                bcg_arr[:, 3].argsort()
            ]  # sort selected galaxies by abs mag (brightness)
            mask = np.isin(
                mag_sort[:, 4], bcg_space_gal_id, invert=True
            )  # filter for galaxies that are not existing centers
            mag_sort = mag_sort[mask]

            if len(mag_sort):
                bcg = mag_sort[0]  # brightest cluster galaxy (bcg)

                # if bcg brighter than current center, replace it as center
                if (abs(bcg[3]) > abs(center.bcg_absMag)) and (bcg[4] != center.gal_id):
                    new_cluster = Cluster(bcg, center.galaxies)  # initialize new center

                    bcg_clusters = np.delete(
                        bcg_clusters,
                        np.where([c.gal_id for c in bcg_clusters] == center.gal_id),
                    )
                    bcg_clusters = np.concatenate(
                        (bcg_clusters, np.array([new_cluster]))
                    )  # add new center to array

    bcg_clusters = np.array(
        [c for c in bcg_clusters if c.richness >= richness]
    )  # select only clusters >= richness
    final_clusters = []

    # N(0.5) and galaxy overdensity
    print("Selecting appropriate clusters")
    for center in bcg_clusters:
        center.N = find_number_count(
            center, center.galaxies, distance=0.5 * u.Mpc / u.littleh
        )  # find number count N(0.5)
        center.D = center_overdensity(
            center, galaxy_data, max_velocity
        )  # find overdensity D

        # Initialize the cluster only if N(0.5) >= 8 and D >= overdensity
        if (
            (center.N >= 8)
            and (center.richness >= richness)
            and (center.D >= overdensity)
        ):
            final_clusters.append(center)

    return final_clusters


# -- interloper removal
def three_sigma(center, member_galaxies, n):
    """
    Performs the 3sigma interloper removal method.

    Parameters
    ----------
    center: Cluster object

    cluster_members: array-like
    Array of member galaxies to be cleaned.

    n: int
    Number of bins.

    Returns
    -------
    cleaned_members: array-like
    Array of cleaned members with interlopers removed.
    """

    # add two additional columns to member_galaxies for peculiar velocity and radial distance
    N = member_galaxies.shape
    member_galaxies = np.column_stack([member_galaxies, np.zeros((N[0], 2))])
    assert member_galaxies.shape[1] == N[1] + 2, "Cluster member array is wrong shape."

    # initialize velocity column with redshift-velocity formula
    member_galaxies[:, -2] = redshift_to_velocity(member_galaxies[:, 2], center.z)

    # initialize projected radial distance column
    center_coord = SkyCoord(center.ra * u.deg, center.dec * u.deg)
    member_galaxies_coord = SkyCoord(
        member_galaxies[:, 0] * u.deg, member_galaxies[:, 1] * u.deg
    )
    separations = center_coord.separation(member_galaxies_coord)
    d_A = cosmo.angular_diameter_distance(center.z)
    member_galaxies[:, -1] = (separations * d_A).to(u.Mpc, u.dimensionless_angles())

    # bin galaxies by radial distance from center
    cleaned_members = []
    s = member_galaxies[:, -1]
    bins = np.linspace(min(s), max(s), n)
    digitized = np.digitize(s, bins)  # bin members into n radial bins

    for i in range(1, len(bins) + 1):
        bin_galaxies = member_galaxies[
            np.where(digitized == i)
        ]  # galaxies in current bin

        size_before = len(bin_galaxies)
        size_after = 0
        size_change = size_after - size_before
        # assert size_before <= 1, 'Too many bins.'

        while (size_change != 0) and (
            len(bin_galaxies) > 0
        ):  # repeat until convergence for each bin
            size_before = len(bin_galaxies)

            if len(bin_galaxies) > 2:
                bin_vel_arr = bin_galaxies[:, -2]
                sort_idx = np.argsort(
                    np.abs(bin_vel_arr - np.mean(bin_vel_arr))
                )  # sort by deviation from mean
                bin_galaxies = bin_galaxies[sort_idx]

                if (
                    bin_galaxies[-1, -1] == 0.0
                ):  # if galaxy with largest deviation is center
                    ignored_galaxy = bin_galaxies[-2, :]
                    new_bin_galaxies = np.delete(bin_galaxies, (-2), axis=0)
                else:
                    ignored_galaxy = bin_galaxies[-1, :]
                    new_bin_galaxies = bin_galaxies[
                        :-1, :
                    ]  # do not consider galaxy with largest deviation

                assert (
                    ignored_galaxy[-1] != 0.0
                ), "Galaxy with largest deviation is the cluster center"
                new_vel_arr = new_bin_galaxies[:, -2]

                if bin_galaxies[-1, -1] != 0.0:  # if ignored galaxy is not center
                    if len(new_vel_arr) > 1:
                        bin_mean = np.mean(new_vel_arr)  # velocity mean
                        bin_dispersion = sum((new_vel_arr - bin_mean) ** 2) / (
                            len(new_vel_arr) - 1
                        )  # velocity dispersion

                        if abs(ignored_galaxy[-2] - bin_mean) > 3 * np.sqrt(
                            bin_dispersion
                        ):  # if outlier velocity lies outside 3 sigma
                            bin_galaxies = new_bin_galaxies  # remove outlier

            size_after = len(bin_galaxies)
            size_change = size_after - size_before

        cleaned_members.append(bin_galaxies[:, :-2])

    assert len(cleaned_members) == len(bins)
    cleaned_members = np.concatenate(cleaned_members, axis=0)  # flatten binned arrays
    return cleaned_members


def interloper_removal(data):

    print(f"Removing interlopers from {len(data)} clusters")
    number_removed = []

    for i, c in enumerate(data):
        initial_size = len(c.galaxies)
        cleaned_galaxies = three_sigma(c, c.galaxies, 10)
        c.galaxies = cleaned_galaxies

        size_change = initial_size - len(cleaned_galaxies)
        if i % 100 == 0:
            print(f"Interlopers removed from cluster {i}: {size_change}")
        number_removed.append(size_change)

    cleaned_candidates = [c for c in data if c.richness >= richness]
    print(f"Number of clusters: {len(cleaned_candidates)} returned")
    return cleaned_candidates, number_removed
