#!/usr/bin/env python3

import sqlite3
import pandas as pd
from random import uniform

from grispy import GriSPy
from astropy.coordinates import SkyCoord

from params import *
from cluster import Cluster
from plotting import check_plots
from analysis.methods import linear_to_angular_dist, mean_separation, redshift_to_velocity

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


# --- helper functions
def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
    dtype = [('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('z_lower', np.float64), ('z_upper', np.float64), ('absMag', np.float64), ('logSM', np.float64), ('logSFR', np.float64), ('ID', np.float64), ('logLR', np.float64), ('N(0.5)', np.float64)]
    arr_1_struc = arr_1.ravel().view(dtype=dtype)
    arr_2_struc = arr_2.ravel().view(dtype=dtype)
    S = np.setdiff1d(arr_1_struc, arr_2_struc)
    return S


def find_number_count(center, galaxies, distance=0.5*u.Mpc/u.littleh):
    '''
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

    '''
    if isinstance(center, Cluster):
        coords = [center.ra, center.dec]
        z = center.z
    else:
        coords = center[:2]
        z = center[2]

    N_gsp = GriSPy(galaxies[:,:2], metric='haversine')
    distance = linear_to_angular_dist(distance, z).value # convert to angular distance
    n_dist, n_idx = N_gsp.bubble_neighbors(np.array([coords]), distance_upper_bound=distance)
    n_points = galaxies[tuple(n_idx)]

    return len(n_points)


def center_overdensity(center, galaxy_data, max_velocity): # calculate overdensity of cluster

    # select 300 random points (RA and dec)
    n = 300
    ra_random = np.random.uniform(low=min(galaxy_data[:,0]), high=max(galaxy_data[:,0]), size=n)
    dec_random = np.random.uniform(low=min(galaxy_data[:,1]), high=max(galaxy_data[:,1]), size=n)
    points = np.vstack((ra_random, dec_random)).T
    assert points.shape == (n,2)

    # select all galaxies within max velocity
    velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2], center.z)) <= max_velocity]
    # velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center.z) <= 0.02*(1+center.z)]

    virial_gsp = GriSPy(velocity_bin[:,:2], metric='haversine')
    max_dist = linear_to_angular_dist(0.5*u.Mpc/u.littleh, center.z).value
    v_dist, v_idx = virial_gsp.bubble_neighbors(points, distance_upper_bound=max_dist)

    # find the N(0.5) mean and rms for the 300 points
    N_arr = np.array([len(idx) for idx in v_idx])

    N_mean = np.mean(N_arr)
    N_rms = np.sqrt(np.mean(N_arr**2)) #rms
    D = (center.N - N_mean)/N_rms
    return D


# --- search algorithms ---
def luminous_search(galaxy_data, max_velocity, path='FoF\\analysis\\derived_datasets\\'):
    '''
    This search selects galaxies of interest that would be considered as cluster centers. This speeds up the following FoF step by limiting the number of galaxies to search around.
    Galaxies of interest are selected by their number count, N(d) >= 8 where d = 0.5
    2 arrays are returned and saved into the default path.

    Parameters
    ----------
    galaxy_data: ndarray, shape (n, 6)
        Galaxy data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'LR']

    max_velocity: float
        Largest velocity a galaxy can have with respect to the cluster center in km/s

    Returns
    -------
    galaxy_data: ndarray, shape (n, 7)
        Galaxy data with additional column of N(0.5). N(0.5) is the number of galaxies within 0.5h^-1 Mpc of the galaxy.

    luminous_galaxy: ndarray, shape (m, 7)
        m < n sample of galaxy_data with galaxies of N(0.5) >= 8 and sorted by decreasing N(0.5).

    '''

    if isinstance(galaxy_data, pd.DataFrame):
        galaxy_data = galaxy_data.values
    print(f'No. of galaxies to search through: {len(galaxy_data)}')

    # add additional N(0.5) column to galaxy_data
    main_arr = np.column_stack([galaxy_data, np.zeros(galaxy_data.shape[0])])
    assert galaxy_data.shape[1]+1 == main_arr.shape[1], 'Galaxy array is wrong shape.'

    # count N(0.5) for each galaxy
    for i, galaxy in enumerate(main_arr): 

        vel_bin = main_arr[abs(redshift_to_velocity(galaxy_data[:,2], galaxy[2])) <= max_velocity] # select galaxies within appropriate redshift range
        # vel_bin = main_arr[abs(main_arr[:,2]-galaxy[2]) <= 0.02*(1+galaxy[2])]

        if (len(vel_bin) >= 8):
            main_arr[i,-1] = find_number_count(galaxy, vel_bin[:,:2], 0.5*u.Mpc/u.littleh) # galaxies within 0.5 h^-1 Mpc

    luminous_galaxy = main_arr[main_arr[:,-1] >= 8] # select for galaxies with N(0.5) >= 8
    luminous_galaxy = luminous_galaxy[luminous_galaxy[:,-1].argsort()[::-1]] # sort by N(0.5) in desc
    assert luminous_galaxy[0,-1] == max(luminous_galaxy[:,-1]), 'Galaxy array is not sorted by N(0.5)'

    print(f'Total Number of Galaxies in sample: {len(main_arr)}, Number of candidate centers: {len(luminous_galaxy)}')

    np.savetxt(path+'COSMOS_galaxy_data.csv', main_arr, delimiter=',')
    np.savetxt(path+'luminous_galaxy_velocity.csv', luminous_galaxy, delimiter=',')

    return main_arr, luminous_galaxy


def FoF(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, richness, overdensity):
    '''
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

    luminous_data: ndarray, shape (m,7)

    max_velocity: float, units [km/s]

    linking_length_factor: float

    virial_radius: float, units [Mpc/littleh]

    richness: integer

    overdensity: float

    Returns
    -------
    candidates: list of cluster.Cluster

    '''
    candidates = []
    # sep_arr = [] # tracks change in linking length with redshift

    # tracker identifies galaxies that have been included in another cluster previously 
    # this speeds up algorithm a lot by including such galaxies
    # luminous_data was sorted by N(0.5) before to ensure larger clusters are prioritized
    tracker = np.ones(len(luminous_data)) 

    # identify cluster candidates
    for i, center in enumerate(luminous_data): # each row is a candidate center to search around

        if tracker[i]:
            velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2], center[2])) <= max_velocity] # select galaxies within max velocity
            # velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center[2]) <= 0.02*(1+center[2])]

            virial_gsp = GriSPy(velocity_bin[:,:2], metric='haversine')

            # given virial radius is in proper distances, we convert to comoving distance to account for cosmological expansion.
            ang_virial_radius = linear_to_angular_dist(virial_radius, center[2]).to('rad') # convert proper virial radius to angular distance
            max_dist = (ang_virial_radius*cosmo.comoving_transverse_distance(center[2])).to(u.Mpc, u.dimensionless_angles()) # convert to comoving distance
            max_dist = linear_to_angular_dist(max_dist, center[2]).value # convert comoving distance to angular distance

            virial_dist, virial_idx = virial_gsp.bubble_neighbors(np.array([center[:2]]), distance_upper_bound=max_dist) # for some reason, the center must be a ndarray of (n,2)
            virial_points = velocity_bin[tuple(virial_idx)] # convert to tuple for deprecation warning

            if len(virial_points) >= 12: # reject if less than 12 surrounding galaxies within virial radius (to save time) 
                mean_sep = mean_separation(len(virial_points), center[2], max_dist*u.degree, max_velocity) # Mpc
                linking_length = linking_length_factor*mean_sep # determine transverse LL from local mean separation
                # sep_arr.append([linking_length.value, center[2]])
                linking_length = linear_to_angular_dist(linking_length, center[2]).value # fix linking length here

                f_gsp = GriSPy(virial_points[:,:2], metric='haversine')
                f_dist, f_idx = f_gsp.bubble_neighbors(np.array([center[:2]]), distance_upper_bound=linking_length) # select galaxies within linking length
                f_points = virial_points[tuple(f_idx)]

                member_galaxies = f_points
                fof_dist, fof_idx = f_gsp.bubble_neighbors(f_points[:,:2], distance_upper_bound=linking_length) # select all galaxies within 2 linking lengths

                for idx in fof_idx:
                    fof_points = virial_points[idx]

                    # ensure no repeated points in cluster
                    mask = np.isin(fof_points, member_galaxies, invert=True) # filter for points not already accounted for
                    vec_mask = np.isin(mask.sum(axis=1), center.shape[0])
                    fof_points = fof_points[vec_mask].reshape((-1,center.shape[0])) # points of 2 linking lengths (FoF)

                    if len(fof_points):
                        member_galaxies = np.concatenate((member_galaxies, fof_points)) # merge all FoF points within 2 linking lengths

                if len(member_galaxies) >= richness: # must have >= richness
                    c = Cluster(center, member_galaxies)
                    candidates.append(c)

                    if not i % 100:
                        logging.info(f'{i} ' + c.__str__())

                    # locate centers within member_galaxies (centers of interest)
                    member_gal_id = member_galaxies[:,4]
                    luminous_gal_id = luminous_data[:,4]
                    coi, _, coi_idx = np.intersect1d(member_gal_id, luminous_gal_id, return_indices=True)

                    # update tracker to 0 for these points
                    for i in coi_idx:
                        tracker[i] = 0

            # if len(candidates) >= 300: # for quick testing
            #     break

    # check_plots(candidates)

    # tracks mean separation across redshift
    # sep_arr = np.array(sep_arr)
    # plt.plot(sep_arr[:,1], sep_arr[:,0], '.')
    # plt.show()

    # perform overlap removal and merger
    print('Performing overlap removal')
    candidate_centers = np.array([[c.ra, c.dec, c.z, c.gal_id] for c in candidates]) # get specific attributes from candidate center space
    candidates = np.array(candidates)
    merged_candidates = candidates.copy()
    gal_id_space = candidate_centers[:,3]

    for center in candidates:

        # identity overlapping centers (centers lying within virial radius of current cluster)
        velocity_bin = candidate_centers[abs(redshift_to_velocity(candidate_centers[:,2], center.z)) <= max_velocity] # select galaxies within max velocity
        # velocity_bin = candidate_centers[abs(candidate_centers[:,2]-center.z) <= 0.02*(1+center.z)]

        center_gsp = GriSPy(velocity_bin[:,:2], metric='haversine')
        c_coords = [center.ra, center.dec]
        max_dist = linear_to_angular_dist(virial_radius, center.z).value # convert virial radius to angular distance
        c_dist, c_idx = center_gsp.bubble_neighbors(np.array([c_coords]), distance_upper_bound=max_dist) # for some reason, the center must be a ndarray of (n,2)
        c_points = velocity_bin[tuple(c_idx)]

        # merge each overlapping cluster
        if len(c_points):
            for c in c_points:
                c = candidates[gal_id_space == c[-1]][0] 
                
                if center.gal_id == c.gal_id: # if same center, ignore
                    continue

                # modify the cluster's galaxies in merged_candidates array
                if len(c.galaxies) and len(center.galaxies): # check both clusters are not empty
                    S = setdiff2d(c.galaxies, center.galaxies) # identify overlapping galaxies
                    if len(S):
                        new_c = merged_candidates[gal_id_space == c.gal_id][0] # c from merged_candidates
                        new_center = merged_candidates[gal_id_space == center.gal_id][0] # center from merged_candidates

                        c_galaxies, center_galaxies = c.remove_overlap(center) 
                        new_c.galaxies = c_galaxies
                        new_center.galaxies = center_galaxies


    merged_candidates = np.array([c for c in merged_candidates if c.richness >= richness]) # select only clusters >= richness
    if len(merged_candidates) >= len(candidates):
        logging.warning('No candidates were merged!')

    center_shifted_candidates = merged_candidates.copy()
    # check_plots(merged_candidates)

    # replace candidate center with brightest galaxy in cluster
    print('Searching for BCGs')
    merged_candidates = sorted(merged_candidates, key=lambda x: x.N, reverse=True) # sort by N

    for center in merged_candidates:
        center_shifted_gal_id_space =  np.array([c.gal_id for c in center_shifted_candidates])

        # identify galaxies within 0.25*virial radius
        cluster_gsp = GriSPy(center.galaxies[:,:2], metric='haversine') # for galaxies within a cluster
        c_coords = [center.ra, center.dec]
        max_dist = 0.25*(linear_to_angular_dist(virial_radius, center.z).value) # convert virial radius to angular distance
        c_dist, c_idx = cluster_gsp.bubble_neighbors(np.array([c_coords]), distance_upper_bound=max_dist) # for some reason, the center must be a ndarray of (n,2)
        bcg_arr = center.galaxies[tuple(c_idx)]

        if len(bcg_arr) and len(center.galaxies): # check there are any galaxies within 0.25*virial radius

            mag_sort = bcg_arr[bcg_arr[:,3].argsort()] # sort selected galaxies by abs mag (brightness)
            mask = np.isin(mag_sort[:,4], center_shifted_gal_id_space, invert=True) # filter for galaxies that are not existing centers
            mag_sort = mag_sort[mask]
            
            if len(mag_sort):
                bcg = mag_sort[0] # brightest cluster galaxy (bcg)

                # if bcg brighter than current center, replace it as center
                if (abs(bcg[3]) > abs(center.bcg_absMag)) and (bcg[4] != center.gal_id):
                    new_cluster = Cluster(bcg, center.galaxies) # initialize new center

                    center_shifted_candidates = np.delete(center_shifted_candidates, np.where([c.gal_id for c in center_shifted_candidates] == center.gal_id))
                    center_shifted_candidates = np.concatenate((center_shifted_candidates, np.array([new_cluster]))) # add new center to array


    center_shifted_candidates = np.array([c for c in center_shifted_candidates if c.richness >= richness]) # select only clusters >= richness
    final_candidates = []
    # check_plots(center_shifted_candidates)

    # N(0.5) and galaxy overdensity
    print('Selecting appropriate clusters')
    for center in center_shifted_candidates:
        center.N = find_number_count(center, center.galaxies, distance=0.5*u.Mpc/u.littleh) # find number count N(0.5)
        center.D = center_overdensity(center, galaxy_data, max_velocity) # find overdensity D

        # Initialize the cluster only if N(0.5) >= 8 and D >= overdensity
        if center.N >= 8 and center.richness >= richness and center.D >= overdensity:
            final_candidates.append(center)

    # check_plots(final_candidates)
    return final_candidates


# -- interloper removal
def three_sigma(center, member_galaxies, n):
    '''
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
    '''

    # add two additional columns to member_galaxies for peculiar velocity and radial distance
    N = member_galaxies.shape
    member_galaxies = np.column_stack([member_galaxies, np.zeros((N[0], 2))])
    assert member_galaxies.shape[1] == N[1]+2, 'Cluster member array is wrong shape.'

    # initialize velocity column with redshift-velocity formula
    member_galaxies[:,-2] = redshift_to_velocity(member_galaxies[:,2], center.z)

    # initialize projected radial distance column
    center_coord = SkyCoord(center.ra*u.deg, center.dec*u.deg)
    member_galaxies_coord = SkyCoord(member_galaxies[:,0]*u.deg, member_galaxies[:,1]*u.deg)
    separations = center_coord.separation(member_galaxies_coord)
    d_A = cosmo.angular_diameter_distance(center.z)
    member_galaxies[:,-1] = (separations*d_A).to(u.Mpc, u.dimensionless_angles())

    # bin galaxies by radial distance from center
    cleaned_members = []
    s = member_galaxies[:,-1]
    bins = np.linspace(min(s), max(s), n)
    digitized = np.digitize(s, bins) # bin members into n radial bins

    for i in range(1, len(bins)+1):
        bin_galaxies = member_galaxies[np.where(digitized==i)] # galaxies in current bin

        size_before = len(bin_galaxies)
        size_after = 0
        size_change = size_after - size_before
        # assert size_before <= 1, 'Too many bins.'

        while (size_change != 0) and (len(bin_galaxies) > 0): # repeat until convergence for each bin
            size_before = len(bin_galaxies)

            if len(bin_galaxies) > 2:
                bin_vel_arr = bin_galaxies[:,-2]
                sort_idx = np.argsort(np.abs(bin_vel_arr-np.mean(bin_vel_arr))) # sort by deviation from mean
                bin_galaxies = bin_galaxies[sort_idx]

                if bin_galaxies[-1,-1] == 0.0: # if galaxy with largest deviation is center
                    ignored_galaxy = bin_galaxies[-2,:]
                    new_bin_galaxies = np.delete(bin_galaxies, (-2), axis=0)
                else:
                    ignored_galaxy = bin_galaxies[-1,:]
                    new_bin_galaxies = bin_galaxies[:-1,:] # do not consider galaxy with largest deviation

                assert ignored_galaxy[-1] != 0.0, 'Galaxy with largest deviation is the cluster center'
                new_vel_arr = new_bin_galaxies[:,-2]

                if (bin_galaxies[-1,-1] != 0.0): # if ignored galaxy is not center
                    if len(new_vel_arr) > 1:
                        bin_mean = np.mean(new_vel_arr) # velocity mean
                        bin_dispersion = sum((new_vel_arr-bin_mean)**2)/(len(new_vel_arr)-1) # velocity dispersion

                        if (abs(ignored_galaxy[-2] - bin_mean) > 3*np.sqrt(bin_dispersion)): # if outlier velocity lies outside 3 sigma
                            bin_galaxies = new_bin_galaxies # remove outlier

            size_after = len(bin_galaxies)
            size_change = size_after - size_before
            
        cleaned_members.append(bin_galaxies[:,:-2])
    
    assert len(cleaned_members) == len(bins)
    cleaned_members = np.concatenate(cleaned_members, axis=0) # flatten binned arrays
    return cleaned_members


def interloper_removal(data):

    print(f'Removing interlopers from {len(data)} clusters')
    number_removed = []

    for i, c in enumerate(data):
        initial_size = len(c.galaxies)
        cleaned_galaxies = three_sigma(c, c.galaxies, 10)
        c.galaxies = cleaned_galaxies
    
        size_change = initial_size - len(cleaned_galaxies)
        if i % 100 == 0:
            print(f'Interlopers removed from cluster {i}: {size_change}')
        number_removed.append(size_change)

    cleaned_candidates = [c for c in data if c.richness >= richness]
    print(f'Number of clusters: {len(cleaned_candidates)} returned')
    return cleaned_candidates, number_removed



if __name__ == "__main__":

    ###########################
    # IMPORT RAW DATA FROM DB #
    ###########################
    # print('Selecting galaxy survey...')
    # conn = sqlite3.connect('FoF\\processing\\datasets\\galaxy_clusters.db')
    # df = pd.read_sql_query('''
    #     SELECT *
    #     FROM mag_lim
    #     WHERE redshift BETWEEN ? AND ?
    #     ORDER BY redshift''', conn, params=(min_redshift, max_redshift+0.2))

    # print(df.columns)
    # df = df.loc[:, ['ra','dec','redshift', 'z_lower', 'z_upper+', 'RMag', 'log_Stellar_Mass', 'log_SFR', 'ID', 'LR']] # select relevant columns
    # df = df.sort_values('redshift')


    #############################
    # LUMINOUS CANDIDATE SEARCH #
    #############################
    # main_arr, luminous_arr = luminous_search(df, max_velocity=max_velocity)


    ######################
    # FRIENDS-OF-FRIENDS #
    ######################
    # main_arr = np.loadtxt('FoF\\analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 
    # luminous_arr = np.loadtxt('FoF\\analysis\\derived_datasets\\luminous_galaxy_velocity.csv', delimiter=',')

    # main_arr = main_arr[(main_arr[:,2] >= min_redshift) & (main_arr[:,2] <= max_redshift + 0.02)]
    # luminous_arr = luminous_arr[(luminous_arr[:,2] >= min_redshift) & (luminous_arr[:,2] <= max_redshift)]
    # print(f'Number of galaxies: {len(main_arr)}, Number of candidates: {len(luminous_arr)}')

    # candidates = FoF(main_arr, luminous_arr, max_velocity=max_velocity, linking_length_factor=linking_length_factor, 
    #                      virial_radius=virial_radius, richness=richness, overdensity=D)
    # print(f'{len(candidates)} candidate clusters found.')

    # # pickle candidates list
    # with open(fname+'candidates.dat', 'wb') as f:
    #     pickle.dump(candidates, f)


    ######################
    # INTERLOPER REMOVAL #
    ######################
    # with open(fname+'candidates.dat', 'rb') as f:
        # candidates = pickle.load(f)

    # cleaned_candidates, number_removed = interloper_removal(candidates)

    # check interloper removal
    # plt.hist(number_removed, bins=10)
    # plt.show()

    # with open(fname+'cleaned_candidates.dat', 'wb') as f:
    #     pickle.dump(cleaned_candidates, f)



    ####################
    # CLUSTER FLAGGING #
    ####################
    # manually flag clusters near edges or unvirialized (bad clusters)
    # with open(fname+'cleaned_candidates_flagged.dat', 'rb') as f:
    #     cleaned_candidates = pickle.load(f)

    # check clusters manually through check_plots
    def flag_plots(clusters):

        clusters = [c for c in clusters if c.flag_poor == False]
        coords = np.array([[c.ra, c.dec] for c in clusters])
        clusters = np.array(clusters)
        z_arr = np.array([c.z for c in clusters])        

        bins = np.arange(0.5,2.53,0.00666)
        digitized = np.digitize(z_arr, bins)

        for i in range(150,len(bins)):
            binned_data = clusters[np.where(digitized==i)]
            binned_data = sorted(binned_data, key=lambda x: x.ra)
            logging.info(f'Number of clusters in bin {i}: {len(binned_data)}')

            if len(binned_data): # plot clusters for checking
                logging.info(f'Plotting bin {i}/{len(bins)}. Clusters with binned redshift: {bins[i]}')

                fig, ax = plt.subplots(figsize=(10,8))
                ax.minorticks_on()
                ax.tick_params(top=True, right=True, labeltop=True, labelright=True, which='both', direction='inout')

                for i, center in enumerate(binned_data):
                    plt.scatter(center.galaxies[:,0], center.galaxies[:,1], s=5)
                    plt.scatter(center.ra, center.dec, s=10)
                    plt.axis([149.4, 150.8, 1.6, 2.8])
                    logging.info(f'{i}. Plotting ' + center.__str__())
                plt.show()


                print('Scan the plotted clusters for flagging. Clusters that are poor, merging or on the outer edges should be flagged.')
                flagged = input('Choose the clusters to flag: ') # format: 1,2,4
                
                if flagged != '-':
                    idx_list = list(map(int, flagged.strip().split(',')))
                    flagged_clusters = np.array(binned_data)[idx_list]

                    # if clusters are deemed poor, self.flag_poor = True
                    for c in flagged_clusters:
                        c.flag_poor = True

                    print([c.flag_poor for c in binned_data])

        with open(fname+'cleaned_candidates_flagged.dat', 'wb') as f:
            pickle.dump(clusters, f)

    # flag_plots(cleaned_candidates)
