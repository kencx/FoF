# import time
# import pickle
import sqlite3
import numpy as np
import pandas as pd
from random import uniform
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.stats import norm, linregress
from scipy.spatial import cKDTree

from astropy import units as u
from astropy.constants import G
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord

from mass_estimator import redshift_to_velocity, virial_mass_estimator, projected_mass_estimator
from data_processing import split_df_into_groups

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


def linear_to_angular_dist(distance, photo_z):
    '''
    Converts proper distance (Mpc) to angular distance (deg). Used to find angular separations within clusters.

    Parameters
    ----------
    distance: float, array-like
        Distance in h^-1 * Mpc

    photo_z: float, array-like
        Associated redshift

    Returns
    -------
    d: float, array-like
        Angular distance in deg
    '''

    return (distance.to(u.Mpc, u.with_H0(cosmo.H0)) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


def mean_separation(n, radius):
    '''
    Average mean separation of n objects in an area of a given radius. Calculated by taking rho**(-1/3).

    Parameters
    ----------
    n: int
        Number of objects

    radius: float
        Radius of area

    Returns
    -------
    mean_separation: float, array-like
        Average mean separation in units of radius

    '''
    volume = 4/3 * np.pi * radius**3
    density = n/volume
    return 1/(density**(1/3))


def find_number_count(center, member_galaxies, distance):
    N_tree = cKDTree(member_galaxies[:,:2])
    n_idx = N_tree.query_ball_point(center[:2], linear_to_angular_dist(distance, center[2]).value)
    n_arr = member_galaxies[n_idx]
    return len(n_arr)


def center_overdensity(center, galaxy_data, max_velocity):
    # select 300 random points (RA and dec)
    n = 300
    ra_random = np.random.uniform(low=min(galaxy_data[:,0]), high=max(galaxy_data[:,0]), size=n)
    dec_random = np.random.uniform(low=min(galaxy_data[:,1]), high=max(galaxy_data[:,1]), size=n)
    points = np.vstack((ra_random, dec_random)).T
    assert points.shape == (n,2)

    # select all galaxies within max velocity
    # velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s]
    velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center[2]) <= 0.02*(1+center[2])]
    virial_tree = cKDTree(velocity_bin[:,:2])

    # find the N(0.5) mean and rms for the 300 points
    N_arr = np.zeros(n)
    for i, p in enumerate(points):
        idx = virial_tree.query_ball_point(p, linear_to_angular_dist(0.5*u.Mpc/u.littleh, center[2]).value)
        N_arr[i] = len(idx)

    N_mean = np.mean(N_arr)
    N_rms = np.sqrt(np.mean(N_arr**2)) #rms

    return N_mean, N_rms


def luminous_search(galaxy_data, max_velocity):
    '''
    Parameters
    ----------
    galaxy_data: array-like
        Galaxy data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'LR']

    max_velocity: float
        Largest velocity a galaxy can have with respect to the cluster center in km/s

    Returns
    -------
    galaxy_data: array-like
        Galaxy data with additional column of N(0.5). N(0.5) is the number of galaxies within 0.5h^-1 Mpc of the galaxy.

    luminous_galaxy: array-like
        Sample of galaxy_data with galaxies of N(0.5) >= 8 and sorted by decreasing N(0.5). 
    '''
    
    # add additional N(0.5) column to galaxy_data
    N = galaxy_data.shape
    temp = np.zeros((N[0], N[1]+1))
    temp[:,:-1] = galaxy_data
    galaxy_data = temp
    assert galaxy_data.shape[1] == N[1]+1, 'Galaxy array is wrong shape.'

    # count N(0.5) for each galaxy
    for i, galaxy in enumerate(galaxy_data): 
        # vel_cutoff = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(galaxy[2]))/(1+galaxy[2]) <= max_velocity*u.km/u.s] # select galaxies within appropriate redshift range
        vel_cutoff = galaxy_data[abs(galaxy_data[:,2]-galaxy[2]) <= 0.02*(1+galaxy[2])]

        if len(vel_cutoff) > 1:
            galaxy_tree = cKDTree(vel_cutoff[:,:2])
            N_idx = galaxy_tree.query_ball_point(galaxy[:2], linear_to_angular_dist(0.5*u.Mpc/u.littleh, galaxy[2]).value) # galaxies within 0.5 h^-1 Mpc
            galaxy_data[i,-1] = len(N_idx)
        else:
            galaxy_data[i,-1] = 0

    galaxy_data = galaxy_data[galaxy_data[:,-1].argsort()[::-1]] # sort by N(0.5)
    assert galaxy_data[0,-1] == max(galaxy_data[:,-1]), 'Galaxy array is not sorted by N(0.5)'

    luminous_galaxy = galaxy_data[galaxy_data[:,-1] >= 8] # select for galaxies with N(0.5) >= 8
    assert len(luminous_galaxy[luminous_galaxy[:,-1] < 8]) == 0, 'Luminous galaxy array has galaxies with N(0.5) < 8'

    print('Total Number of Galaxies in sample: {ngd}, Number of candidate centers: {nlg}'.format(ngd=len(galaxy_data), nlg=len(luminous_galaxy)))
    return galaxy_data, luminous_galaxy


def FoF(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, richness, overdensity, check_map=True):
    '''
    Steps
    -----
    Search for cluster candidates with the Friends-of-Friends algorithm.

    For each candidate center galaxy in luminous_data: 
    1. Select galaxies within a maximum velocity in galaxy_data
    2. Select galaxies within virial radius with kd-tree implementation
    3. Select galaxies within 2 linking lengths using to obtain FoF member galaxies. The linking length is determined by the mean separation of objects within a distance of the virial radius.
    5. Remove overlapping clusters (see next multi-line comment)
    4. Initialize dict storing candidates: {center: [arr of FoF points]}

    Parameters
    ----------
    galaxy_data: array-like
    
        Galaxy data with properties: ['ra', 'dec', 'redshift', 'abs mag', 'ID', 'luminosity', 'N(0.5)']
    
    luminous_data: array-like 
        Cluster center candidate data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'ID', 'luminosity', 'N(0.5)']
    
    max_velocity: float
        Largest velocity a galaxy can have with respect to the cluster center in km/s
    
    linking_length_factor: float
        Factor for transverse linking length (0 < x < 1)
    
    virial_radius: float 
        Maximum radial boundary of cluster in h^-1 Mpc
    
    check_map: bool
        Plot a scatter plot to check cluster candidates

    Returns
    -------
    final_candidates: dict
        Dictionary of cluster candidates in the given format:
            key: tuple
                Cluster center with properties: 
            value: array-like
                Array of all members belonging to this cluster center
    '''

    candidates = {}

    for i, center in enumerate(luminous_data): # for each luminous galaxy candidate
        # velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s] # select galaxies within max velocity
        velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center[2]) <= 0.02*(1+center[2])]

        virial_tree = cKDTree(velocity_bin[:,:2])
        transverse_virial_idx = virial_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value)
        transverse_virial_points = velocity_bin[transverse_virial_idx] # and select galaxies within virial radius

        if len(transverse_virial_points) >= 12:
            mean_sep = mean_separation(len(transverse_virial_points), virial_radius) # in h^-1 Mpc
            transverse_linking_length = linking_length_factor*(mean_sep.to(u.Mpc, u.with_H0(cosmo.H0))).value # determine transverse LL from local mean separation

            linking_tree = cKDTree(transverse_virial_points[:,:2])
            link_idx = linking_tree.query_ball_point(center[:2], transverse_linking_length) # select galaxies within linking length
            f_points = transverse_virial_points[link_idx]

            for f in f_points[:,:2]:
                fof_idx = linking_tree.query_ball_point(f, transverse_linking_length) # select galaxies within 2 linking lengths (FoF)
                fof_points = transverse_virial_points[fof_idx]

                mask = np.isin(fof_points, f_points, invert=True) # filter for points not already accounted for
                vec_mask = np.isin(mask.sum(axis=1), center.shape[0])
                fof_points = fof_points[vec_mask].reshape((-1,center.shape[0])) # points of 2 linking lengths (FoF)
                if len(fof_points):
                    f_points = np.concatenate((f_points, fof_points)) # merge all FoF points within 2 linking lengths

            if len(f_points) >= richness: # must have >= richness (Abell richness criterion)
                candidates[tuple(center)] = f_points # dictionary {galaxy centre: array of FoF points}
                if not i % 100:
                    print('Candidate {num} identified with {length} members'.format(num=i, length=len(f_points)))

        if len(candidates) >= 50: # for quick testing
            break

    '''
    Remove overlapping clusters

    1. For each candidate center, search for other centers within virial radius
    2. Find the set difference of member galaxies within the 2 centers, where the center with larger N(0.5) remains the primary (larger) cluster. If both centers have equal N(0.5), the brighter center is chosen.
    3. Secondary centers have their centers redefined to a new centroid of the smaller cluster
    4. Combine clusters with centers 0.25 x virial radius from each other. Likewise, the center with larger N(0.5) remains as the primary cluster 
    5. Search for member galaxies brighter than candidate cluster center within 0.25 x virial radius
    6. Replace the brightest galaxy as the new cluster center
    7. Initialize the cluster only if N(0.5) >= 8 (number count is defined as the number of member galaxies within 0.25*virial radius)
    '''

    def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
        arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
        arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
        S = np.setdiff1d(arr_1_struc, arr_2_struc)
        return S

    # def new_centroid(member_galaxies):  # new center is defined by member galaxy that is the minimal distance away from all other member galaxies
    #     center = [np.mean(member_galaxies[:,0]), np.mean(member_galaxies[:,1])]
    #     member_tree = cKDTree(member_galaxies[:,:2])
    #     _, idx = member_tree.query(center, k=1) # search for k=1 nearest neighbour in member_galaxies
    #     centroid = member_galaxies[idx]
    #     return centroid

    def remove_overlap(merged_candidates, center, arr1, arr2):
        S = setdiff2d(arr1, arr2) # finds common rows between two arrays and return the rows of cen_arr not present in member_arr
        if len(S):
            # member_galaxies = S.view(np.float64).reshape(S.shape + (-1,)) # convert back to np array of original shape
            # new_center = new_centroid(member_galaxies) # locate new center for candidate
            # merged_candidates[tuple(new_center)] = member_galaxies # initialize new center
            merged_candidates[center] = np.array([]) # delete old center

    def merge(merged_candidates, center1, center2):
        merged_candidates[center1] = np.concatenate((merged_candidates[center2], merged_candidates[center1]), axis=0) # combine the arrays
        arr, uniq_count = np.unique(merged_candidates[center1], axis=0, return_counts=True)
        merged_candidates[center1] = arr[uniq_count==1] # ensure all points are unique
        merged_candidates[center2] = np.array([]) # delete all galaxies in smaller merged galaxy


    cluster_centers = np.array([k for k in candidates.keys()]) # array of cluster centers
    merged_candidates = candidates.copy()

    for center, member_arr in candidates.items(): # for each cluster candidate center

        # velocity_bin = cluster_centers[abs(redshift_to_velocity(cluster_centers[:,2]) - redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s] # select clusters within max velocity
        velocity_bin = cluster_centers[abs(cluster_centers[:,2]-center[2]) <= 0.02*(1+center[2])]
        center_tree = cKDTree(velocity_bin[:,:2])

        virial_center_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value) # and clusters within virial radius
        centers_within_vr = velocity_bin[virial_center_idx] # these are our overlapping clusters (centers lie within another cluster's radius)

        # remove overlapping points
        if len(centers_within_vr):
            for cen in centers_within_vr:
                cen = tuple(cen)
                if center == cen: # if same center, continue
                    continue

                cen_arr = merged_candidates[cen]
                if len(cen_arr) and len(member_arr): # check both clusters are not empty
                    
                    # compare N(0.5) of two clusters and removes the common galaxies from smaller cluster. if both are same N(0.5), remove common galaxies from cluster with less bright center galaxy
                    if center[-1] > cen[-1]: 
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)

                    elif center[-1] < cen[-1]:
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)
                    
                    elif abs(center[3]) < abs(cen[3]):
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)
                    
                    else:
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)


        merge_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(0.5*virial_radius, center[2]).value) # select centers within 0.25*virial radius
        merge_centers = velocity_bin[merge_idx] # these cluster center lie too close to another center and will be merged into 1 cluster

        # merge cluster centers within 0.25*virial radius from current center
        if len(merge_centers): # check cluster not empty
            for mg in merge_centers:
                mg = tuple(mg)

                if center == mg: # if same center, continue
                    continue

                if ((mg in merged_candidates.keys())       # check if galaxy to be merged is still in candidate list (still exists)
                    and (len(merged_candidates[mg]))       # check if galaxy to be merged has member galaxies
                    and (len(merged_candidates[center]))): # check if center galaxy has member galaxies
                    
                    # compare N(0.5) of two clusters, keep larger cluster with merged points and deletes smaller one. if both have same N(0.5), compare brightness instead
                    if mg[-1] > center[-1]:
                        merge(merged_candidates, mg, center)

                    elif center[-1] > mg[-1]:
                        merge(merged_candidates, center, mg)

                    elif abs(mg[3]) > abs(center[3]):
                        merge(merged_candidates, mg, center)

                    else:
                        merge(merged_candidates, center, mg)


    # merged_candidates = {k: v for k,v in merged_candidates.items() if len(v)} # for testing

    # fig = plt.figure(figsize=(10,8)) # for checking plotting
    # count = 0
    # for k, v in merged_candidates.items():
    #     plt.scatter(v[:,0], v[:,1], s=5)
    #     plt.scatter(k[0], k[1])
    #     count += 1
    #     if count % 5 == 0:
    #         plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    #         plt.show()


    merged_cluster_centers = np.array([k for k in merged_candidates.keys()]) # array of cluster centers that have survived removal and merging
    merged_candidates = {k: v for k,v in merged_candidates.items() if len(v)>=richness} # select only clusters with richness >= 50
    center_shifted_candidates = merged_candidates.copy()

    # replace candidate center with brightest galaxy in cluster
    for mcenter, member_arr in merged_candidates.items():

        cluster_tree = cKDTree(member_arr[:,:2]) # for galaxies within a cluster
        bcg_idx = cluster_tree.query_ball_point(mcenter[:2], linear_to_angular_dist(0.5*virial_radius, mcenter[2]).value) # select galaxies within 0.25*virial radius
        bcg_arr = member_arr[bcg_idx] 

        if len(bcg_arr) and len(member_arr): # check if cluster empty or if there are no galaxies within 0.25*virial radius

            mag_sort = bcg_arr[bcg_arr[:,3].argsort()] # sort selected galaxies by abs mag (brightness)
            bcg = mag_sort[0] # brightest cluster galaxy (bcg)

            if (abs(bcg[3]) > abs(mcenter[3])) and (tuple(bcg) != mcenter): # if bcg brighter than current center, replace it as center
                center_shifted_candidates[tuple(bcg)] = member_arr
                center_shifted_candidates[mcenter] = np.array([])


    center_shifted_candidates = {k: v for k,v in center_shifted_candidates.items() if len(v)>=richness}
    final_candidates = {}

    # N(0.5) and galaxy overdensity
    for odcenter, member_arr in center_shifted_candidates.items():

        number_count = find_number_count(odcenter, member_arr, distance=0.5*u.Mpc/u.littleh) # find number count N(0.5)
        N_mean, N_rms = center_overdensity(odcenter, galaxy_data, max_velocity) # find overdensity D
        D = (odcenter[-1] - N_mean)/N_rms

        # Initialize the cluster only if N(0.5) >= 8 and D >= overdensity
        if D >= overdensity or number_count >= 8:
            final_candidates[odcenter] = member_arr
        else:
            final_candidates[odcenter] = np.array([])

    final_candidates = {k: v for k,v in final_candidates.items() if len(v)>=richness}

    if check_map: # for checking cluster shapes, locations
        fig = plt.figure(figsize=(10,8))
        for k, v in final_candidates.items():
            plt.scatter(v[:,0], v[:,1], s=5)
            plt.scatter(k[0], k[1])
            plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
            plt.show()

    return final_candidates


def three_sigma(cluster_center, cluster_members, n):
    '''
    Performs the 3sigma interloper removal method and removes galaxies beyond 1.5 h^-1 Mpc radius.

    Parameters
    ----------
    cluster_center: array-like
    Cluster center of cluster to be cleaned.

    cluster_members: array-like
    Array of cluster to be cleaned.

    n: int
    Number of bins.
    
    Returns
    -------
    cleaned_members: array-like
    Array of cleaned members with interlopers removed.
    '''

    # add two additonal columns to galaxy_data for peculiar velocity and radial distance
    N = cluster_members.shape
    temp = np.zeros((N[0], N[1]+2))
    temp[:,:-2] = cluster_members
    cluster_members = temp
    assert cluster_members.shape[1] == N[1]+2, 'Cluster member array is wrong shape.'

    # initialize velocity column with redshift-velocity formula
    cluster_center = cluster_center[0]
    cluster_members[:,-2] = (redshift_to_velocity(cluster_members[:,2]) - redshift_to_velocity(cluster_center[2]))/(1+cluster_center[2])

    # initialize radial distance column
    cluster_center_coord = SkyCoord(cluster_center[0]*u.deg, cluster_center[1]*u.deg)
    cluster_members_coord = SkyCoord(cluster_members[:,0]*u.deg, cluster_members[:,1]*u.deg)
    separations = cluster_center_coord.separation(cluster_members_coord)
    d_A = cosmo.angular_diameter_distance(cluster_center[2])
    cluster_members[:,-1] = (separations*d_A).to(u.Mpc, u.dimensionless_angles())

    # bin galaxies by radial distance from center
    cleaned_members = []
    s = cluster_members[:,-1]
    bins = np.linspace(min(s), max(s), n)
    digitized = np.digitize(s, bins) # bin members into n radial bins

    # distance_tree = cKDTree(cluster_members[:,:2])
    # dist_idx = distance_tree.query_ball_point(cluster_center[:2], linear_to_angular_dist(1.5*u.Mpc/u.littleh, cluster_center[2]).value) # select galaxies within 1.5 h^-1 Mpc
    # cluster_members = cluster_members[dist_idx]

    for i in range(1, len(bins)+1):
        bin_galaxies = cluster_members[np.where(digitized==i)] # galaxies in current bin
        assert len(bin_galaxies) == len(digitized[digitized==i])

        size_before = len(bin_galaxies)
        size_after = 0
        size_change = size_after - size_before

        while (size_change != 0) and (len(bin_galaxies) > 0): # repeat until convergence for each bin
            size_before = len(bin_galaxies)

            if len(bin_galaxies) > 1:
                bin_vel_arr = bin_galaxies[:,-2]
                sort_idx = np.argsort(np.abs(bin_vel_arr-np.mean(bin_vel_arr))) # sort by deviation from mean
                bin_galaxies = bin_galaxies[sort_idx]
                new_bin_galaxies = bin_galaxies[:-1,:] # do not consider galaxy with largest deviation
                new_vel_arr = new_bin_galaxies[:,-2]

                if (bin_galaxies[-1,-2] != 0.0):
                    if len(new_vel_arr) > 1:
                        bin_mean = np.mean(new_vel_arr) # velocity mean
                        bin_dispersion = sum((new_vel_arr-bin_mean)**2)/(len(new_vel_arr)-1) # velocity dispersion

                        if (abs(bin_vel_arr[-1] - bin_mean) >= 3*np.sqrt(bin_dispersion)): # if outlier velocity lies outside 3 sigma
                            bin_galaxies = new_bin_galaxies # remove outlier

            size_after = len(bin_galaxies)
            size_change = size_after - size_before
            
        cleaned_members.append(bin_galaxies[:,:-2])
    
    assert len(cleaned_members) == len(bins)
    cleaned_members = np.concatenate(cleaned_members, axis=0)

    return cleaned_members


def create_df(d, mode):
    '''Create two dataframes to store cluster information

    Parameters
    ----------
    d: dict (k:v -> bcg: cluster members)

    mode: str
    'before_interloper' is for data before interloper removal
    'after_interloper' is for data after interloper removal
    
    Returns
    -------
    bcg_df: pd.Dataframe
    df of cluster centers: ['ra', 'dec', 'redshift', 'brightness', 'LR', 'N(0.5)', 'total_N', 'gal_id', 'cluster_id']

    member_df: pd.Dataframe
    df of member galaxies: ['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'cluster_id']
    '''

    if mode == 'before_interloper':
        columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'LR', 'N(0.5)']

    elif mode == 'after_interloper':
        columns = ['ra', 'dec', 'redshift', 'brightness', 'LR', 'N(0.5)', 'total_N', 'gal_id', 'cluster_id']
        member_columns = ['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'cluster_id']
    
    else:
        return Exception('No mode stated')

    bcg_df = pd.DataFrame(columns=columns).fillna(0)
    member_df = pd.DataFrame(columns=columns).fillna(0)

    for i, (k,v) in enumerate(d.items()):
        bcg_df.loc[i, columns] = k
        bcg_df.loc[i, 'N(0.5)'] = find_number_count(k,v, distance=0.5*u.Mpc/u.littleh)
        bcg_df.loc[i, 'total_N'] = len(v)
        if mode == 'before_interloper':
            bcg_df.loc[i, 'cluster_id'] = i
        
        N = v.shape
        if mode == 'before_interloper':
            temp_v = np.zeros((N[0], N[1]+1))
            temp_v[:,:-1] = v
            temp_v[:,-1] = i
            temp = pd.DataFrame(data=temp_v, columns=columns+['cluster_id'])

        elif mode == 'after_interloper':
            temp_v = np.zeros((N[0], N[1]))
            temp_v = v
            temp = pd.DataFrame(data=temp_v, columns=member_columns)

        member_df = member_df.append(temp)
        
    
    bcg_df = bcg_df.sort_values('redshift')
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'LR', 'N(0.5)', 'total_N', 'gal_id', 'cluster_id']]
    member_df = member_df[['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'cluster_id']]

    return bcg_df, member_df


if __name__ == "__main__":

    # --- Select luminous galaxies from raw survey data
    # print('Selecting galaxy survey...')
    # conn = sqlite3.connect('galaxy_clusters.db')
    # df = pd.read_sql_query('''
    # SELECT *
    # FROM mag_lim
    # WHERE redshift BETWEEN 0.5 AND 2.52
    # ORDER BY redshift
    # ''', conn)

    # df = df.loc[:, ['ra','dec','redshift','RMag','ID', 'LR']] # select relevant columns
    # df = df.sort_values('redshift')

    # galaxy_arr = df.values
    # print('No. of galaxies to search through: {no}'.format(no=len(galaxy_arr)))
    # galaxy_arr, luminous_arr = luminous_search(galaxy_arr, max_velocity=2000)
    # # np.savetxt('COSMOS_galaxy_data.csv', galaxy_arr, delimiter=',')
    # np.savetxt('luminous_galaxy_redshift.csv', luminous_arr, delimiter=',')

    # # -- FoF
    richness = 25
    D = 4
    fname = 'derived_datasets\\R{r}_D{d}_0.02_1.5r\\'.format(r=richness, d=D)
    test = 'der'
    # galaxy_arr = np.loadtxt('derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',')
    # luminous_arr = np.loadtxt('derived_datasets\\luminous_galaxy_redshift.csv', delimiter=',')

    # galaxy_arr = galaxy_arr[(galaxy_arr[:,2] >= 0.5) & (galaxy_arr[:,2] <= 2.52)]
    # luminous_arr = luminous_arr[(luminous_arr[:,2] >= 0.5) & (luminous_arr[:,2] <= 2.5)]
    # print('Number of galaxies: {galaxy}, Number of candidates: {luminous}'.format(galaxy=len(galaxy_arr), luminous=len(luminous_arr)))

    # candidates = FoF(galaxy_arr, luminous_arr, max_velocity=2000, linking_length_factor=0.4, virial_radius=1.5*u.Mpc/u.littleh, richness=richness, overdensity=D, check_map=False)
    # print('{n} candidate clusters found.'.format(n=len(candidates)))

    # print('Exporting candidate clusters to csv...')
    # candidate_df, candidate_member_df = create_df(candidates, mode='before_interloper')
    # candidate_df.to_csv(fname+'candidate_bcg.csv', index=False)
    # candidate_member_df.to_csv(fname+'candidate_members.csv', index=False)


    # -- interloper removal
    # bcg_df = pd.read_csv(fname+'candidate_bcg.csv')
    # member_df = pd.read_csv(fname+'candidate_members.csv')

    # bcg_arr = bcg_df.values
    # arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
    # cleaned_clusters = {}
    # print('No. of candidates: {n}'.format(n=len(bcg_arr)))

    # print('Removing interlopers...')
    # for g in group_n:
    #     center = bcg_arr[bcg_arr[:,-1]==g]
    #     clusters = arr[arr[:,-1]==g]

    #     initial_size = len(clusters)
    #     cleaned_members = three_sigma(center, clusters, 10)
    #     if len(cleaned_members) < richness:
    #         cleaned_clusters[tuple(center[0])] = np.array([])
    #     else:
    #         cleaned_clusters[tuple(center[0])] = cleaned_members
    #     if g % 100 == 0:
    #         print('Interlopers removed from cluster {g}: {size_change}'.format(g=g, size_change=abs(len(cleaned_members)-initial_size)))

    # cleaned_clusters = {k:v for k,v in cleaned_clusters.items() if len(v) >= richness}
    # print('Number of clusters: {n} returned'.format(n=len(cleaned_clusters)))

    # print('Exporting cleaned candidate clusters to csv...')
    # cleaned_df, cleaned_member_df = create_df(cleaned_clusters, mode='after_interloper')


    # # -- extract required sample
    # print(len(cleaned_df), len(cleaned_member_df))

    # bcg_id = cleaned_df['cluster_id']
    # cleaned_member_df = cleaned_member_df.loc[cleaned_member_df['cluster_id'].isin(bcg_id)]
    # member_id = np.unique(cleaned_member_df['cluster_id'])
    # cleaned_df = cleaned_df.loc[cleaned_df['cluster_id'].isin(member_id)]
    # assert len(cleaned_df) == len(np.unique(cleaned_member_df['cluster_id']))

    # cleaned_df = cleaned_df[(cleaned_df['N(0.5)'] >= 8) & (cleaned_df['redshift'] >= 0.5) & (cleaned_df['redshift'] <= 2.5)]
    # bcg_id = cleaned_df['cluster_id']
    # cleaned_member_df = cleaned_member_df.loc[cleaned_member_df['cluster_id'].isin(bcg_id)]

    # assert len(cleaned_df) == len(np.unique(cleaned_member_df['cluster_id']))
    # print(len(cleaned_df), len(cleaned_member_df))

    # cleaned_df.to_csv(fname+'filtered_bcg.csv', index=False)
    # cleaned_member_df.to_csv(fname+'filtered_members.csv', index=False)

    # -- calculate and compare masses
    def test_masses(bcg_file, member_file, estimator, fname):
        bcg_df = pd.read_csv(bcg_file)
        member_df = pd.read_csv(member_file)

        bcg_arr = bcg_df.sort_values('cluster_id').values
        arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
        masses = np.zeros(group_n.shape)

        for i, g in enumerate(group_n):
            cluster = arr[arr[:,-1]==g]
            center = bcg_arr[bcg_arr[:,-1]==g]
            if estimator == 'virial':
                mass, _, _ = virial_mass_estimator(cluster[:,:3])
            if estimator == 'projected':
                mass = projected_mass_estimator(center, cluster)
            masses[i] = mass.value
        
        np.savetxt(fname+'{m}_masses.txt'.format(f=fname, m=estimator), masses)

        # masses = np.loadtxt(fname+'{f}_{m}_masses.txt'.format(f=fname, m=estimator))

        # i, = np.where(bcg_arr[:,2] < 2.50)
        # bcg_arr = bcg_arr[i,:]
        # masses = masses[i]

        # j, = np.where(masses<2e16)
        # bcg_arr = bcg_arr[j,:]
        # masses = masses[j]

        fig, ax = plt.subplots(figsize=(10,8))
        p = ax.scatter(bcg_arr[:,2], np.log10(masses), s=10, alpha=0.75)
        # sns.regplot(bcg_arr[:,2], np.log10(masses), lowess=True, marker='.', scatter_kws={'color': 'b', 'alpha': 0.75, 's':10}, line_kws={'color': 'red'})
        # cbar = fig.colorbar(p)

        ax.set_title('Mass against redshift')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Cluster Mass (log(M/Msun))')
        # cbar.ax.set_ylabel('Abs Mag')
        ax.ticklabel_format(style='sci', axis='y')
        ax.yaxis.major.formatter._useMathText = True

        # x = bcg_arr[:,2]
        # a,b,r,_,_ = linregress(x, masses)
        # ax.plot(x,a*x+b,'k--',alpha=0.75)
        # print(r)

        ax.set_xlim(0.5,2.5)
        # ax.set_ylim(0,6e15)
        plt.show()

    # test_masses(fname+'filtered_bcg.csv', fname+'filtered_members.csv', estimator='virial', fname=fname)

    def compare_masses(projected_masses, virial_masses):
        projected = np.loadtxt(projected_masses)
        virial = np.loadtxt(virial_masses)

        projected = projected[projected < 1e17]
        virial = virial[virial < 1e17]

        fig, ax = plt.subplots()
        ax.scatter(x=np.log10(virial), y=np.log10(projected), s=10, alpha=0.75)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]

        m, c, r, _, _, = linregress(np.log10(virial), np.log10(projected))
        print(m,c,r)
        ax.plot(lims, lims, 'k--', alpha=0.75)
        X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()))
        ax.plot(X, m*X+c, 'r--')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('virial masses')
        ax.set_ylabel('projected masses')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        
        plt.show()

    # compare_masses(fname+'projected_masses.txt', fname+'virial_masses.txt')


    # -- plotting clusters for manual checking
    def check_plots(bcg_df, member_df):
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s -  %(levelname)s -  %(message)s")
        logging.getLogger('matplotlib.font_manager').disabled = True

        
        bcg_df = pd.read_csv(bcg_df)
        member_df = pd.read_csv(member_df)
        bcg_arr = bcg_df.sort_values('redshift').values
        arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)

        bins = np.linspace(0.5,2.03,68) # bins of 0.03 width
        bins = np.arange(0.5,2.03,0.00666)
        digitized = np.digitize(bcg_arr[:,2], bins)

        for i in range(1,len(bins)):
            binned_data = bcg_arr[np.where(digitized==i)]
            logging.debug('Number of clusters in bin {i}: {length}'.format(i=i,length=len(binned_data)))

            if len(binned_data): # plot clusters for checking
                fig = plt.figure(figsize=(10,8))
                # plt.hist2d(x=arr[:,0], y=arr[:,1], bins=(100,80), cmap=plt.cm.Reds)
                for center in binned_data:
                    cluster_id = center[-1]
                    cluster = arr[arr[:,-1]==cluster_id]
                    plt.scatter(cluster[:,0], cluster[:,1], s=5)
                    plt.scatter(center[0], center[1])
                    plt.axis([min(arr[:,0]), max(arr[:,0]), min(arr[:,1]), max(arr[:,1])])
                    logging.debug('Plotting cluster: {cluster}'.format(cluster=center[0:2]))

                logging.debug('Plotting bin {i}. Clusters with binned redshift {redshift}'.format(i=i, redshift=bins[i]))
                plt.show()

    # check_plots(fname+'filtered_bcg.csv', fname+'filtered_members.csv')