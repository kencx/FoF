import pickle
import sqlite3
import numpy as np
import pandas as pd
from random import uniform
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from astropy import units as u
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology

from params import *
from cluster import Cluster
from plotting import check_plots
from analysis.methods import linear_to_angular_dist, mean_separation, redshift_to_velocity

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


# --- helper functions
def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
    arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
    arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
    S = np.setdiff1d(arr_1_struc, arr_2_struc)
    return S


def find_number_count(center, galaxies, distance):
    if isinstance(center, Cluster):
        coords = [center.ra, center.dec]
        z = center.z
    else:
        coords = center[:2]
        z = center[2]

    N_tree = cKDTree(galaxies[:,:2])
    n_idx = N_tree.query_ball_point(coords, linear_to_angular_dist(distance, z).value)
    n_arr = galaxies[n_idx]
    return len(n_arr)


def center_overdensity(center, galaxy_data, max_velocity): # calculate overdensity of cluster

    # select 300 random points (RA and dec)
    n = 300
    ra_random = np.random.uniform(low=min(galaxy_data[:,0]), high=max(galaxy_data[:,0]), size=n)
    dec_random = np.random.uniform(low=min(galaxy_data[:,1]), high=max(galaxy_data[:,1]), size=n)
    points = np.vstack((ra_random, dec_random)).T
    assert points.shape == (n,2)

    # select all galaxies within max velocity
    velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center.z))/(1+center.z) <= max_velocity*u.km/u.s]
    # velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center.z) <= 0.02*(1+center.z)]
    virial_tree = cKDTree(velocity_bin[:,:2])

    # find the N(0.5) mean and rms for the 300 points
    N_arr = np.zeros(n)
    for i, p in enumerate(points):
        idx = virial_tree.query_ball_point(p, linear_to_angular_dist(0.5*u.Mpc/u.littleh, center.z).value)
        N_arr[i] = len(idx)

    N_mean = np.mean(N_arr)
    N_rms = np.sqrt(np.mean(N_arr**2)) #rms
    D = (center.N - N_mean)/N_rms
    return D


# --- search algorithms
def luminous_search(galaxy_data, max_velocity, path='analysis\\derived_datasets\\'):
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

    if isinstance(galaxy_data, pd.DataFrame):
        galaxy_data = galaxy_data.values
    print(f'No. of galaxies to search through: {len(galaxy_data)}')

    # add additional N(0.5) column to galaxy_data
    main_arr = np.column_stack([galaxy_data, np.zeros(galaxy_data.shape[0])])
    assert galaxy_data.shape[1]+1 == main_arr.shape[1], 'Galaxy array is wrong shape.'

    # count N(0.5) for each galaxy
    for i, galaxy in enumerate(main_arr): 

        # vel_cutoff = main_arr[abs(redshift_to_velocity(main_arr[:,2])-redshift_to_velocity(galaxy[2]))/(1+galaxy[2]) <= max_velocity*u.km/u.s] # select galaxies within appropriate redshift range
        vel_cutoff = main_arr[abs(main_arr[:,2]-galaxy[2]) <= 0.02*(1+galaxy[2])]

        if (len(vel_cutoff) >= 8):
            main_arr[i,-1] = find_number_count(galaxy, vel_cutoff, 0.5*u.Mpc/u.littleh) # galaxies within 0.5 h^-1 Mpc

    luminous_galaxy = main_arr[main_arr[:,-1] >= 8] # select for galaxies with N(0.5) >= 8
    luminous_galaxy = luminous_galaxy[luminous_galaxy[:,-1].argsort()[::-1]] # sort by N(0.5) in desc
    assert luminous_galaxy[0,-1] == max(luminous_galaxy[:,-1]), 'Galaxy array is not sorted by N(0.5)'

    print(f'Total Number of Galaxies in sample: {len(main_arr)}, Number of candidate centers: {len(luminous_galaxy)}')

    np.savetxt(path+'COSMOS_galaxy_data.csv', main_arr, delimiter=',')
    np.savetxt(path+'luminous_galaxy_redshift.csv', luminous_arr, delimiter=',')

    return main_arr, luminous_galaxy


def new_FoF(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, richness, overdensity):

    candidates = []
    sep_arr = []
    tracker = np.ones(len(luminous_data)) # identifies galaxies that have been included

    # identify cluster candidates
    for i, center in enumerate(luminous_data): # each row is a candidate center to search around

        if tracker[i]:
            velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s] # select galaxies within max velocity
            # velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center[2]) <= 0.02*(1+center[2])]

            # cKDTree uses euclidean distance? Find a way to search angular distances instead or convert to cartesian coords
            virial_tree = cKDTree(velocity_bin[:,:2])
            transverse_virial_idx = virial_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value)
            transverse_virial_points = velocity_bin[transverse_virial_idx] # select galaxies within virial radius

            if len(transverse_virial_points) >= 12: # reject if less than 12 surrounding galaxies within virial radius (to save time) 
                mean_sep = mean_separation(len(transverse_virial_points), virial_radius) # in h^-1 Mpc
                transverse_linking_length = linking_length_factor*(mean_sep.to(u.Mpc, u.with_H0(cosmo.H0))).value # determine transverse LL from local mean separation

                linking_tree = cKDTree(transverse_virial_points[:,:2])
                link_idx = linking_tree.query_ball_point(center[:2], transverse_linking_length) # select galaxies within linking length
                f_points = transverse_virial_points[link_idx]
                # sep_arr.append([mean_sep.value, center[2]])

                member_galaxies = f_points

                for f in f_points[:,:2]:
                    fof_idx = linking_tree.query_ball_point(f, transverse_linking_length) # select galaxies within 2 linking lengths (FoF)
                    fof_points = transverse_virial_points[fof_idx]

                    mask = np.isin(fof_points, member_galaxies, invert=True) # filter for points not already accounted for
                    vec_mask = np.isin(mask.sum(axis=1), center.shape[0])
                    fof_points = fof_points[vec_mask].reshape((-1,center.shape[0])) # points of 2 linking lengths (FoF)
                    if len(fof_points):
                        member_galaxies = np.concatenate((member_galaxies, fof_points)) # merge all FoF points within 2 linking lengths

                if len(member_galaxies) >= richness: # must have >= richness (Abell richness criterion)
                    c = Cluster(center, member_galaxies)
                    candidates.append(c)

                    if not i % 100:
                        # print(f'{i} ' + c.__str__())
                        logging.info(f'{i} ' + c.__str__())

                    # locate centers within member_galaxies (centers of interest)
                    member_gal_id = member_galaxies[:,4]
                    luminous_gal_id = luminous_data[:,4]
                    coi, _, coi_idx = np.intersect1d(member_gal_id, luminous_gal_id, return_indices=True)

                    # update tracker to 0 for these points
                    for i in coi_idx:
                        tracker[i] = 0

            # if len(candidates) >= 100: # for quick testing
            #     break

    # check_plots(candidates)
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
        velocity_bin = candidate_centers[abs(redshift_to_velocity(candidate_centers[:,2])-redshift_to_velocity(center.z))/(1+center.z) <= max_velocity*u.km/u.s] # select galaxies within max velocity
        # velocity_bin = candidate_centers[abs(candidate_centers[:,2]-center.z) <= 0.02*(1+center.z)]
        center_tree = cKDTree(velocity_bin[:,:2])

        c_coords = [center.ra, center.dec]
        virial_center_idx = center_tree.query_ball_point(c_coords, linear_to_angular_dist(virial_radius, center.z).value)
        centers_within_vr = velocity_bin[virial_center_idx] 

        # merge each overlapping cluster
        if len(centers_within_vr):
            for c in centers_within_vr:
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
        cluster_tree = cKDTree(center.galaxies[:,:2]) # for galaxies within a cluster
        c_coords = [center.ra, center.dec]
        bcg_idx = cluster_tree.query_ball_point(c_coords, linear_to_angular_dist(0.25*virial_radius, center.z).value)
        bcg_arr = center.galaxies[bcg_idx] 

        if len(bcg_arr) and len(center.galaxies): # check there are any galaxies within 0.25*virial radius

            mag_sort = bcg_arr[bcg_arr[:,3].argsort()] # sort selected galaxies by abs mag (brightness)
            mask = np.isin(mag_sort[:,4], center_shifted_gal_id_space, invert=True) # filter for galaxies that are not existing centers
            mag_sort = mag_sort[mask]
            
            if len(mag_sort):
                bcg = mag_sort[0] # brightest cluster galaxy (bcg)

                # if bcg brighter than current center, replace it as center
                if (abs(bcg[3]) > abs(center.bcg_brightness)) and (bcg[4] != center.gal_id):
                    new_cluster = Cluster(bcg, center.galaxies) # initialize new center

                    # new_center = [c for c in center_shifted_candidates if c.gal_id == center.gal_id][0]
                    # new_center.galaxies = np.array([]) # delete old center

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
    member_galaxies[:,-2] = (redshift_to_velocity(member_galaxies[:,2]) - redshift_to_velocity(center.z))/(1+center.z)

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

    # remove galaxies further than virial radius
    # distance_tree = cKDTree(member_galaxies[:,:2])
    # dist_idx = distance_tree.query_ball_point(center[:2], linear_to_angular_dist(1.5*u.Mpc/u.littleh, center[2]).value) # select galaxies within 1.5 h^-1 Mpc
    # member_galaxies = member_galaxies[dist_idx]

    for i in range(1, len(bins)+1):
        bin_galaxies = member_galaxies[np.where(digitized==i)] # galaxies in current bin

        size_before = len(bin_galaxies)
        size_after = 0
        size_change = size_after - size_before
        # assert size_before <= 1, 'Too many bins.'

        while (size_change != 0) and (len(bin_galaxies) > 0): # repeat until convergence for each bin
            size_before = len(bin_galaxies)

            if len(bin_galaxies) > 1:
                bin_vel_arr = bin_galaxies[:,-2]
                sort_idx = np.argsort(np.abs(bin_vel_arr-np.mean(bin_vel_arr))) # sort by deviation from mean
                bin_galaxies = bin_galaxies[sort_idx]
                new_bin_galaxies = bin_galaxies[:-1,:] # do not consider galaxy with largest deviation
                new_vel_arr = new_bin_galaxies[:,-2]
                ignored_galaxy = bin_galaxies[-1,:]
                # assert ignored_galaxy[-1] == 0.0, 'Galaxy with largest deviation is the cluster center'

                if (bin_galaxies[-1,-1] != 0.0): # if ignored galaxy is not center
                    if len(new_vel_arr) > 1:
                        bin_mean = np.mean(new_vel_arr) # velocity mean
                        bin_dispersion = sum((new_vel_arr-bin_mean)**2)/(len(new_vel_arr)-1) # velocity dispersion

                        if (abs(ignored_galaxy[-2] - bin_mean) >= 3*np.sqrt(bin_dispersion)): # if outlier velocity lies outside 3 sigma
                            bin_galaxies = new_bin_galaxies # remove outlier

            size_after = len(bin_galaxies)
            size_change = size_after - size_before
            
        cleaned_members.append(bin_galaxies[:,:-2])
    
    assert len(cleaned_members) == len(bins)
    cleaned_members = np.concatenate(cleaned_members, axis=0) # flatten binned arrays
    return cleaned_members


def interloper_removal(data):

    print(f'Removing interlopers from {len(data)} clusters')
    for i, c in enumerate(data):
        initial_size = len(c.galaxies)
        cleaned_galaxies = three_sigma(c, c.galaxies, 10)
        c.galaxies = cleaned_galaxies
    
        size_change = initial_size - len(cleaned_galaxies)
        if i % 100 == 0:
            print(f'Interlopers removed from cluster {i}: {size_change}')

    cleaned_candidates = [c for c in data if c.richness >= richness]
    print(f'Number of clusters: {len(cleaned_candidates)} returned')
    return cleaned_candidates



if __name__ == "__main__":

#%%
    ###########################
    # IMPORT RAW DATA FROM DB #
    ###########################
    # print('Selecting galaxy survey...')
    # conn = sqlite3.connect('processing\\datasets\\galaxy_clusters.db')
    # df = pd.read_sql_query('''
    #     SELECT *
    #     FROM mag_lim
    #     WHERE redshift BETWEEN ? AND ?
    #     ORDER BY redshift''', conn, params=(min_redshift, max_redshift+0.2))

    # df = df.loc[:, ['ra','dec','redshift','RMag','ID', 'LR']] # select relevant columns
    # df = df.sort_values('redshift')


    #############################
    # LUMINOUS CANDIDATE SEARCH #
    #############################
    # main_arr, luminous_arr = luminous_search(df, max_velocity=2000)


#%%
    ######################
    # FRIENDS-OF-FRIENDS #
    ######################
    main_arr = np.loadtxt('analysis\\derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',') 
    luminous_arr = np.loadtxt('analysis\\derived_datasets\\luminous_galaxy_velocity.csv', delimiter=',')

    main_arr = main_arr[(main_arr[:,2] >= min_redshift) & (main_arr[:,2] <= max_redshift + 0.02)]
    luminous_arr = luminous_arr[(luminous_arr[:,2] >= min_redshift) & (luminous_arr[:,2] <= max_redshift)]
    print(f'Number of galaxies: {len(main_arr)}, Number of candidates: {len(luminous_arr)}')

    candidates = new_FoF(main_arr, luminous_arr, max_velocity=max_velocity, linking_length_factor=linking_length_factor, 
                         virial_radius=virial_radius, richness=richness, overdensity=D)
    print(f'{len(candidates)} candidate clusters found.')

    # pickle candidates list
    with open(fname+'candidates.dat', 'wb') as f:
        pickle.dump(candidates, f)

#%%
    ######################
    # INTERLOPER REMOVAL #
    ######################
    with open(fname+'candidates.dat', 'rb') as f:
        candidates = pickle.load(f)

    cleaned_candidates = interloper_removal(candidates)

    with open(fname+'cleaned_candidates.dat', 'wb') as f:
        pickle.dump(cleaned_candidates, f)


    # save all attributes into df 