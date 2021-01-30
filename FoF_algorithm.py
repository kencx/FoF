# import time
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.spatial import cKDTree, distance
from scipy.stats import norm, linregress

from astropy import units as u
from astropy.constants import G
from astropy.cosmology import WMAP5 as cosmo

from mass_estimator import virial_mass_estimator, redshift_to_velocity, projected_mass_estimator
from data_processing import split_df_into_groups, add_to_db


def linear_to_angular_dist(distance, photo_z):
    '''
    Converts proper distance (Mpc) to angular distance (deg). Used to find angular separations within clusters.

    Parameters
    ----------
    distance: float, array-like
        Distance in Mpc

    photo_z: float, array-like
        Associated redshift

    Returns
    -------
    d: float, array-like
        Angular distance in deg
    '''

    return ((distance*u.Mpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


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


def find_richness(point, member_galaxies, distance):
    richness_tree = cKDTree(member_galaxies[:,:2])
    r_idx = richness_tree.query_ball_point(point[:2], linear_to_angular_dist(distance, point[2]).value)
    r_arr = member_galaxies[r_idx]
    return len(r_arr)


def luminous_search(galaxy_data, max_velocity):
    '''
    Parameters
    ----------
    galaxy_data: array-like
        Galaxy data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'LR', 'doppler_velocity']

    max_velocity: float
        Largest velocity galaxy can have with respect to cluster centre in km/s.

    Returns
    -------
    galaxy_data: array-like
        Galaxy data with additional column of N(0.5). N(0.5) is the number of galaxies within 0.5Mpc of the galaxy.

    luminous_galaxy: array-like
        Sample of galaxy_data with galaxies of N(0.5) >= 8 and sorted by decreasing N(0.5). 
    '''
    
    # add additional column to galaxy_data
    N = galaxy_data.shape
    temp = np.zeros((N[0], N[1]+1))
    temp[:,:-1] = galaxy_data
    galaxy_data = temp

    assert galaxy_data.shape[1] == N[1]+1, 'Galaxy array is wrong shape.'
    
    for i, galaxy in enumerate(galaxy_data): # count N(0.5) for each galaxy
        vel_cutoff = galaxy_data[abs(galaxy_data[:,-2]-galaxy[-2]) <= max_velocity] # select for galaxies within max_velocity

        if len(vel_cutoff) > 1:
            galaxy_tree = cKDTree(vel_cutoff[:,:2])
            N_idx = galaxy_tree.query_ball_point(galaxy[:2], linear_to_angular_dist(0.5, galaxy[2]).value) # select for galaxies within 0.5Mpc
            galaxy_data[i,-1] = len(N_idx)
        else:
            galaxy_data[i,-1] = 0

    galaxy_data = galaxy_data[galaxy_data[:,-1].argsort()[::-1]] # sort by N(0.5)
    luminous_galaxy = galaxy_data[galaxy_data[:,-1] >= 8] # select for galaxies with N(0.5) >= 8 (this is our center sample)
    
    assert galaxy_data[0,-1] == max(galaxy_data[:,-1]), 'Galaxy array is not sorted by N(0.5)'
    assert len(luminous_galaxy[luminous_galaxy[:,-1] < 8]) == 0, 'Luminous galaxy array has galaxies with N(0.5) < 8'

    print(len(galaxy_data), len(luminous_galaxy))

    return galaxy_data, luminous_galaxy


def FoF(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, check_map=True):
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
        Galaxy data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'doppler_velocity', 'N(0.5)']
    
    luminous_data: array-like 
        Cluster center candidate data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'doppler_velocity', 'N(0.5)']
    
    max_velocity: float
        Largest velocity galaxy can have with respect to cluster centre in km/s
    
    linking_length_factor: float
        Factor for transverse linking length (0 < x < 1)
    
    virial_radius: float 
        Maximum radial boundary of cluster in Mpc
    
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

    for i, center in enumerate(luminous_data):
        velocity_cutoff = galaxy_data[abs(galaxy_data[:,-2]-center[-2]) <= max_velocity] # select for galaxies within max_velocity of candidate center

        virial_tree = cKDTree(velocity_cutoff[:,:2])
        transverse_virial_idx = virial_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value)
        transverse_virial_points = velocity_cutoff[transverse_virial_idx] # galaxies within virial radius of 2 Mpc

        if len(transverse_virial_points):
            mean_sep = mean_separation(len(transverse_virial_points), virial_radius) # in Mpc
            transverse_linking_length = linking_length_factor*mean_sep

            linking_tree = cKDTree(transverse_virial_points[:,:2]) 
            link_idx = linking_tree.query_ball_point(center[:2], transverse_linking_length) # select all galaxies within linking length
            f_points = transverse_virial_points[link_idx] # points within 1 linking length

            for f in f_points[:,:2]:
                fof_idx = linking_tree.query_ball_point(f, transverse_linking_length) # select for galaxies within 2 linking lengths (FoF)
                fof_points = transverse_virial_points[fof_idx]

                mask = np.isin(fof_points, f_points, invert=True) # filter for points not already accounted for
                vec_mask = np.isin(mask.sum(axis=1), center.shape[0])
                fof_points = fof_points[vec_mask].reshape((-1,center.shape[0])) # points of 2 linking lengths (FoF)
                if len(fof_points):
                    f_points = np.concatenate((f_points, fof_points)) # merge all FoF points within 2 linking lengths

            if len(f_points) >= 50: # must have >= 50 points
                candidates[tuple(center)] = f_points # dictionary {galaxy centre: array of FoF points}
                if not i % 100:
                    print('Candidate {num} identified with {length} members'.format(num=i, length=len(f_points)))

        # if len(candidates) >= 50:
        #     break
        

    # if check_map:    
    #     fig = plt.figure(figsize=(10,8))
    #     for k, v in candidates.items():
    #         plt.scatter(v[:,0], v[:,1], s=5)
    #         plt.scatter(k[0], k[1])
    #     plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    #     plt.show()

    '''
    Remove overlapping clusters

    1. For each candidate center, search for other centers within virial radius
    2. Find the set difference of member galaxies within the 2 centers, where the center with larger N(0.5) remains the primary (larger) cluster. If both centers have equal N(0.5), the brighter center is chosen.
    3. Secondary centers have their centers redefined to a new centroid of the smaller cluster
    4. Combine clusters with centers 0.25 x virial radius from each other. Likewise, the center with larger N(0.5) remains as the primary cluster 
    5. Search for member galaxies brighter than candidate cluster center within 0.25 x virial radius
    6. Replace the brightest galaxy as the new cluster center
    7. Initialize the cluster only if richness >= 8 (richness is defined as the number of member galaxies within 0.25*virial radius)
    '''

    def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
        arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('LR', np.float64), ('doppler_velocity', np.float64), ('N(0.5)', np.float64)])
        arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('LR', np.float64), ('doppler_velocity', np.float64), ('N(0.5)', np.float64)])
        S = np.setdiff1d(arr_1_struc, arr_2_struc)
        return S

    def new_centroid(member_galaxies):  # new center is defined by member galaxy that is the minimal distance away from all other member galaxies
        center = [np.mean(member_galaxies[:,0]), np.mean(member_galaxies[:,1])]
        member_tree = cKDTree(member_galaxies[:,:2])
        distances, idx = member_tree.query(center, k=1) # search for k=1 nearest neighbour in member_galaxies
        centroid = member_galaxies[idx]
        return centroid

    def remove_overlap(merged_candidates, center, arr1, arr2):
        S = setdiff2d(arr1, arr2) # finds common rows between two arrays and return the rows of cen_arr not present in member_arr
        if len(S):
            member_galaxies = S.view(np.float64).reshape(S.shape + (-1,)) # convert back to np array of original shape
            new_center = new_centroid(member_galaxies) # locate new center for candidate
            merged_candidates[tuple(new_center)] = member_galaxies # initialize new center
            merged_candidates[center] = np.array([]) # delete old center

    def merge(merged_candidates, center1, center2):
        merged_candidates[center1] = np.concatenate((merged_candidates[center2], merged_candidates[center1]), axis=0) # combine the arrays
        arr, uniq_count = np.unique(merged_candidates[center1], axis=0, return_counts=True)
        merged_candidates[center1] = arr[uniq_count==1] # ensure all points are unique
        merged_candidates[center2] = np.array([]) # delete all galaxies in smaller merged galaxy


    cluster_centers = np.array([k for k in candidates.keys()]) # array of cluster centers
    merged_candidates = candidates.copy()

    for center, member_arr in candidates.items():

        vel_cutoff = cluster_centers[abs(cluster_centers[:,-2] - center[-2]) <= max_velocity]
        center_tree = cKDTree(vel_cutoff[:,:2])

        virial_center_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value) # search for centers within virial radius
        centers_within_vr = vel_cutoff[virial_center_idx]

        # remove overlapping points
        if len(centers_within_vr):
            for cen in centers_within_vr:
                cen = tuple(cen)
                if center == cen:
                    continue

                cen_arr = merged_candidates[cen]
                if len(cen_arr) and len(member_arr): # check not empty
                    
                    # compare size of two arrays and removes common points from smaller array. if both are same size, remove common points from less bright centre
                    if center[-1] > cen[-1]: 
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)

                    elif center[-1] < cen[-1]:
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)
                    
                    elif abs(center[3]) < abs(cen[3]):
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)
                    
                    else:
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)


        merge_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(0.25*virial_radius, center[2]).value) # search for centers within 0.25*virial radius
        merge_centers = vel_cutoff[merge_idx] # array of centers within 0.25*virial radius, to be merged with center galaxy

        # merge cluster centers within 0.25*virial radius from current center
        if len(merge_centers): # check not empty
            for mg in merge_centers:
                mg = tuple(mg)

                if center == mg:
                    continue

                if ((mg in merged_candidates.keys())       # check if galaxy to be merged is still in candidate list
                    and (len(merged_candidates[mg]))       # check if galaxy to be merged has member galaxies
                    and (len(merged_candidates[center]))): # check if center galaxy has member galaxies
                    
                    # compare size of two arrays, keeps larger array with merged points and deletes smaller one. if both are same size, compare brightness instead
                    if mg[-1] > center[-1]:
                        merge(merged_candidates, mg, center)

                    elif center[-1] > mg[-1]:
                        merge(merged_candidates, center, mg)

                    elif abs(mg[3]) > abs(center[3]):
                        merge(merged_candidates, mg, center)

                    else:
                        merge(merged_candidates, center, mg)

    # merged_candidates = {k: v for k,v in merged_candidates.items() if len(v)}

    # fig = plt.figure(figsize=(10,8))
    # count = 0
    # for k, v in merged_candidates.items():
    #     plt.scatter(v[:,0], v[:,1], s=5)
    #     plt.scatter(k[0], k[1])
    #     count += 1
    #     if count % 5 == 0:
    #         plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    #         plt.show()

    merged_cluster_centers = np.array([k for k in merged_candidates.keys()])
    merged_candidates = {k: v for k,v in merged_candidates.items() if len(v)>=50}
    center_shifted_candidates = merged_candidates.copy()

    for mcenter, member_arr in merged_candidates.items():

        # vel_cutoff = merged_cluster_centers[abs(merged_cluster_centers[:,-2] - mcenter[-2]) <= max_velocity]
        center_tree = cKDTree(member_arr[:,:2])

        bcg_idx = center_tree.query_ball_point(mcenter[:2], linear_to_angular_dist(0.25*virial_radius, mcenter[2]).value) # search for centers within 0.25*virial radius
        bcg_arr = member_arr[bcg_idx] 

        if len(bcg_arr) and len(member_arr):
            mag_sort = bcg_arr[bcg_arr[:,3].argsort()]
            bcg = mag_sort[0]
            if (abs(bcg[3]) > abs(mcenter[3])) and (tuple(bcg) != mcenter): # if bcg brighter than current center, replace it as center
                center_shifted_candidates[tuple(bcg)] = member_arr
                center_shifted_candidates[mcenter] = np.array([])

    center_shifted_candidates = {k: v for k,v in center_shifted_candidates.items() if len(v)>=50}
    final_candidates = {}
    
    for fcenter, member_arr in center_shifted_candidates.items():
        richness = find_richness(fcenter, member_arr, distance=0.25*virial_radius)

        if richness < 8:  # Initialize the cluster only if richness >= 8 (richness is defined as the number of member galaxies within 0.25*virial radius)
            final_candidates[fcenter] = np.array([]) 
        else:
            final_candidates[fcenter + (richness,)] = member_arr

    final_candidates = {k: v for k,v in final_candidates.items() if len(v)>=50}

    if check_map:
        count = 0
        fig = plt.figure(figsize=(10,8))
        for k, v in final_candidates.items():
            plt.scatter(v[:,0], v[:,1], s=5)
            plt.scatter(k[0], k[1])
            count += 1
            if count % 5 == 0:
                plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
                plt.show()
        # plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
        # plt.show()

    return final_candidates


def three_sigma(cluster_center, cluster_members, n):
    '''
    Performs the 3sigma interloper removal method and removes galaxies beyond 2 Mpc virial radius.

    Parameters
    ----------
    cluster_members: array-like
    Array of cluster to be cleaned.

    n: int
    Number of bins.
    
    Returns
    -------
    cleaned_members: array-like
    Array of cleaned members with interlopers removed.
    '''

    cluster_center = cluster_center[0]
    cluster_members = cluster_members[cluster_members[:,-1].argsort()] # sort by doppler velocity
    assert min(cluster_members[:,-1]) == cluster_members[0,-1]

    distance_tree = cKDTree(cluster_members[:,:2])
    dist_idx = distance_tree.query_ball_point(cluster_center[:2], linear_to_angular_dist(2, cluster_center[2]).value)
    cluster_members = cluster_members[dist_idx]
    
    cleaned_members = []
    velocity_arr = cluster_members[:,-1]
    bins = np.linspace(min(velocity_arr), max(velocity_arr), n)
    digitized = np.digitize(cluster_members[:,-2], bins) # bin members into n velocity bins

    for i in range(1, len(bins)+1):
        bin_galaxies = cluster_members[np.where(digitized==i)] # galaxies in current bin
        assert len(bin_galaxies) == len(digitized[digitized==i])

        if len(bin_galaxies) > 1:
            size_before = len(bin_galaxies)
            size_after = 0
            size_change = size_after - size_before

            while (size_change != 0) and (len(bin_galaxies) > 0): # repeat until convergence for each bin
                size_before = len(bin_galaxies)

                bin_vel_arr = bin_galaxies[:,-1]

                exclude_idx = np.argmax(abs(bin_vel_arr - np.mean(bin_vel_arr))) # exclude most outlying galaxy
                new_vel_arr = np.delete(bin_vel_arr, exclude_idx)

                if len(new_vel_arr) > 1:
                    bin_mean = np.mean(new_vel_arr) # velocity mean
                    bin_dispersion = sum((new_vel_arr-bin_mean)**2)/(len(new_vel_arr)-1) # velocity dispersion

                    bin_galaxies = bin_galaxies[abs(bin_galaxies[:,-1] - bin_mean) <= 3*np.sqrt(bin_dispersion)] # remove galaxies outside 3sigma

                size_after = len(bin_galaxies)
                size_change = size_after - size_before
            
        cleaned_members.append(bin_galaxies)
    
    assert len(cleaned_members) == len(bins)
    cleaned_members = np.concatenate(cleaned_members, axis=0)

    return cleaned_members


def create_candidate_df(d):
    '''Create two dataframes to store cluster information

    Parameters
    ----------
    d: dict (k:v -> bcg: cluster members)
    
    Returns
    -------
    bcg_df: pd.Dataframe
    df of cluster centers: ['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'gal_id', 'cluster_id']

    member_galaxy_df: pd.Dataframe
    df of member galaxies: ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'cluster_id']
    '''

    columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'LR', 'doppler_velocity', 'N(0.5)']
    bcg_df = pd.DataFrame(columns=columns+['richness']).fillna(0)
    member_galaxy_df = pd.DataFrame(columns=columns).fillna(0)

    for i, (k,v) in enumerate(d.items()):
        bcg_df.loc[i, columns+['richness']] = k
        bcg_df.loc[i, 'cluster_id'] = i
        bcg_df.loc[i, 'total_N'] = len(v)
        
        N = v.shape
        temp_v = np.zeros((N[0], N[1]+1))
        temp_v[:,:-1] = v
        temp_v[:,-1] = i
        temp = pd.DataFrame(data=temp_v, columns=columns+['cluster_id'])
        member_galaxy_df = member_galaxy_df.append(temp)
    
    bcg_df = bcg_df.sort_values('redshift')
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']]
    member_galaxy_df = member_galaxy_df[['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']]

    return bcg_df, member_galaxy_df


def create_cleaned_df(d):
    columns = ['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']
    member_columns = ['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']

    bcg_df = pd.DataFrame(columns=columns).fillna(0)
    member_df = pd.DataFrame(columns=member_columns).fillna(0)

    for i, (k,v) in enumerate(d.items()):
        bcg_df.loc[i, columns] = k
        bcg_df.loc[i, 'cluster_id'] = i
        bcg_df.loc[i, 'total_N'] = len(v)
        bcg_df.loc[i, 'richness'] = find_richness(k, v, distance=0.5)

        N = v.shape
        temp_v = np.zeros((N[0], N[1]))
        temp_v = v
        temp = pd.DataFrame(data=temp_v, columns=member_columns)
        member_df = member_df.append(temp)

    bcg_df = bcg_df.sort_values('redshift')
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']]
    member_df = member_df[['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'LR', 'gal_id', 'doppler_velocity', 'cluster_id']]

    return bcg_df, member_df


if __name__ == "__main__":

    # --- cluster search
    # print('Selecting galaxy survey...')
    # conn = sqlite3.connect('galaxy_clusters.db')
    # df = pd.read_sql_query('''
    # SELECT *
    # FROM mag_lim
    # ORDER BY redshift
    # ''', conn)

    # df = df.loc[:, ['ra','dec','redshift','RMag','ID', 'LR']] # select relevant columns
    # df['Doppler_vel'] = redshift_to_velocity(df['redshift'])
    # df = df.sort_values('redshift')

    # galaxy_arr = df.values
    # galaxy_arr, luminous_arr = luminous_search(galaxy_arr, max_velocity=2000)
    # np.savetxt('raw_galaxy2.csv', galaxy_arr, delimiter=',')
    # np.savetxt('luminous_galaxy2.csv', luminous_arr, delimiter=',')

    # galaxy_arr = np.loadtxt('raw_galaxy2.csv', delimiter=',')
    # luminous_arr = np.loadtxt('luminous_galaxy2.csv', delimiter=',')

    # galaxy_arr = galaxy_arr[(galaxy_arr[:,2] > 0.5) & (galaxy_arr[:,2] <= 2.05)]
    # luminous_arr = luminous_arr[(luminous_arr[:,2] > 0.5) & (luminous_arr[:,2] <= 2.05)]
    # print('Number of galaxies: {galaxy}, Number of candidates: {luminous}'.format(galaxy=len(galaxy_arr), luminous=len(luminous_arr)))

    # candidates = FoF(galaxy_arr, luminous_arr, max_velocity=2000, linking_length_factor=0.4, virial_radius=2, check_map=False)
    # print('{n} candidate clusters found.'.format(n=len(candidates)))

    # print('Exporting candidate clusters to csv...')
    # candidate_df, candidate_member_df = create_candidate_df(candidates)
    # fname = 'test2'
    # candidate_df.to_csv(fname + '_candidate_bcg.csv', index=False)
    # candidate_member_df.to_csv(fname + '_candidate_members.csv', index=False)

    # -- interloper removal
    # bcg_df = pd.read_csv('test2_candidate_bcg.csv')
    # member_df = pd.read_csv('test2_candidate_members.csv')

    # bcg_arr = bcg_df.values
    # arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
    # cleaned_clusters = {}

    # print('Removing interlopers...')
    # for g in group_n:
    #     center = bcg_arr[bcg_arr[:,-1]==g]
    #     clusters = arr[arr[:,-1]==g]

    #     initial_size = len(clusters)
    #     cleaned_members = three_sigma(center, clusters, 10)
    #     if len(cleaned_members) < 50:
    #         cleaned_members = np.array([])
    #     if g % 500 == 0:
    #         print('Interlopers removed from cluster {g}: {size_change}'.format(g=g, size_change=abs(len(cleaned_members)-initial_size)))

    #     cleaned_clusters[tuple(center[0])] = cleaned_members
    
    # cleaned_clusters = {k:v for k,v in cleaned_clusters.items() if len(v) >= 50}
    # print('Number of clusters: {n}'.format(n=len(cleaned_clusters)))

    # print('Exporting cleaned candidate clusters to csv...')
    # cleaned_df, cleaned_member_df = create_cleaned_df(cleaned_clusters)

    # cleaned_df.to_csv('cleaned_bcg.csv', index=False)
    # cleaned_member_df.to_csv('cleaned_members.csv', index=False)

    # add_to_db(cleaned_df, '0-3_interloper_removed_bcg')
    # add_to_db(cleaned_member_df, '0-3_interloper_removed_members')


    # -- extract required sample
    # bcg_df = pd.read_csv('cleaned_bcg.csv')
    # member_df = pd.read_csv('cleaned_members.csv')
    # print(len(bcg_df), len(member_df))
    # bcg_id = bcg_df['cluster_id']
    # member_df = member_df.loc[member_df['cluster_id'].isin(bcg_id)]
    # member_id = np.unique(member_df['cluster_id'])
    # bcg_df = bcg_df.loc[bcg_df['cluster_id'].isin(member_id)]
    # assert len(bcg_df) == len(np.unique(member_df['cluster_id']))

    # bcg_df = bcg_df[(bcg_df['richness'] >= 8) & (bcg_df['redshift'] <= 2.05) & (bcg_df['redshift'] >= 0.5)]
    # bcg_id = bcg_df['cluster_id']
    # member_df = member_df.loc[member_df['cluster_id'].isin(bcg_id)]

    # assert len(bcg_df) == len(np.unique(member_df['cluster_id']))
    # print(len(bcg_df), len(member_df))

    # bcg_df.to_csv('filtered_bcg.csv', index=False)
    # member_df.to_csv('filtered_members.csv', index=False)


    # -- calculate masses
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
        
        np.savetxt('{f}_{m}_masses.txt'.format(f=fname, m=estimator), masses)

        # masses = np.loadtxt('{f}_{m}_masses.txt'.format(f=fname, m=estimator))

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

        ax.set_xlim(0.5,2.05)
        # ax.set_ylim(0,6e15)
        plt.show()

    # test_masses('filtered_bcg.csv', 'filtered_members.csv', estimator='virial', fname='test')

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

    # compare_masses('test_projected_masses.txt', 'test_virial_masses.txt')

    # -- plotting clusters for manual checking
    # import logging
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s -  %(levelname)s -  %(message)s")
    
    # bcg_df = pd.read_csv('filtered_bcg.csv')
    # member_df = pd.read_csv('filtered_members.csv')
    # bcg_arr = bcg_df.sort_values('redshift').values
    # arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)

    # bins = np.linspace(0.5,2.53,68) # bins of 0.03 width
    # bins = np.arange(0.5,2.53,0.00666)
    # digitized = np.digitize(bcg_arr[:,2], bins)

    # for i in range(1,len(bins)):
    #     binned_data = bcg_arr[np.where(digitized==i)]
    #     logging.debug('Number of clusters in bin {i}: {length}'.format(i=i,length=len(binned_data)))

    #     if len(binned_data): # plot clusters for checking
    #         fig = plt.figure(figsize=(10,8))
            # plt.hist2d(x=arr[:,0], y=arr[:,1], bins=(100,80), cmap=plt.cm.Reds)
    #         for center in binned_data:
    #             cluster_id = center[-1]
    #             cluster = arr[arr[:,-1]==cluster_id]
    #             plt.scatter(cluster[:,0], cluster[:,1], s=5)
    #             plt.scatter(center[0], center[1])
    #             plt.axis([min(arr[:,0]), max(arr[:,0]), min(arr[:,1]), max(arr[:,1])])
    #             logging.debug('Plotting cluster: {cluster}'.format(cluster=center[0:2]))

    #         logging.debug('Plotting bin {i}. Clusters with binned redshift {redshift}'.format(i=i, redshift=bins[i]))
    #         plt.show()
