# import time
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree, distance
from scipy.stats import norm, linregress

from astropy import units as u
from astropy.constants import G
from astropy.cosmology import WMAP5 as cosmo

from virial_mass_estimator import virial_mass_estimator, redshift_to_velocity, projected_mass_estimator
from data_processing import split_df_into_groups


def linear_to_angular_dist(distance, photo_z):
    '''
    Converts proper distance (Mpc) to angular distance (deg)

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

    return ((distance*u.Mpc).to(u.kpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


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


def luminous_search(galaxy_data, max_velocity):
    '''
    Parameters
    ----------
    galaxy_data: array-like
        Galaxy data with properties: ['ra', 'dec', 'photoz', 'abs mag', 'id', 'doppler_velocity']

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

            if len(f_points) >= 50: # must have >= 20 points
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
        arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('doppler_velocity', np.float64), ('N(0.5)', np.float64)])
        arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('doppler_velocity', np.float64), ('N(0.5)', np.float64)])
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

    def find_richness(point, member_galaxies, richness_tree):
        r_idx = richness_tree.query_ball_point(point[:2], linear_to_angular_dist(0.25*virial_radius, point[2]).value)
        r_arr = member_galaxies[r_idx]
        return len(r_arr)


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
        bcg_arr = member_arr[bcg_idx] # array of centers within 0.25*virial radius

        if len(bcg_arr) and len(member_arr):
            mag_sort = bcg_arr[bcg_arr[:,3].argsort()]
            bcg = mag_sort[0]
            if (abs(bcg[3]) > abs(mcenter[3])) and (tuple(bcg) != mcenter): # if bcg brighter than current center, replace it as center
                center_shifted_candidates[tuple(bcg)] = member_arr
                center_shifted_candidates[mcenter] = np.array([])

    center_shifted_candidates = {k: v for k,v in center_shifted_candidates.items() if len(v)>=50}
    final_candidates = {}
    
    for fcenter, member_arr in center_shifted_candidates.items():
        richness_tree = cKDTree(member_arr[:,:2])
        richness = find_richness(fcenter, member_arr, richness_tree)

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


def create_df(d):
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

    bcg_list = [k for k in d.keys()]
    columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'doppler_velocity', 'N(0.5)']
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
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'gal_id', 'cluster_id']]
    member_galaxy_df = member_galaxy_df[['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'cluster_id']]

    return bcg_df, member_galaxy_df


def interloper_removal(cluster_center, cluster_members):
    ''' Removes interlopers of an individual cluster using the max velocity and virial radius approach.

    The virial mass is estimated and used to calculate the virial radius and maximum velocity of a cluster. 
    Galaxies that lie outside the virial radius or have a larger than maximum velocity are removed. The process is repeated until convergence.
    If number of galaxies < 20, cluster is deleted.

    Parameters
    ----------
    cluster_center: tuple
        Cluster center candidate

    cluster_members: array-like
        Array of cluster members

    Returns
    -------
    cleaned_members: array-like
        Cleaned array of cluster members
    '''

    def v_cir(M, r):
        return (np.sqrt(G*M/(r*u.Mpc))).to(u.km/u.s)

    def v_inf(v_cir):
        return v_cir*2**(1/2)

    def max_velocity(cluster_center, cluster_members, virial_radius, tree, n):

        '''
        Calculation of maximum velocity (see Wojtak, R. et al., (2007). Interloper treatment in dynamical modelling of galaxy clusters.). 
        The max velocity is calculated from the infall and circular velocity with max(v_inf*cos(theta),v_cir*sin(theta))

        Parameters
        ----------
        cluster_center: tuple
        Cluster center

        cluster_members: array-like
        Array of cluster members

        virial_radius: float
        Virial radius in Mpc

        tree: cKDtree
            Binary search tree of cluster members

        n: int
            Number of bins to split plane into


        Returns
        -------
        final_members: array-like
            Array of members with velocity < max velocity
        '''
        
        r_ij = np.zeros((n,n))
        R = np.linspace(0, virial_radius, n) # bin projected plane into n steps

        r_ij[:,0] = R # initialize projected distance into first column
        for j in range(len(R)):
            # col0: projected 2D plane distances
            # from col1: sqrt(binned projected distance^2 + binned LOS distance^2)
            r_ij[:,j] = np.sqrt(R**2 + R[j]**2) # bin LOS plane into n steps and initialize actual distance
        r_ij[r_ij > virial_radius] = virial_radius # replace all values > virial_radius with virial_radius
        
        theta = np.zeros(r_ij.shape)
        v_max = np.zeros(r_ij.shape)
        c_mem = {}

        cluster_redshift = np.median(cluster_members[:,2])
        A = linear_to_angular_dist(R, cluster_center[2]).value # angular distance

        for i in range(len(R)):
            theta[:,i] = np.arcsin(r_ij[:,0]/r_ij[:,i]) # divide all cols by first col, then take arcsin 
        theta[1:,0] = 0.0

        for i in range(len(R)): # bottleneck

            concentric_idx = tree.query_ball_point(cluster_center[:2], r=A[i]) # search for members within each bin
            concentric_members = cluster_members[concentric_idx]
            c_mem[i] = concentric_members # dict of members in each concentric bin, key is the index for each bin
            
        M = np.zeros(R.shape)
        for k in c_mem.keys(): # evaluate mass of galaxies in concentric bins
            if len(c_mem[k]) > 1:
                if (len(c_mem[k]) - len(c_mem[k-1])) != 0:
                    M[k] = virial_mass_estimator(c_mem[k], cluster_redshift)[0].value
                else:
                    M[k] = M[k-1] # if num of galaxies does not change, mass remains the same

        for i in range(len(R)):
            v_c = v_cir(M[i]*u.M_sun, r_ij[i,:])
            v_max[i,:] = np.maximum(v_c*np.cos(theta[i,:]), v_inf(v_c)*np.sin(theta[i,:]))
        
        v_max = np.amax(v_max, axis=1) # maximum velocity for each row (projected radius step)
        v_max[0] = 0

        # configure dict to have only set members in each concentric bin, ie. the members in a specific bin should contain galaxies from the first to the current bin
        c_mem2 = c_mem.copy()
        for k in c_mem.keys():
            if k >= 1:
                a1 = c_mem[k]
                a2 = c_mem[k-1]
                a1_rows = a1.view([('', a1.dtype)]*a1.shape[1])
                a2_rows = a2.view([('', a2.dtype)]*a2.shape[1])
                c_mem2[k] = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1,a1.shape[1])

        # remove those with v > v_max
        for k in c_mem2.keys():
            mem = c_mem2[k]
            mem = mem[abs(mem[:,-1] - cluster_center[-2]) <= v_max[k]] 
            c_mem2[k] = mem

        final_members = np.concatenate([v for v in c_mem2.values()]).ravel().reshape(-1,cluster_members.shape[1]) # unravel dict into arr of members

        return final_members

    def virial_radius_estimator(virial_mass, redshift):
        '''
        Calculation of virial radius

        Parameters
        ----------
        virial_mass: float, array-like
            Virial mass of object in M_sun

        redshift: float, array-like
            Redshift

        Returns
        -------
        virial_radius: float, array-like
            Virial radius in Mpc
        '''

        critical_density = cosmo.critical_density(z=redshift)
        r_200 = virial_mass/(200*np.pi*4/3*critical_density)
        return (r_200**(1/3)).to(u.Mpc)


    cleaned_members = np.copy(cluster_members)
    size_before = len(cleaned_members)
    size_after = 0
    size_change = size_after - size_before

    while (size_change != 0) and (len(cleaned_members) > 0): # repeat until convergence

        size_before = len(cleaned_members)
        virial_mass, _, _ = virial_mass_estimator(cluster_members, np.median(cluster_members[:,2]))
        
        virial_radius = virial_radius_estimator(virial_mass, cluster_center[2]).value
        tree = cKDTree(cleaned_members[:,:2])
        idx = tree.query_ball_point(cluster_center[:2], virial_radius)
        cleaned_members = cleaned_members[idx]
        if len(cleaned_members) < 20:
            cleaned_members = np.array([])
            break
        
        cleaned_members = max_velocity(cluster_center, cleaned_members, virial_radius, tree, n=50)

        size_after = len(cleaned_members)
        size_change = size_after - size_before

    if len(cleaned_members) < 20:
        cleaned_members = np.array([])

    return cleaned_members


def cluster_search(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, check_map=True, export=False, fname=''):
    '''
    Performs a FoF search for cluster candidates, followed by removal of interlopers. Saves the data to a pandas dataframe.

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

    check_map: bool (default: True)
        Plot a scatter plot to check cluster members

    export: bool (default: False)
        Exports df to csv file with filename fname

    fname: str (default: empty)
        Filename of csv file to be exported, if export=True.

    Returns
    -------
    bcg_df: pd.Dataframe
        df of cluster centers

    member_df: pd.Dataframe
        df of members in clusters
    '''

    # search for candidates
    candidates = FoF(galaxy_data, luminous_data, max_velocity=max_velocity, linking_length_factor=linking_length_factor, virial_radius=virial_radius, check_map=check_map)
    print('{n} candidate clusters found.'.format(n=len(candidates)))

    candidate_df, candidate_member_df = create_df(candidates)
    if export:
        candidate_df.to_csv(fname + '_candidate_bcg.csv', index=False)
        candidate_member_df.to_csv(fname + '_candidate_members.csv', index=False)
    

    # interloper treatment
    # final_candidates = candidates.copy()
    # for center, members in candidates.items():
    #     final_candidates[center] = interloper_removal(center, members)
    #     print(str(len(members)-len(final_candidates[center])) + ' interlopers removed.')

    # final_candidates = {k: v for k,v in final_candidates.items() if len(v)}

    # # export to df
    # bcg_df, member_df = create_df(final_candidates)
    # if export:
    #     bcg_df.to_csv(fname + '_final_bcg.csv', index=False)
    #     member_df.to_csv(fname + '_final_members.csv', index=False)

    try:
        print('Cluster search returned {bcg} clusters'.format(bcg=len(bcg_df)))
        return bcg_df, member_df
    except NameError:
        print('Cluster search returned {candidates} clusters'.format(candidates=len(candidate_df)))
        return candidate_df, candidate_member_df


if __name__ == "__main__":

    # conn = sqlite3.connect('galaxy_clusters.db')
    # df = pd.read_sql_query('''
    # SELECT *
    # FROM mag_lim
    # ORDER BY redshift
    # ''', conn)
    # conn.close()

    # df = df.loc[:, ['ra','dec','redshift','RMag','ID']] # select relevant columns
    # df['Doppler_vel'] = redshift_to_velocity(df['redshift']).to('km/s').value
    # df = df.sort_values('redshift')

    # galaxy_arr = df.values
    # galaxy_arr, luminous_arr = luminous_search(galaxy_arr, max_velocity=2000)
    # np.savetxt('counted_galaxy.csv', galaxy_arr, delimiter=',')
    # np.savetxt('luminous_galaxy.csv', luminous_arr, delimiter=',')

    # galaxy_arr = np.loadtxt('counted_galaxy.csv', delimiter=',')
    # luminous_arr = np.loadtxt('luminous_galaxy.csv', delimiter=',')

    # galaxy_arr = galaxy_arr[galaxy_arr[:,2] <= 2.05]
    # luminous_arr = luminous_arr[luminous_arr[:,2] <= 2.05]
    # print(len(galaxy_arr), len(luminous_arr))

    # -- testing cluster_search function
    # bcg_df, member_df = cluster_search(galaxy_arr, luminous_arr, max_velocity=2000, linking_length_factor=0.4, virial_radius=2, check_map=False, export=True, fname='fofv2')


    def test_masses(bcg_file, member_file, estimator, fname='test'):
        bcg_df = pd.read_csv(bcg_file)
        member_df = pd.read_csv(member_file)

        bcg_arr = bcg_df.sort_values('cluster_id').values
        arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
        masses = np.zeros(group_n.shape)

        for g in group_n:
            cluster = arr[arr[:,-1]==g]
            center = bcg_arr[bcg_arr[:,-1]==g]
            if estimator == 'virial':
                mass, _, _ = virial_mass_estimator(cluster[:,:3])
            if estimator == 'projected':
                mass = projected_mass_estimator(center, cluster)
            masses[int(g)] = mass.value
        
        np.savetxt(fname + '_masses.txt', masses)

        # masses = np.loadtxt(fname + '_masses.txt')

        i, = np.where(bcg_arr[:,2] < 2)
        bcg_arr = bcg_arr[i,:]
        masses = masses[i]

        j, = np.where(masses<1e16)
        bcg_arr = bcg_arr[j,:]
        masses = masses[j]

        fig, ax = plt.subplots(figsize=(10,8))
        p = ax.scatter(bcg_arr[:,2], masses, s=10, c=bcg_arr[:,3], alpha=0.75)
        cbar = fig.colorbar(p)

        ax.set_title('Mass against redshift')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Cluster Mass')
        cbar.ax.set_ylabel('Abs Mag')
        ax.ticklabel_format(style='sci', axis='y')
        ax.yaxis.major.formatter._useMathText = True

        x = bcg_arr[:,2]
        a,b,r,_,_ = linregress(x, masses)
        ax.plot(x,a*x+b,'k--',alpha=0.75)
        print(r)

        ax.set_xlim(0,2.05)
        ax.set_ylim(0,6e15)
        plt.show()

    # test_masses('fofv2_candidate_bcg.csv', 'fofv2_candidate_members.csv', estimator='projected', fname='fofv2_projected')

    def compare_masses(projected_masses, virial_masses):
        projected = np.loadtxt(projected_masses)
        virial = np.loadtxt(virial_masses)

        projected = projected[projected < 1e17]
        virial = virial[virial < 1e17]

        fig, ax = plt.subplots()
        ax.scatter(x=virial, y=projected, s=10, alpha=0.75)
        lims = [0,
                np.max([ax.get_xlim(), ax.get_ylim()])]

        m, c, r, _, _, = linregress(virial, projected)
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

    # compare_masses('fofv2_projected_masses.txt', 'fofv2_masses.txt')


# -- plotting clusters for manual checking
    
    # bcg_df = pd.read_csv('mag_cut_candidate_bcg.csv')
    # member_df = pd.read_csv('mag_cut_candidate_members.csv')

    # bcg_arr = bcg_df.sort_values('cluster_id').values
    # arr, group_n = split_df_into_groups(member_df, 'cluster_id', -1)
    
    # fig = plt.figure(figsize=(10,8))
    # for g in group_n:
    #     cluster = arr[arr[:,-1]==g]
    #     center = bcg_arr[bcg_arr[:,-1]==g]
    #     plt.scatter(cluster[:,0], cluster[:,1], s=5)
    #     plt.scatter(center[:,0], center[:,1])
    #     plt.axis([min(arr[:,0]), max(arr[:,0]), min(arr[:,1]), max(arr[:,1])])
    #     if not g % 10:        
    #         plt.show()

    # cluster = arr[arr[:,-1]==0]
    # center = bcg_arr[bcg_arr[:,-1]==0]
    # plt.scatter(cluster[:,0], cluster[:,1], s=5)
    # plt.scatter(center[:,0], center[:,1])
    # # if not g % 20:        
    # plt.axis([min(arr[:,0]), max(arr[:,0]), min(arr[:,1]), max(arr[:,1])])
    # plt.show()
