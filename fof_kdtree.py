import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance
from astropy import units as u
# from astropy.constants import G
from astropy.cosmology import WMAP5 as cosmo
from virial_mass_estimator import virial_mass_estimator as vm_estimator
from virial_mass_estimator import redshift_to_velocity

'''
Todo:
    find a way to deal with incomplete sample areas (corners, poor masked areas)
'''

def linear_to_angular_dist(distance, photo_z):
    return ((distance*u.Mpc).to(u.kpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)

def mean_separation(n, radius):
    volume = 4/3 * np.pi * radius**3
    density = n/volume
    return 1/(density**(1/3))


df = pd.read_csv('datasets/cosmos2015_filtered.csv')
df = df.loc[:, ['ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR', 'NUMBER']]

# select from 0.5 <= z <= 1.0 and L >= -40
df = df.loc[(df['PHOTOZ'] <= 1.0) & (df['PHOTOZ'] >= 0.5) & (df['MR'] >= -40)]
galaxy_arr = np.asarray(df)

galaxy_arr = galaxy_arr[galaxy_arr[:,2].argsort()] # sort by redshift
galaxy_arr = np.hstack((galaxy_arr, np.zeros((galaxy_arr.shape[0], 1)))) # add column for galaxy velocity

galaxy_arr[:,-1] = redshift_to_velocity(galaxy_arr[:,2]).to('km/s').value
luminous_galaxy_data = galaxy_arr[galaxy_arr[:,3] <= -20.5] # filter for galaxies brighter than -20.5

# fig = plt.figure(figsize=(10,8))
# plt.hist2d(data['ALPHA_J2000'], data['DELTA_J2000'], bins=(100,80))
# plt.show()


def FoF(galaxy_data, center_data, max_velocity, linking_length_factor, virial_radius, check_map=True):
    '''
    galaxy_data (arr): Galaxy data ['ra', 'dec', 'photoz', 'luminosity', 'id', 'doppler_velocity']
    center_data (arr): Cluster center candidate data ['ra', 'dec', 'photoz', 'luminosity', 'id', 'doppler_velocity']
    max_velocity (float): Maximum velocity with respect to cluster centre in km/s
    linking_length_factor (float): Factor for transverse linking length cutoff (0 < f < 1)
    virial_radius (float): Maximum radius of cluster in Mpc
    '''

    '''
    Search for cluster candidates

    1. For each galaxy brighter than -20.5, select for galaxies within a maximum velocity
    2. Select for galaxies within virial radius using kdtree
    3. Select for galaxies joined by 2x the linking length using kdtree to obtain FoF galaxies
    4. Initialize dict storing candidates: {centre: [arr of FoF points]}
    '''

    candidates = {}

    for i, coord in enumerate(center_data):

        coord_velocity = redshift_to_velocity(coord[2]).to('km/s').value # candidate centre doppler velocity
        velocity_cutoff = galaxy_data[(galaxy_data[:,-1] >= coord_velocity - max_velocity) & (galaxy_data[:,-1] <= coord_velocity + max_velocity)] # select for galaxies within max_velocity of cluster centre

        virial_tree = cKDTree(velocity_cutoff[:,:2])

        transverse_virial_idx = virial_tree.query_ball_point(coord[:2], linear_to_angular_dist(virial_radius, coord[2]).value) # binary search of all galaxies within photoz boundary
        transverse_virial_points = velocity_cutoff[transverse_virial_idx] # galaxies within virial radius of 2 Mpc

        mean_sep = mean_separation(len(transverse_virial_points), virial_radius) # in Mpc
        transverse_linking_length = linking_length_factor*mean_sep

        linking_tree = cKDTree(transverse_virial_points[:,:2]) 

        link_idx = linking_tree.query_ball_point(coord[:2], transverse_linking_length) # binary search of all galaxies within linking length
        f_points = transverse_virial_points[link_idx] # points within 1 linking length

        for f in f_points[:,:2]:
            fof_idx = linking_tree.query_ball_point(f, transverse_linking_length) # binary search of galaxies within 2 linking lengths (FoF)
            fof_points = transverse_virial_points[fof_idx]

            mask = np.isin(fof_points, f_points, invert=True) # filter for points not already accounted for
            vec_mask = np.isin(mask.sum(axis=1), [6])
            fof_points = fof_points[vec_mask].reshape((-1,6)) # points within 2 linking lengths (FoF)
            if len(fof_points) != 0:
                f_points = np.concatenate((f_points, fof_points)) # merge all FoF points within 2 linking lengths

        if len(f_points) >= 20: # must have >= 20 points
            candidates[tuple(coord)] = f_points # dictionary {galaxy centre: array of FoF points}
            print('Candidate ' + str(i) + ' identified with ' + str(len(f_points)) + ' members')

        if len(candidates) >= 10:
            break
        

    if check_map:    
        fig = plt.figure(figsize=(10,8))
        for k, v in candidates.items():
            plt.scatter(v[:,0], v[:,1], s=5)
            plt.scatter(k[0], k[1])
        plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
        plt.show()

    '''
    Remove overlapping clusters

    1. Obtain an array of all candidate cluster centers
    2. For each candidate center, search for other centers within virial radius
    3. Remove overlapping points from smaller (or less bright) array
    4. Overlapping, removed centers have their centers redefined to a new centroid of the smaller cluster
    4. Combine clusters with centers 0.25 x virial radius from each other
    5. Search for member galaxies brighter than candidate cluster center that are 0.25 x virial radius away
    6. Replace the brightest galaxy as the new cluster center
    7. Initialize the cluster only if richness >= 10 (richness is defined as the number of member galaxies within 0.25*virial radius)
    '''

    def setdiff2d(arr_1, arr_2): # finds common rows between two arrays and return the rows of arr1 not present in arr2
        arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('doppler_velocity', np.float64)])
        arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64), ('doppler_velocity', np.float64)])
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

        vel_cutoff = cluster_centers[(cluster_centers[:,-1] >= center[-1] - max_velocity) & (cluster_centers[:,-1] <= center[-1] + max_velocity)]
        center_tree = cKDTree(vel_cutoff[:,:2])

        virial_center_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value) # search for centers within virial radius
        centers_within_vr = cluster_centers[virial_center_idx]

        # remove overlapping points
        if len(centers_within_vr):
            for cen in centers_within_vr:
                cen = tuple(cen)
                if center == cen:
                    continue

                cen_arr = merged_candidates[cen]
                if len(cen_arr) and len(member_arr): # check not empty
                    
                    # compare size of two arrays and removes common points from smaller array. if both are same size, remove common points from less bright centre
                    if len(member_arr) > len(cen_arr): 
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)

                    elif len(member_arr) < len(cen_arr):
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)
                    
                    elif center[2] < cen[2]:
                        remove_overlap(merged_candidates, cen, cen_arr, member_arr)
                    
                    else:
                        remove_overlap(merged_candidates, center, member_arr, cen_arr)


        merge_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(0.25*virial_radius, center[2]).value) # search for centers within 0.25*virial radius
        merge_centers = cluster_centers[merge_idx] # array of centers within 0.25*virial radius, to be merged with center galaxy

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
                    if len(mg) > len(center):
                        merge(merged_candidates, mg, center)

                    elif len(center) > len(mg):
                        merge(merged_candidates, center, mg)

                    elif mg[2] > center[2]:
                        merge(merged_candidates, mg, center)

                    else:
                        merge(merged_candidates, center, mg)

    
    merged_cluster_centers = np.array([k for k in merged_candidates.keys()])
    center_shifted_candidates = merged_candidates.copy()

    for mcenter, member_arr in merged_candidates.items():

        vel_cutoff = merged_cluster_centers[(merged_cluster_centers[:,-1] >= mcenter[-1] - max_velocity) & (merged_cluster_centers[:,-1] <= mcenter[-1] + max_velocity)]
        center_tree = cKDTree(vel_cutoff[:,:2])

        bcg_idx = center_tree.query_ball_point(mcenter[:2], linear_to_angular_dist(0.25*virial_radius, mcenter[2]).value) # search for centers within 0.25*virial radius
        bcg_arr = merged_cluster_centers[bcg_idx] # array of centers within 0.25*virial radius

        if len(bcg_arr):
            bcg = tuple(bcg_arr[np.argmin(bcg_arr[:,3])]) # galaxy with largest brightness
            if not bcg == mcenter:
                center_shifted_candidates[tuple(bcg)] = member_arr # replace cluster center with bcg within 0.25*virial radius
                center_shifted_candidates[mcenter] = np.array([])
    
    center_shifted_candidates = {k: v for k,v in center_shifted_candidates.items() if len(v)}
    final_candidates = {}
    
    for fcenter, member_arr in center_shifted_candidates.items():
        richness_tree = cKDTree(member_arr[:,:2])

        r_idx = richness_tree.query_ball_point(fcenter[:2], linear_to_angular_dist(0.25*virial_radius, fcenter[2]).value)
        r_arr = member_arr[r_idx]
        richness = len(r_arr)

        if richness < 8:  # Initialize the cluster only if richness >= 20 (richness is defined as the number of member galaxies within 0.25*virial radius)
            final_candidates[fcenter] = np.array([]) 
        else:
            final_candidates[fcenter + (richness,)] = member_arr

    final_candidates = {k: v for k,v in final_candidates.items() if len(v)}

    if check_map:
        fig = plt.figure(figsize=(10,8))
        for k, v in final_candidates.items():
            plt.scatter(v[:,0], v[:,1], s=5)
            plt.scatter(k[0], k[1])
        plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
        plt.show()

    return final_candidates


def create_df(d):
    '''
    Create two dataframes to store cluster information
    1. BCG table ['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'gal_id', 'cluster_id']
    2. Member galaxy table ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'cluster_id']
    '''
    bcg_list = [k for k in d.keys()]
    columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'doppler_velocity']
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
        # member_galaxy_df.loc[i ,'cluster_id'] = i  
        
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'gal_id', 'cluster_id']]
    member_galaxy_df = member_galaxy_df[['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'cluster_id']]

    return bcg_df, member_galaxy_df


def interloper_removal(cluster_center, cluster_members):

    '''
    - calculate virial mass and use it to:
        - calculate virial radius
        - calculate max velocity
    - repeat until converge
    '''

    def max_velocity(M, r): # max velocity in km/s
        v_cir = np.sqrt(G*M/r)
        v_inf = 2**(1/2)*v_cir
        return max(v_cir, v_inf)

    def virial_radius_estimator(virial_mass, photo_z):
        critical_density = cosmo.critical_density(z=photo_z)
        r_200 = virial_mass/(200*np.pi*4/3*critical_density)
        return r_200**(1/3).to('Mpc')


    virial_mass, velocity_dispersion, projected_radius = vm_estimator(cluster_members, savetxt=False)
    cleaned_members = np.copy(cluster_members)

    size_before = len(cleaned_members)
    size_after = 0

    while size_after >= size_before: # repeat until convergence
        size_before = len(cleaned_members)

        for idx, member in enumerate(cluster_members):
            if (member[5] >= cluster_center[5] + max_velocity) or (member[5] <= cluster_center[5] - max_velocity): # identify interlopers further than max velocity
                np.delete(cleaned_members, (idx), axis=0) # remove interloper

        virial_radius = virial_radius_estimator(virial_mass, cluster_center[:2])
        tree = cKDTree(cleaned_members[:,:2])
        idx = tree.query_ball_point(cluster_center[:2], r=virial_radius.value)
        cleaned_members = cleaned_members[idx]
        
        size_after = len(cleaned_members)

    return cluster_center, cleaned_members

    

def cluster_search(galaxy_data, center_data, max_velocity, linking_length_factor, virial_radius, check_map=True, export=False):
    final_candidates = FoF(galaxy_data, center_data, max_velocity=max_velocity, linking_length_factor=linking_length_factor, virial_radius=virial_radius, check_map=check_map)
    # final_candidates = interloper_removal()

    bcg_df, member_df = create_df(final_candidates)
    if export:
        bcg_df.to_csv('bcg_df', index=False)
        member_df.to_csv('member_df', index=False)

    print('Cluster search returned ' + str(len(bcg_df)) + ' clusters')
    return bcg_df, member_df


bcg_df, member_galaxy_df = cluster_search(galaxy_arr, luminous_galaxy_data, max_velocity=2000, linking_length_factor=0.4, virial_radius=2)


# --- testing interloper removal
# cleaned_candidates = {}
# for k,v in candidate_list.items():
#     cleaned_k, cleaned_v = interloper_removal(k,v)
#     cleaned_candidates[cleaned_k] = cleaned_v

# virial_masses = {}
# for center,candidate in candidate_list.items():
#     mass, vel_disp, radius = vm_estimator(candidate)
#     virial_masses[center] = mass
# print(virial_masses)
