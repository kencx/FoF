import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree, distance
from astropy import units as u
from astropy.table import Table
from astropy.cosmology import WMAP5 as cosmo


data = Table.read('datasets/cosmos2015_filtered.csv')
# print(data.colnames)
data = data['NUMBER', 'ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR'] # extract only colummns we need

def linear_to_angular_dist(distance, photo_z):
    return ((distance*u.Mpc).to(u.kpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)

def mean_separation(n, radius):
    volume = 4/3 * np.pi * radius**3
    density = n/volume
    return 1/(density**(1/3))

def setdiff2d(arr_1, arr_2): # finds common rows between two arrays and return the rows of arr1 not present in arr2
    arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64)])
    arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('photoz', np.float64), ('brightness', np.float64), ('name', np.float64)])
    S = np.setdiff1d(arr_1_struc, arr_2_struc)
    return S

# select from 0.5 <= z <= 2.0 and L >= -40
data = data[(data['PHOTOZ'] <= 2.0) & (data['PHOTOZ'] >= 0.5) & (data['MR'] >= -40)]
galaxy_arr = np.asarray(list(zip(data['ALPHA_J2000'], data['DELTA_J2000'], data['PHOTOZ'], data['MR'], data['NUMBER'])))
galaxy_arr = galaxy_arr[galaxy_arr[:,2].argsort()] # sort by redshift

# fig = plt.figure(figsize=(10,8))
# plt.hist2d(data['ALPHA_J2000'], data['DELTA_J2000'], bins=(100,80))
# plt.show()

def FoF(galaxy_data, photoz_factor, linking_length_factor, virial_radius):
    '''
    galaxy_data (arr): Galaxy data
    photoz_factor (float): Factor for photoz cutoff
    linking_length_factor (float): Factor for transverse linking length cutoff
    virial_radius (float): Maximum radius of cluster in Mpc
    '''

    final = []
    candidates = {}
    luminous_galaxy_data = galaxy_data[galaxy_data[:,3] <= -20.5] # filter for galaxies brighter than -20.5

    '''
    Search for cluster candidates

    1. For each galaxy brighter than -20.5, select for galaxies within z_factor*(1+z) of galaxy's z
    2. Select for galaxies within virial radius using kdtree
    3. Select for galaxies within linking_length using kdtree
    4. For each friend galaxy, search for FoF galaxy using kdtree
    5. Initialize dict storing candidates: {centre: [arr of FoF points]}
    '''

    for i, coord in enumerate(luminous_galaxy_data):
        photoz = coord[2] 
        z_boundary = photoz_factor*(1+photoz)
        photoz_cutoff = galaxy_data[(galaxy_data[:,2] >= photoz-z_boundary) & (galaxy_data[:,2] <= photoz+z_boundary)] # select for galaxies within z +- photoz_factor(1+z)
        
        virial_tree = cKDTree(photoz_cutoff[:,:2])

        transverse_virial_idx = virial_tree.query_ball_point(coord[:2], linear_to_angular_dist(virial_radius, coord[2]).value) # binary search of all galaxies within photoz boundary
        transverse_virial_points = galaxy_data[transverse_virial_idx] # galaxies within virial radius of 2 Mpc

        mean_sep = mean_separation(len(transverse_virial_points), virial_radius) # in Mpc
        transverse_linking_length = linking_length_factor*mean_sep

        linking_tree = cKDTree(transverse_virial_points[:,:2]) 

        link_idx = linking_tree.query_ball_point(coord[:2], transverse_linking_length) # binary search of all galaxies within linking length
        f_points = transverse_virial_points[link_idx] # points within 1 linking length

        for f in f_points[:,:2]:
            fof_idx = linking_tree.query_ball_point(f, transverse_linking_length) # binary search of galaxies within 2 linking lengths (FoF)
            fof_points = transverse_virial_points[fof_idx]

            mask = np.isin(fof_points, f_points, invert=True) # filter for points not already accounted for
            vec_mask = np.isin(mask.sum(axis=1), [5])
            fof_points = fof_points[vec_mask].reshape((-1,5)) # points within 2 linking lengths (FoF)
            if len(fof_points) != 0:
                f_points = np.concatenate((f_points, fof_points)) # merge all FoF points within 2 linking lengths

        if len(f_points) >= 100: # must have >= 100 points
            candidates[tuple(coord)] = f_points # dictionary {galaxy centre: array of FoF points}
            print('Candidate ' + str(i) + ' identified with ' + str(len(f_points)) + ' members')

        if len(candidates) >= 3:
            break
        
    # fig = plt.figure(figsize=(10,8))
    # for k, v in candidates.items():
    #     plt.scatter(v[:,0], v[:,1], s=5)
    #     plt.scatter(k[0], k[1])
    # plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    # plt.show()

    '''
    Remove overlapping points

    1. Obtain an array of all candidate cluster centers
    2. For each candidate center, search for other centers within virial radius
    3. Now, compare the size of points for candidate center and center within virial radius
    4. Remove overlapping points from smaller array
    '''

    cluster_centers = np.array([k for k in candidates.keys()]) # array of cluster centers
    center_tree = cKDTree(cluster_centers[:,:2])
    merged_candidates = candidates.copy()

    for center, member_arr in candidates.items():
        virial_center_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value) # search for centers within virial radius
        centers_within_vr = cluster_centers[virial_center_idx]

        # mask = np.isin(centers_within_vr, np.asarray(center), invert=True) # this removes the center itself so there are no repeats
        # vec_mask = np.isin(mask.sum(axis=1), [5])
        # centers_within_vr = centers_within_vr[vec_mask].reshape((-1,5))

        merge_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(0.25*virial_radius, center[2]).value)
        merge_centers = cluster_centers[merge_idx] # array of centers to be merged with center galaxy


        ### find a way to remove points from centers that lie outside virial radius but still
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
                        S = setdiff2d(cen_arr, member_arr)
                        if len(S): 
                            merged_candidates[cen] = S.view(np.float64).reshape(S.shape + (-1,)) 

                    elif len(member_arr) < len(cen_arr):
                        S = setdiff2d(member_arr, cen_arr)
                        if len(S):
                            merged_candidates[center] = S.view(np.float64).reshape(S.shape + (-1,))
                    
                    elif center[2] < cen[2]:
                        S = setdiff2d(cen_arr, member_arr)
                        if len(S):
                            merged_candidates[cen] = S.view(np.float64).reshape(S.shape + (-1,))
                    
                    else:
                        S = setdiff2d(member_arr, cen_arr)
                        if len(S):
                            merged_candidates[center] = S.view(np.float64).reshape(S.shape + (-1,))


        # merge cluster centers within 0.25*virial radius from current center
        if len(merge_centers): # check not empty
            for mg in merge_centers:
                mg = tuple(mg)

                if center == mg:
                    continue

                if (mg in merged_candidates.keys()) and (len(merged_candidates[mg])) and (len(merged_candidates[center])): # search within candidates
                    
                    merged_candidates[center] = np.concatenate((merged_candidates[center], merged_candidates[mg]), axis=0) # combine the arrays
                    arr, uniq_count = np.unique(candidates[center], axis=0, return_counts=True)
                    merged_candidates[center] = arr[uniq_count==1] # ensure all points are unique
                    merged_candidates[mg] = np.array([]) # delete all galaxies in merged galaxy
        

    final_candidates = merged_candidates.copy()

    # replace cluster center with bcg within 0.25*virial radius
    for center, member_arr in merged_candidates.items():
        bcg_idx = center_tree.query_ball_point(center[:2], linear_to_angular_dist(0.25*virial_radius, center[2]).value)
        bcg_arr = cluster_centers[bcg_idx]
        
        if len(bcg_arr):
            for bcg in bcg_arr:
                if bcg[3] > center[3]: # if bcg brighter than current center, replace it as center
                    final_candidates[tuple(bcg)] = member_arr
                    final_candidates[center] = np.array([])
    
    final_candidates = {k: v for k,v in final_candidates.items() if len(v)}

        
    # fig = plt.figure(figsize=(10,8))
    # for k, v in candidates.items():
    #     plt.scatter(v[:,0], v[:,1], s=5)
    #     plt.scatter(k[0], k[1])
    # plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    # plt.show()


    # k = sorted(merge_d, key=lambda x: len(merge_d[x]), reverse=True)
    # merge_d = OrderedDict((x, merge_d[x]) for x in k) # sort dictionary by decreasing length of FoF points

    return final_candidates


'''
Todo:
    remove corner points of data
    remove interlopers
        - calculate virial radius and remove those outside
        - remove those with velocity higher than max velocity relative to cluster centre
    store candidates somewhere
    calculate mass of clusters
    obtain mass distribution
'''

# candidate_list = FoF(galaxy_arr, photoz_factor=0.04, linking_length_factor=0.4, virial_radius=2)
