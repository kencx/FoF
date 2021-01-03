import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance
from astropy import units as u
from astropy.constants import G
from astropy.cosmology import WMAP5 as cosmo
from virial_mass_estimator import virial_mass_estimator as vm_estimator
from virial_mass_estimator import redshift_to_velocity, split_df_into_groups
import time

'''
Todo:
    find a way to deal with incomplete sample areas (corners, poor masked areas)
    write interloper_removal function
'''

def linear_to_angular_dist(distance, photo_z):
    return ((distance*u.Mpc).to(u.kpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)

def mean_separation(n, radius):
    volume = 4/3 * np.pi * radius**3
    density = n/volume
    return 1/(density**(1/3))


# -- preparing COSMOS photometric data

df = pd.read_csv('datasets/cosmos2015_filtered.csv')
df = df.loc[:, ['ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR', 'NUMBER']]

# select from 0.5 <= z <= 1.0 and L >= -40
df = df.loc[(df['PHOTOZ'] <= 1.0) & (df['PHOTOZ'] >= 0.5)] # & (df['MR'] >= -40)
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
        velocity_cutoff = galaxy_data[abs(galaxy_data[:,-1] - coord_velocity) <= max_velocity] # select for galaxies within max_velocity of cluster centre

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
            fof_points = fof_points[vec_mask].reshape((-1,6)) # points of 2 linking lengths (FoF)
            if len(fof_points) != 0:
                f_points = np.concatenate((f_points, fof_points)) # merge all FoF points within 2 linking lengths

        if len(f_points) >= 20: # must have >= 20 points
            candidates[tuple(coord)] = f_points # dictionary {galaxy centre: array of FoF points}
            if not i % 100:
                print('Candidate ' + str(i) + ' identified with ' + str(len(f_points)) + ' members')

        # if len(candidates) >= 100:
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

    1. Obtain an array of all candidate cluster centers
    2. For each candidate center, search for other centers within virial radius
    3. Remove overlapping points from smaller (or less bright) array
    4. Overlapping, removed centers have their centers redefined to a new centroid of the smaller cluster
    4. Combine clusters with centers 0.25 x virial radius from each other
    5. Search for member galaxies brighter than candidate cluster center that are 0.25 x virial radius away
    6. Replace the brightest galaxy as the new cluster center
    7. Initialize the cluster only if richness >= 8 (richness is defined as the number of member galaxies within 0.25*virial radius)
    '''

    def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
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

    def find_richness(point, member_galaxies, richness_tree):
        r_idx = richness_tree.query_ball_point(point[:2], linear_to_angular_dist(0.25*virial_radius, point[2]).value)
        r_arr = member_galaxies[r_idx]
        return len(r_arr)


    cluster_centers = np.array([k for k in candidates.keys()]) # array of cluster centers

    merged_candidates = candidates.copy()

    for center, member_arr in candidates.items():

        vel_cutoff = cluster_centers[abs(cluster_centers[:,-1] - center[-1]) <= max_velocity]
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

        vel_cutoff = merged_cluster_centers[abs(merged_cluster_centers[:,-1] - mcenter[-1]) <= max_velocity]
        center_tree = cKDTree(vel_cutoff[:,:2])

        bcg_idx = center_tree.query_ball_point(mcenter[:2], linear_to_angular_dist(0.25*virial_radius, mcenter[2]).value) # search for centers within 0.25*virial radius
        bcg_arr = merged_cluster_centers[bcg_idx] # array of centers within 0.25*virial radius

        if len(bcg_arr) and len(member_arr):
            sort_brightness = bcg_arr[bcg_arr[:,3].argsort()]
            bcg = sort_brightness[0]
            if (abs(bcg[3]) > abs(center[3])) and (tuple(bcg) != mcenter): # if bcg brighter than current center, replace it as center
                center_shifted_candidates[tuple(bcg)] = member_arr
                center_shifted_candidates[mcenter] = np.array([])

        # if (len(bcg_arr)) and (len(member_arr)):
        #     sort_brightness = bcg_arr[bcg_arr[:,3].argsort()] # sort bcg_arr by brightness

        #     if sort_brightness[0,3] > mcenter[3]:
        #         bcg = tuple(sort_brightness[0])
            
        #     else: bcg = mcenter

        #     # for i in range(len(sort_brightness)):
        #     #     richness = find_richness(sort_brightness[i], member_arr) # find richness
            
        #     #     if richness >= 8:
        #     #         bcg = tuple(sort_brightness[i]) # galaxy with largest brightness
        #     #         break
        #     #     else:
        #     #         bcg = mcenter
        #     #         break

        #     if not bcg == mcenter:
        #         center_shifted_candidates[tuple(bcg)] = member_arr # replace cluster center with bcg within 0.1*virial radius if richness > 8
        #         center_shifted_candidates[mcenter] = np.array([])
    
    center_shifted_candidates = {k: v for k,v in center_shifted_candidates.items() if len(v)}
    final_candidates = {}
    
    for fcenter, member_arr in center_shifted_candidates.items():
        richness_tree = cKDTree(member_arr[:,:2])
        richness = find_richness(fcenter, member_arr, richness_tree)

        if richness < 8:  # Initialize the cluster only if richness >= 8 (richness is defined as the number of member galaxies within 0.25*virial radius)
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
        bcg_df.loc[i, 'cluster_redshift'] = np.median(v[:,2])
        
        N = v.shape
        temp_v = np.zeros((N[0], N[1]+1))
        temp_v[:,:-1] = v
        temp_v[:,-1] = i
        temp = pd.DataFrame(data=temp_v, columns=columns+['cluster_id'])
        member_galaxy_df = member_galaxy_df.append(temp)
        # member_galaxy_df.loc[i ,'cluster_id'] = i  
    
    bcg_df = bcg_df.sort_values('redshift')
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'richness', 'total_N', 'gal_id', 'cluster_id', 'cluster_redshift']]
    member_galaxy_df = member_galaxy_df[['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'cluster_id']]

    return bcg_df, member_galaxy_df


def interloper_removal(cluster_center, cluster_members):

    '''
    - calculate virial mass and use it to:
        - calculate virial radius
        - calculate max velocity
    - repeat until converge
    TODO: find max_velocity of galaxies in LOS of cluster center
    '''

    def v_cir(M, r):
        return (np.sqrt(G*M/(r*u.Mpc))).to(u.km/u.s)

    def v_inf(v_cir):
        return v_cir*2**(1/2)

    def max_velocity(cluster_center, cluster_members, virial_radius, tree, n):
        
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
                    M[k] = vm_estimator(c_mem[k], cluster_redshift)[0].value
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

    def virial_radius_estimator(virial_mass, photo_z):
        critical_density = cosmo.critical_density(z=photo_z)
        r_200 = virial_mass/(200*np.pi*4/3*critical_density)
        return (r_200**(1/3)).to(u.Mpc)


    cleaned_members = np.copy(cluster_members)
    size_before = len(cleaned_members)
    size_after = 0
    size_change = size_after - size_before

    while (size_change != 0) and (len(cleaned_members) > 0): # repeat until convergence

        size_before = len(cleaned_members)
        virial_mass, _, _ = vm_estimator(cluster_members, np.median(cluster_members[:,2]))
        
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

    
def cluster_search(galaxy_data, center_data, max_velocity, linking_length_factor, virial_radius, check_map=True, export=False, fname=''):
    candidates = FoF(galaxy_data, center_data, max_velocity=max_velocity, linking_length_factor=linking_length_factor, virial_radius=virial_radius, check_map=check_map)
    candidate_df, candidate_member_df = create_df(candidates)
    if export:
        candidate_df.to_csv(fname + '_candidate_bcg.csv', index=False)
        candidate_member_df.to_csv(fname + '_candidate_members.csv', index=False)
    
    print((str(len(candidates)) + ' candidate clusters found.'))

    final_candidates = candidates.copy()
    for center, members in candidates.items():
        final_candidates[center] = interloper_removal(center, members)
        print(str(len(members)-len(final_candidates[center])) + ' interlopers removed.')

    final_candidates = {k: v for k,v in final_candidates.items() if len(v)}

    bcg_df, member_df = create_df(final_candidates)
    if export:
        bcg_df.to_csv(fname + '_final_bcg.csv', index=False)
        member_df.to_csv(fname + '_final_members.csv', index=False)

    print('Cluster search returned ' + str(len(bcg_df)) + ' clusters')
    return bcg_df, member_df


if __name__ == "__main__":

    # -- testing cluster_search function
    bcg_df, member_galaxy_df = cluster_search(galaxy_arr, luminous_galaxy_data, max_velocity=2000, linking_length_factor=0.4, virial_radius=2, check_map=True, export=True, fname='test')
    

    # -- testing speed of virial_mass_estimator
    # bcg_df = pd.read_csv('test_bcg.csv')
    # bcg_arr = bcg_df.values
    # member_galaxy_df = pd.read_csv('test_members.csv')
    # arr, group_n = split_df_into_groups(member_galaxy_df,'cluster_id')

    # test_id = group_n[4]
    # test_group = arr[arr[:,-1]==test_id]
    # test_center = bcg_arr[bcg_arr[:,-2]==test_id][0]

    # start = time.time()
    # interloper_removal(test_center, test_group)
    # end= time.time()
    # print(str(end-start) + 's for ' + str(len(test_group)) + ' galaxies')


    # masses = np.zeros(group_n.shape)

    # for g in group_n[:100]:
    #     cluster = arr[arr[:,-1]==g]
    #     bcg = bcg_arr[int(g),:]
    #     mass, vel_disp, rad = vm_estimator(cluster[:,:3], bcg[-1])
    #     masses[int(g)] = mass.value

    # np.savetxt(fname='cluster_masses.txt', X=masses)

    # -- distribution of estimated cluster virial masses
    # masses = np.loadtxt(fname='cluster_masses.txt')
    # masses = masses[masses > 0]
    # plt.hist(masses, bins='auto')
    # plt.show()
