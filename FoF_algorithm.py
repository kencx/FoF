# import pickle
import sqlite3
import numpy as np
import pandas as pd
from random import uniform
import matplotlib.pyplot as plt

from scipy.stats import norm, linregress
from scipy.spatial import cKDTree

from astropy import units as u
from astropy.constants import G
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord

from mass_estimator import virial_mass_estimator, projected_mass_estimator
from methods import linear_to_angular_dist, mean_separation, redshift_to_velocity
from data_processing import split_df_into_groups

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


def setdiff2d(arr_1, arr_2): # finds common rows between two 2d arrays and return the rows of arr1 not present in arr2
    arr_1_struc = arr_1.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
    arr_2_struc = arr_2.ravel().view(dtype=[('ra', np.float64), ('dec', np.float64), ('redshift', np.float64), ('brightness', np.float64), ('ID', np.float64), ('LR', np.float64), ('N(0.5)', np.float64)])
    S = np.setdiff1d(arr_1_struc, arr_2_struc)
    return S


def find_number_count(center, distance):
    member_galaxies = center.galaxies
    N_tree = cKDTree(member_galaxies[:,:2])
    coords = [center.ra, center.dec]
    n_idx = N_tree.query_ball_point(coords, linear_to_angular_dist(distance, center.z).value)
    n_arr = member_galaxies[n_idx]
    return len(n_arr)


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

        if (len(vel_cutoff) > 1) & (len(vel_cutoff) >= 8):
            galaxy_data[i,-1] = find_number_count(galaxy, vel_cutoff, 0.5*u.Mpc/u.littleh) # galaxies within 0.5 h^-1 Mpc
        else:
            galaxy_data[i,-1] = 0

    galaxy_data = galaxy_data[galaxy_data[:,-1].argsort()[::-1]] # sort by N(0.5) in desc
    assert galaxy_data[0,-1] == max(galaxy_data[:,-1]), 'Galaxy array is not sorted by N(0.5)'

    luminous_galaxy = galaxy_data[galaxy_data[:,-1] >= 8] # select for galaxies with N(0.5) >= 8
    assert len(luminous_galaxy[luminous_galaxy[:,-1] < 8]) == 0, 'Luminous galaxy array has galaxies with N(0.5) < 8'

    print('Total Number of Galaxies in sample: {ngd}, Number of candidate centers: {nlg}'.format(ngd=len(galaxy_data), nlg=len(luminous_galaxy)))
    return galaxy_data, luminous_galaxy


def center_overdensity(center, galaxy_data, max_velocity): # calculate overdensity of cluster

    # select 300 random points (RA and dec)
    n = 300
    ra_random = np.random.uniform(low=min(galaxy_data[:,0]), high=max(galaxy_data[:,0]), size=n)
    dec_random = np.random.uniform(low=min(galaxy_data[:,1]), high=max(galaxy_data[:,1]), size=n)
    points = np.vstack((ra_random, dec_random)).T
    assert points.shape == (n,2)

    # select all galaxies within max velocity
    # velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s]
    velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center.z) <= 0.02*(1+center.z)]
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


class Cluster:
    # initialize with __slots__
    
    def __init__(self, center, galaxies):

        # center properties
        self.ra = center[0]
        self.dec = center[1]
        self.z = center[2]
        self.bcg_brightness = center[3]
        self.gal_id = center[4]
        self.bcg_luminosity = center[5]
        self.N = center[-1]
        self.galaxies = galaxies # np array

        # self.velocity = velocity
        # self.cluster_mass = mass (before and after interloper removal)
        # self.velocity_dispersion = velocity_dispersion
        # self.virial_radius = virial_radius

        # self.cluster_id = cluster_id

    @property
    def richness(self):
        return len(self.galaxies)

    @property
    def center_attributes(self):
        center = np.array([self.ra, self.dec, self.z, self.bcg_brightness, self.gal_id, self.bcg_luminosity, self.N])
        return center


    def remove_overlap(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        
        # compare N(0.5) of two clusters and removes the common galaxies from smaller cluster. if both are same N(0.5), remove common galaxies from cluster with less bright center galaxy
        if self.N > other.N:
            self_galaxies = self.galaxies
            other_galaxies = np.array([])
        elif self.N < other.N:
            self_galaxies = np.array([])
            other_galaxies = other.galaxies
        elif abs(self.bcg_brightness) > abs(other.bcg_brightness):
            self_galaxies = self.galaxies
            other_galaxies = np.array([])
        else:
            self_galaxies = np.array([])
            other_galaxies = other.galaxies
        return self_galaxies, other_galaxies


    def merge(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        pass # compare N(0.5) of two clusters, keep larger cluster with merged points and deletes smaller one. if both have same N(0.5), compare brightness instead

        combined = np.concatenate((self.galaxies, other.galaxies), axis=0) # combine the galaxy arrays
        arr, uniq_count = np.unique(combined, axis=0, return_counts=True)
        combined = arr[uniq_count==1] # ensure all points are unique

        if self.N > other.N:
            self.galaxies = combined
            other.galaxies = np.array([])
        elif self.N < other.N:
            self.galaxies = np.array([])
            other.galaxies = combined
        elif abs(self.bcg_brightness) > abs(other.bcg_brightness):
            self.galaxies = combined
            other.galaxies = np.array([]) 
        else:
            self.galaxies = np.array([])
            other.galaxies = combined


    def __str__(self):
        return 'Cluster at ({self.ra},{self.dec},{self.z}) with {self.richness} galaxies'.format(self=self)



def new_FoF(galaxy_data, luminous_data, max_velocity, linking_length_factor, virial_radius, richness, overdensity):

    candidates = []

    # identify cluster candidates
    for i, center in enumerate(luminous_data): # each row is a candidate center to search around
        # velocity_bin = galaxy_data[abs(redshift_to_velocity(galaxy_data[:,2])-redshift_to_velocity(center[2]))/(1+center[2]) <= max_velocity*u.km/u.s] # select galaxies within max velocity
        velocity_bin = galaxy_data[abs(galaxy_data[:,2]-center[2]) <= 0.02*(1+center[2])]

        # cKDTree uses euclidean distance? Find a way to search angular distances instead

        virial_tree = cKDTree(velocity_bin[:,:2])
        transverse_virial_idx = virial_tree.query_ball_point(center[:2], linear_to_angular_dist(virial_radius, center[2]).value)
        transverse_virial_points = velocity_bin[transverse_virial_idx] # select galaxies within virial radius

        if len(transverse_virial_points) >= 12: # reject if less than 12 surrounding galaxies within virial radius (to save time) 
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
                c = Cluster(center, f_points)
                candidates.append(c)

                if not i % 100:
                    print(f'{i} ' + c.__str__())
                    # print('Candidate {num} identified with {length} members'.format(num=len(candidates), length=len(f_points)))

        if len(candidates) >= 200: # for quick testing
            break


    # perform overlap removal and merger
    candidate_centers = np.array([[c.ra, c.dec, c.z, c.gal_id] for c in candidates]) # get specific attributes from candidate center space

    candidates = np.array(candidates)
    merged_candidates = candidates.copy()
    gal_id_space = [c.gal_id for c in candidates]

    for center in candidates:

        # identity overlapping centers (centers lying within virial radius of current cluster)
        velocity_bin = candidate_centers[abs(candidate_centers[:,2]-center.z) <= 0.02*(1+center.z)]
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
    center_shifted_candidates = merged_candidates.copy()

    # fig = plt.figure(figsize=(10,8))
    # for c in merged_candidates:
    #     plt.scatter(c.galaxies[:,0], c.galaxies[:,1], s=5)
    #     plt.scatter(c.ra, c.dec)
    #     print(c.__str__())
    # plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    # plt.show()

    # replace candidate center with brightest galaxy in cluster
    for center in merged_candidates:

        # identify galaxies within 0.25*virial radius
        cluster_tree = cKDTree(center.galaxies[:,:2]) # for galaxies within a cluster
        c_coords = [center.ra, center.dec]
        bcg_idx = cluster_tree.query_ball_point(c_coords, linear_to_angular_dist(0.25*virial_radius, center.z).value)
        bcg_arr = center.galaxies[bcg_idx] 

        if len(bcg_arr) and len(center.galaxies): # check there are any galaxies within 0.25*virial radius

            mag_sort = bcg_arr[bcg_arr[:,3].argsort()] # sort selected galaxies by abs mag (brightness)
            bcg = mag_sort[0] # brightest cluster galaxy (bcg)

            # if bcg brighter than current center, replace it as center
            if (abs(bcg[3]) > abs(center.bcg_brightness)) and (bcg[4] != center.gal_id):
                new_cluster = Cluster(bcg, center.galaxies) # initialize new center
                new_center = [c for c in center_shifted_candidates if c.gal_id == center.gal_id][0]

                new_center.galaxies = np.array([]) # delete old center
                center_shifted_candidates = np.concatenate((center_shifted_candidates, np.array([new_cluster]))) # add new center to array


    center_shifted_candidates = np.array([c for c in center_shifted_candidates if c.richness >= richness]) # select only clusters >= richness
    final_candidates = []

    # fig = plt.figure(figsize=(10,8))
    # for c in center_shifted_candidates:
    #     plt.scatter(c.galaxies[:,0], c.galaxies[:,1], s=5)
    #     plt.scatter(c.ra, c.dec)
    #     print(c.__str__())
    # plt.axis([min(galaxy_data[:,0]), max(galaxy_data[:,0]), min(galaxy_data[:,1]), max(galaxy_data[:,1])])
    # plt.show()

    # N(0.5) and galaxy overdensity
    for center in center_shifted_candidates:
        center.N = find_number_count(center, distance=0.5*u.Mpc/u.littleh) # find number count N(0.5)
        center.D = center_overdensity(center, galaxy_data, max_velocity) # find overdensity D

        # Initialize the cluster only if N(0.5) >= 8 and D >= overdensity
        if center.D >= overdensity and center.N >= 8 and center.richness >= richness:
            final_candidates.append(center)

    fig = plt.figure(figsize=(10,8))
    for c in final_candidates:
        plt.scatter(c.galaxies[:,0], c.galaxies[:,1], s=5)
        plt.scatter(c.ra, c.dec)
        print(c.__str__())
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
    galaxy_arr = np.loadtxt('derived_datasets\\COSMOS_galaxy_data.csv', delimiter=',')
    luminous_arr = np.loadtxt('derived_datasets\\luminous_galaxy_redshift0.02.csv', delimiter=',')

    galaxy_arr = galaxy_arr[(galaxy_arr[:,2] >= 0.5) & (galaxy_arr[:,2] <= 2.52)]
    luminous_arr = luminous_arr[(luminous_arr[:,2] >= 0.5) & (luminous_arr[:,2] <= 2.5)]
    print('Number of galaxies: {galaxy}, Number of candidates: {luminous}'.format(galaxy=len(galaxy_arr), luminous=len(luminous_arr)))

    candidates = new_FoF(galaxy_arr, luminous_arr, max_velocity=2000, linking_length_factor=0.4, virial_radius=1.5*u.Mpc/u.littleh, richness=richness, overdensity=D)
    print('{n} candidate clusters found.'.format(n=len(candidates)))

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