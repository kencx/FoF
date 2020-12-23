from astropy.table import Table
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import WMAP5 as cosmo
import itertools
from scipy.spatial import cKDTree, distance

'''
To-do:
- Add threshold for cluster size
- Determine linking length and virial radius: LL is mean separation, virial radius is 2Mpc
    run an analysis of mean separation lengths: in all redshift bins, calculate the mean separations of every cluster's bcg, plot on histogram

- Determine method to identify BCGs: establish a system to find FoF using the top 5 brightest galaxies in a bin (how to ensure is not interloper?)
- Method to remove interpolers
'''


data = Table.read('datasets/cosmos2015_filtered.csv')
# print(data.colnames)
data = data['NUMBER', 'ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR'] # extract only colummns we need


# sort redshift into bins
interval = 0.04 # bin intervals
bins = np.arange(0.1, 2+interval, interval, dtype=float)
data['bins'] = np.digitize(data['PHOTOZ'], bins=bins)
data = data.group_by('bins') # group table by redshift bins

num_of_bins = len(data.groups) # number of redshift bins


def linear_to_angular_dist(distance, photo_z):
    return ((distance*u.Mpc).to(u.kpc) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


def mean_separation(n, virial_radius):
    volume = 4/3 * np.pi * virial_radius**3
    density = n/volume
    return 1/(density**(1/3))

# for i, g in enumerate(data.groups):
#     mean_sep_arr[i] = mean_separation(len(g), 2)

# plt.scatter(range(len(mean_sep_arr)), mean_sep_arr)
# plt.hist(mean_sep_arr[mean_sep_arr<1.0], bins=30)
# plt.show()


# ---- KD-tree Implementation
first_group = data.groups[0]['ALPHA_J2000', 'DELTA_J2000', 'PHOTOZ', 'MR'] # extract first bin

def FoF(group, max_radius):

    '''
    Performs a Friends-of-Friends search on group to identify cluster centre (BCG) and member galaxies

    Input:
    group (arr): Table of galaxies to carry out FoF search
    max_radius (float): Threshold for maximum search radius in Mpc units

    Output:
    brightest (arr): Brightest cluster galaxy
    friend_points (arr): Array of member galaxies that are FoF to brightest
    '''        

    group.sort('MR') # sort by abs mag in MR filter
    brightest = group[1] # select brightest galaxy
    bcg_photo_z = brightest['PHOTOZ']
    brightest = np.asarray([brightest['ALPHA_J2000'], brightest['DELTA_J2000']])
    group = np.asarray(list(zip(group['ALPHA_J2000'], group['DELTA_J2000'])))
    
    tree = cKDTree(group) # form KD-tree of extracted data

    # find all galaxies within virial radius of brightest galaxy
    idx = tree.query_ball_point(brightest, linear_to_angular_dist(max_radius, bcg_photo_z).value)
    virial_points = group[idx]

    if virial_points.size == 0:
        return 'No galaxies within maximum radius'

    linking_length = mean_separation(len(virial_points), max_radius) # in Mpc units

    linking_tree = cKDTree(virial_points) # form KD-tree of data within virial radius

    # find all galaxies within linking length of brightest galaxy
    linking_idx = linking_tree.query_ball_point(brightest, linking_length)
    friend_points = virial_points[linking_idx]

    for galaxy in friend_points: # select one friend galaxy
        friends_of_idx = linking_tree.query_ball_point(galaxy, linking_length) # find all galaxies within LL of each galaxy (fof)
        fof_points = virial_points[friends_of_idx]
        
        mask = np.isin(fof_points, friend_points, invert=True)
        fof_points = fof_points[mask].reshape(-1,2) # filter points not in final array
        friend_points = np.concatenate((friend_points, fof_points)) # add all fof to final list

    return brightest, friend_points

bcg, fof_array = FoF(first_group, 1.5)


fig = plt.figure(figsize=(10,8))
plt.hist2d(first_group['ALPHA_J2000'], first_group['DELTA_J2000'], bins=(100,80))
# plt.scatter(first_group[:, 0], first_group[:, 1], color='black')
plt.scatter(fof_array[:, 0], fof_array[:, 1], color='orange')
plt.scatter(bcg[0], bcg[1], color='red')
plt.show()
