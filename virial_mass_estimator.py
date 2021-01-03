import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP5 as cosmo
import astropy.constants as const

import itertools
import time
import sqlite3


# convert photo_z to los_vel
def redshift_to_velocity(z):
    c = const.c.to('m/s')
    v = c*((1+z)**2 - 1)/((1+z)**2 + 1)
    return v

# calculate velocity dispersion for an array of velocities
def velocity_dispersion(v_arr):
    return np.sum(v_arr**2)/len(v_arr)

# evaluate projected virial radius
def projected_radius(group_size, separation):
    return group_size*(group_size-1)*(1/separation).to(u.m)

# evaluate cluster mass
def cluster_mass(cluster_vel_disp, projected_radius):
    mass = (3/2)*np.pi*(cluster_vel_disp*projected_radius)/const.G
    return mass.to(u.M_sun)


def virial_mass_estimator(cluster, cluster_redshift):

    '''
    Can I speed up this function even more?
    INCLUDE MASS WEIGHTING!!!

    Uses the virial mass estimator formula to calculate virial mass

    Input:
    cluster (arr): Array of galaxies in one cluster with the following properties: ra, dec, photo_z

    Output: 
    mass (float): Mass of cluster in M_sun units
    cluster_vel_disp (float): Velocity dispersion of cluster in m^2/s^2 units
    cluster_radius (float): Radius of cluster in m units
    '''
    # velocity dispersion
    photoz_arr = cluster[:,2]

    cluster_doppler_z = photoz_arr - cluster_redshift
    cluster_vel = redshift_to_velocity(cluster_doppler_z)
    cluster_vel_disp = velocity_dispersion(cluster_vel)

    # galaxy separations
    c = SkyCoord(ra=cluster[:,0]*u.degree, dec=cluster[:,1]*u.degree) # array of coordinates
    
    pairs_idx = np.asarray(list((i,j) for ((i,_), (j,_)) in itertools.combinations(enumerate(c), 2))) # index of gals in combinations
    photoz_pairs = np.take(photoz_arr, pairs_idx) # photoz values of gals from the indices
    d_A = cosmo.angular_diameter_distance(z=photoz_pairs[:,0])

    pairs = c[pairs_idx]
    total_separation = 0

    sep = pairs[:,0].separation(pairs[:,1])
    actual_sep = (sep*d_A).to(u.Mpc, u.dimensionless_angles())
    total_separation = sum(1/actual_sep)

    cluster_size = len(cluster)
    cluster_radius = projected_radius(cluster_size, total_separation)
    mass = cluster_mass(cluster_vel_disp, cluster_radius)

    return mass, cluster_vel_disp, cluster_radius


def split_df_into_groups(df, column): # group index is last column
    arr = df.sort_values(column).values
    group_n = np.unique(arr[:,-1])

    # groups = temp_arr[:,-1]
    # parameters = temp_arr[:,:-1]
    # ukeys, index = np.unique(groups, True)
    # group_arrays = np.split(parameters, index[1:])
    # df_groups = pd.DataFrame({'groups': ukeys, 'parameters': [a for a in group_arrays]})

    return arr, group_n


if __name__ == "__main__":

    # test virial_mass estimator on largest group in cosmic_web

    conn = sqlite3.connect('galaxy_clusters.db')
    test_df = pd.read_sql_query('''
        SELECT *
        FROM cosmic_web_members
        WHERE (redshift > 0.1) and (redshift < 0.990)''', conn)
    arr, group_n = split_df_into_groups(test_df.iloc[:,:5], 'group')

    largest_idx = np.argmax([len(arr[arr[:,-1]==n]) for n in group_n])
    largest_n_idx = group_n[largest_idx]
    largest_arr = arr[arr[:,-1]==largest_n_idx]

    mass, vel_disp, rad = virial_mass_estimator(largest_arr[:,1:4], np.median(largest_arr[:,3]))
    print(mass)


