import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import WMAP5 as cosmo
import astropy.constants as const

import itertools

def redshift_to_velocity(z):
    '''
    Converts redshift to LOS velocity

    Parameters
    ----------
    z: array-like, float
        Redshift
    
    Returns
    -------
    v: array-like, float
        Velocity in m/s
    '''

    c = const.c.to('m/s')
    v = c*((1+z)**2 - 1)/((1+z)**2 + 1)
    return v

# calculate velocity dispersion for an array of velocities
def velocity_dispersion(v_arr, group_size):
    # return np.sum(v_arr**2)/group_size
    return np.sum(v_arr**2)/(group_size-1)

# evaluate projected virial radius
def projected_radius(group_size, separation):
    return group_size*(group_size-1)*((1/separation).to(u.m))

# evaluate cluster mass
def cluster_mass(cluster_vel_disp, projected_radius):
    mass = (3/2)*np.pi*(cluster_vel_disp*projected_radius)/const.G
    return mass.to(u.M_sun)


def virial_mass_estimator(cluster_members): # speed this up further? INCLUDE MASS WEIGHTING

    ''' Calculates virial masses of galaxy clusters with the virial mass formula

    Parameters
    ----------
    cluster: array-like
        Array of cluster members with key properties: ['ra', 'dec', 'redshift']
    
    Returns
    -------
    mass: float
        Mass of cluster in M_sun

    cluster_vel_disp: float
        Estimated velocity dispersion of cluster in (m/s)**2

    cluster_radius: float
        Estimated radius of cluster in m
    '''

    # velocity dispersion
    cluster_size = len(cluster_members)
    photoz_arr = cluster_members[:,2]

    average_redshift = np.mean(photoz_arr)
    cluster_vel = redshift_to_velocity(photoz_arr) - redshift_to_velocity(average_redshift)
    cluster_vel_disp = velocity_dispersion(cluster_vel, cluster_size)

    # galaxy separations
    c = SkyCoord(ra=cluster_members[:,0]*u.degree, dec=cluster_members[:,1]*u.degree) # array of coordinates
    
    pairs_idx = np.asarray(list((i,j) for ((i,_), (j,_)) in itertools.combinations(enumerate(c), 2))) # index of gals in combinations
    photoz_pairs = np.take(photoz_arr, pairs_idx) # photoz values of gals from the indices
    d_A = cosmo.angular_diameter_distance(z=photoz_pairs[:,0]) # angular diameter distance

    pairs = c[pairs_idx]
    total_separation = 0

    projected_sep = pairs[:,0].separation(pairs[:,1]) # projected separation (in deg)
    actual_sep = (np.multiply(projected_sep,d_A)).to(u.Mpc, u.dimensionless_angles()) # convert projected separation to Mpc
    total_separation = sum(1/actual_sep)
    
    cluster_radius = projected_radius(cluster_size, total_separation)
    mass = cluster_mass(cluster_vel_disp, cluster_radius)

    return mass, cluster_vel_disp, cluster_radius


def projected_mass_estimator(cluster_center, cluster_members):
    N = len(cluster_members)

    average_redshift = np.mean(cluster_members[:,2])
    cluster_velocity = redshift_to_velocity(cluster_members[:,2]) - redshift_to_velocity(average_redshift) # in m/s

    c = SkyCoord(ra=cluster_members[:,0]*u.degree, dec=cluster_members[:,1]*u.degree)
    center = SkyCoord(ra=cluster_center[:,0]*u.degree, dec=cluster_center[:,1]*u.degree)

    cluster_separation = center.separation(c)
    d_A = cosmo.angular_diameter_distance(z=cluster_members[:,2])
    actual_separation = (cluster_separation*d_A).to(u.m, u.dimensionless_angles())

    sumproduct = np.sum(actual_separation*cluster_velocity**2)
    projected_mass = (32/np.pi)*(1/const.G)*(1/(N-1.5))*sumproduct

    return projected_mass.to(u.M_sun)


if __name__ == "__main__":

    pass


    