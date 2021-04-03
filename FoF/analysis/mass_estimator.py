#!/usr/bin/env python3

import itertools
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

from analysis.methods import redshift_to_velocity

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


# calculate velocity dispersion for an array of velocities
def velocity_dispersion(v_arr, group_size):
    return np.sum(v_arr**2)/(group_size-1)

# evaluate projected virial radius
def projected_radius(group_size, separation):
    return group_size*(group_size-1)/separation

# evaluate cluster mass
def cluster_mass(cluster_vel_disp, projected_radius):
    mass = (3/2)*np.pi*(cluster_vel_disp*projected_radius)/const.G
    return mass.to(u.M_sun/u.littleh)


def virial_mass_estimator(cluster):
    ''' 
    Calculates virial masses of galaxy clusters with the virial mass formula

    Parameters
    ----------
    cluster: Cluster object
    
    Returns
    -------
    mass: float
        Mass of cluster in M_sun

    cluster_vel_disp: float
        Estimated velocity dispersion of cluster in (km/s)**2

    cluster_radius: float
        Estimated radius of cluster in Mpc
    '''
    member_galaxies = cluster.galaxies

    # velocity dispersion
    cluster_size = len(member_galaxies)
    z_arr = member_galaxies[:,2]

    average_redshift = np.mean(z_arr)
    cluster_vel = redshift_to_velocity(z_arr, average_redshift)
    cluster_vel_err = velocity_error(cluster, z_arr, average_redshift, cluster_vel)
    cluster_vel_disp = velocity_dispersion(cluster_vel, cluster_size)

    bootfunc = lambda x: (velocity_dispersion(x, len(x)))
    bootstrap_disp = np.std(bootstrapping(cluster_vel.value, bootfunc))
    harmonic_disp = (1/cluster_size)*(sum(1/(cluster_vel_err**2)))**(-1)
    combined_disp_err = (np.sqrt(harmonic_disp.value**2 + (bootstrap_disp)**2))*(u.km**2/u.s**2)

    # galaxy separations
    c = SkyCoord(ra=member_galaxies[:,0]*u.degree, dec=member_galaxies[:,1]*u.degree) # array of coordinates
    
    pairs_idx = np.asarray(list((i,j) for ((i,_), (j,_)) in itertools.combinations(enumerate(c), 2))) # index of galaxies in combinations
    photoz_pairs = np.take(z_arr, pairs_idx) # photoz values of galaxies from the indices
    d_A = cosmo.angular_diameter_distance(z=average_redshift) # angular diameter distance

    pairs = c[pairs_idx]

    projected_sep = pairs[:,0].separation(pairs[:,1]) # projected separation (in deg)
    actual_sep = (projected_sep*d_A).to(u.Mpc, u.dimensionless_angles()) # convert projected separation to Mpc
    actual_sep = ((actual_sep*((cosmo.H0/100).value))/u.littleh).to(u.Mpc/u.littleh) # include littleh scaling
    total_separation = sum(1/actual_sep)
    
    cluster_radius = projected_radius(cluster_size, total_separation)
    mass = cluster_mass(cluster_vel_disp, cluster_radius)
    mass_err = mass_error(mass, cluster_vel_disp, combined_disp_err)

    return mass, mass_err, cluster_vel_disp, combined_disp_err, cluster_radius # M_sun/littleh, km^2/s^2, Mpc/littleh


def projected_mass_estimator(cluster):

    member_galaxies = cluster.galaxies
    N = len(member_galaxies)
    z_arr = member_galaxies[:,2]
    average_redshift = np.mean(z_arr)
    cluster_vel = redshift_to_velocity(z_arr, average_redshift) # in km/s

    c = SkyCoord(ra=member_galaxies[:,0]*u.degree, dec=member_galaxies[:,1]*u.degree)
    centroid = SkyCoord(ra=np.mean(member_galaxies[:,0])*u.degree, dec=np.mean(member_galaxies[:,1])*u.degree)
    # center = SkyCoord(ra=cluster_center[:,0]*u.degree, dec=cluster_center[:,1]*u.degree)

    cluster_separation = centroid.separation(c)
    d_A = cosmo.angular_diameter_distance(z=average_redshift)
    actual_separation = (cluster_separation*d_A).to(u.Mpc, u.dimensionless_angles())
    actual_separation = (actual_separation*((cosmo.H0/100).value)/u.littleh).to(u.Mpc/u.littleh) # include littleh scaling

    sumproduct = np.sum(actual_separation*cluster_vel**2)
    projected_mass = (32/np.pi)*(1/const.G)*(1/(N-1.5))*sumproduct

    vel_err = velocity_error(cluster, z_arr, average_redshift, cluster_vel)
    mass_err = np.sqrt(sum((2*cluster_vel*vel_err)**2))

    return projected_mass.to(u.M_sun/u.littleh), mass_err


def velocity_error(cluster, z_arr, z_average, velocities):
    galaxies = cluster.galaxies
    galaxies_z_err = abs(galaxies[:,3:5]-z_arr[:,np.newaxis]) # upper and lower error

    # take the maximum error range
    galaxies_z_err = np.amax(galaxies_z_err, axis=1) 

    # calculate numerator and denominator errors
    z_average_err = np.sqrt(sum(galaxies_z_err**2))/len(z_arr) # standard error of mean
    z_diff_err = np.sqrt(np.add(z_average_err**2,galaxies_z_err**2))

    # combined error of velocities
    err = abs(velocities)*np.sqrt((z_average_err/z_average)**2 + ((z_diff_err)/(z_arr-z_average))**2)
    return err


def mass_error(mass, vel_disp, vel_disp_err):
    return mass*(vel_disp_err/vel_disp)


def log_error(error, quantity):
    return 0.434*(error/quantity)


def bootstrapping(bootarr, bootfunc):
    # gives multiple velocity dispersion
    with NumpyRNGContext(1):
        bootresult = bootstrap(bootarr, bootnum=100, samples=len(bootarr)-1, bootfunc=bootfunc)
    return bootresult

    