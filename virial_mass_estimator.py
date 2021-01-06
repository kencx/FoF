import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import WMAP5 as cosmo
import astropy.constants as const

import itertools
import time
import sqlite3

from data_processing import split_df_into_groups
from scipy.stats import linregress


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


def virial_mass_estimator(cluster_members, cluster_redshift): # speed this up further? INCLUDE MASS WEIGHTING

    ''' Calculates virial masses of galaxy clusters with the virial mass formula

    Parameters
    ----------
    cluster: array-like
        Array of cluster members with key properties: ['ra', 'dec', 'redshift']
    
    cluster_redshift: float
        Median redshift of cluster members

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
    cluster_velocity = redshift_to_velocity(cluster_members[:,2]) - redshift_to_velocity(average_redshift)

    c = SkyCoord(ra=cluster_members[:,0]*u.degree, dec=cluster_members[:,1]*u.degree)
    center = SkyCoord(ra=cluster_center[:,0]*u.degree, dec=cluster_center[:,1]*u.degree)

    cluster_separation = center.separation(c)
    d_A = cosmo.angular_diameter_distance(z=cluster_members[:,2])
    actual_separation = (np.multiply(cluster_separation,d_A)).to(u.Mpc, u.dimensionless_angles())

    sumproduct = np.sum(actual_separation*cluster_velocity**2)
    projected_mass = (32/np.pi)*(1/const.G)*(1/(N-1.5))*sumproduct

    return projected_mass.to(u.M_sun)


if __name__ == "__main__":

    # test virial_mass estimator on largest group in cosmic_web

    # conn = sqlite3.connect('galaxy_clusters.db')
    # test_df = pd.read_sql_query('''
    #     SELECT *
    #     FROM cosmic_web_members
    #     WHERE (redshift > 0.1) and (redshift < 0.990)''', conn)
    # arr, group_n = split_df_into_groups(test_df.iloc[:,:5], 'group')

    # largest_idx = np.argmax([len(arr[arr[:,-1]==n]) for n in group_n])
    # largest_n_idx = group_n[largest_idx]
    # largest_arr = arr[arr[:,-1]==largest_n_idx]

    # mass, vel_disp, rad = virial_mass_estimator(largest_arr[:,1:4], np.median(largest_arr[:,3]))
    # print(mass)

    bcg = pd.read_csv('0.1-0.9_bcg (interloper removed).csv')
    df = pd.read_csv('0.1-0.9_members (interloper removed).csv')
    bcg_arr = bcg.values
    arr, group_n = split_df_into_groups(df, 'cluster_id')

    bcg1 = pd.read_csv('0.99-1.5_candidate_bcg.csv')
    bcg_df1 = pd.read_csv('0.99-1.5_candidate_members.csv')
    bcg_arr1 = bcg1.values
    arr1, group_n1 = split_df_into_groups(bcg_df1, 'cluster_id')


    # n = group_n[10]
    # bcg_ = bcg_arr[bcg_arr[:,-2]==n]
    # test_arr = arr[arr[:,-1]==n]
    # projected_mass = projected_mass_estimator(bcg_, test_arr)
    # virial_mass = virial_mass_estimator(test_arr, np.median(test_arr[:,2]))
    # print(projected_mass, virial_mass[0])

    virial_masses = np.zeros(group_n1.shape)
    projected_masses = np.zeros(group_n1.shape)

    # for g in group_n:
    #     cluster = arr[arr[:,-1]==g]
    #     center = bcg_arr[bcg_arr[:,-2]==g]
        # projected_masses[int(g)] = projected_mass_estimator(center, cluster).value
    #     virial_masses[int(g)] = virial_mass_estimator(cluster, np.median(cluster[:,2]))[0].value

    # for g in group_n1:
    #     cluster = arr1[arr1[:,-1]==g]
    #     center = bcg_arr1[bcg_arr1[:,-2]==g]
    #     projected_masses[int(g)] = projected_mass_estimator(center, cluster).value
    #     virial_masses[int(g)] = virial_mass_estimator(cluster, np.median(cluster[:,2]))[0].value

    # np.savetxt(fname='projected_masses1.txt', X=projected_masses)
    # np.savetxt(fname='virial_masses1.txt', X=virial_masses)

    projected_masses = np.loadtxt('projected_masses.txt')
    virial_masses = np.loadtxt('virial_masses.txt')
    projected_masses1 = np.loadtxt('projected_masses1.txt')
    virial_masses1 = np.loadtxt('virial_masses1.txt')
    
    total_bcg_arr = np.concatenate((bcg_arr, bcg_arr1))
    total_virial = np.concatenate((virial_masses, virial_masses1))
    total_projected = np.concatenate((projected_masses, projected_masses1))

    m, c, r, _, _, = linregress(total_projected, total_virial)
    print(m,c,r)
    
    fig, ax = plt.subplots()
    ax.scatter(total_projected, total_virial, s=10)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]

    ax.plot(lims, lims, 'k-', alpha=0.75)
    X = np.linspace(np.min(ax.get_xlim()), np.max(ax.get_xlim()))
    ax.plot(X, m*X+c, 'k--')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('projected masses')
    ax.set_ylabel('virial masses')
    ax.set_yscale('log')
    ax.set_xscale('log')

    # ax.plot(total_bcg_arr[:,-1], total_virial, '.')
    plt.show()
    