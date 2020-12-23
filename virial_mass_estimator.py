import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP5 as cosmo
import astropy.constants as const
import itertools


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


def virial_mass_estimator(cluster, savetxt=True):

    '''
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
    # photoz_arr = np.array(cluster['photo_z']) # array of photo z
    photoz_arr = cluster[:,2]
    cluster_mean_photo_z = np.mean(photoz_arr)

    cluster_doppler_z = photoz_arr - cluster_mean_photo_z
    cluster_vel = redshift_to_velocity(cluster_doppler_z)
    cluster_vel_disp = velocity_dispersion(cluster_vel)

    # galaxy separations
    # c = SkyCoord(ra=cluster['ra']*u.degree, dec=cluster['dec']*u.degree) # array of coordinates
    c = SkyCoord(ra=cluster[:,0]*u.degree, dec=cluster[:,1]*u.degree) # array of coordinates


    pairs_idx = np.asarray(list((i,j) for ((i,_), (j,_)) in itertools.combinations(enumerate(c), 2))) # index of gals in combinations
    photoz_pairs = np.take(photoz_arr, pairs_idx) # photoz values of gals from the indices
    d_A = cosmo.angular_diameter_distance(z=photoz_pairs)

    pairs = np.asarray(list(itertools.combinations(c,2))) # pairwise combinations
    total_separation = 0

    for idx, pair in enumerate(pairs):
        sep = pair[0].separation(pair[1])
        actual_sep = (sep*d_A[idx, 0]).to(u.Mpc, u.dimensionless_angles())
        total_separation += (1/actual_sep)

    cluster_size = len(cluster)
    cluster_radius = projected_radius(cluster_size, total_separation)

    mass = cluster_mass(cluster_vel_disp, cluster_radius)

    return mass, cluster_vel_disp, cluster_radius