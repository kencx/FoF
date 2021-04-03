#!/usr/bin/env python3

import numpy as np
from scipy.integrate import quad
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology

def linear_to_angular_dist(distance, photo_z):
    '''
    Converts proper distance (Mpc) to angular distance (deg). Used to find angular separations within clusters.

    Parameters
    ----------
    distance: float, array-like
        Distance in h^-1 * Mpc

    photo_z: float, array-like
        Associated redshift

    Returns
    -------
    d: float, array-like
        Angular distance in deg
    '''

    return (distance.to(u.Mpc, u.with_H0(cosmo.H0))*cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


def mean_separation(n, z, max_dist, max_velocity, survey_area):
    '''
    Average mean separation at redshift z of n galaxies within a maximum radius of max_dist and velocity range of +- max_velocity (equivalent to redshift bin, dz).

    Parameters
    ----------
    n: int
        Number of objects

    z: float
        redshift

    max_dist: float
        maximum radius in angular units

    max_velocity: float
        velocity range in velocity units

    Returns
    -------
    mean_separation: float, array-like
        Average mean separation in Mpc

    '''
    dz = (max_velocity.to('km/s'))/(const.c.to('km/s'))

    temp = lambda x: cosmo.differential_comoving_volume(x).value
    solid_angle = (np.pi*(survey_area*u.deg)**2).to(u.sr)
    V = ((quad(temp, z, z+dz)[0])*u.Mpc**3/u.sr)*solid_angle
    ndensity = (n/V).to(u.Mpc**-3)
    return ndensity**(-1/3)
    

def redshift_to_velocity(z, center_z):
    '''
    Converts redshift to LOS velocity

    Parameters
    ----------
    z: array-like, float
        Redshift
    center_z: float
        Redshift of cluster center or cluster average
    
    Returns
    -------
    v: array-like, float
        Velocity in km/s
    '''

    c = const.c.to('km/s')
    v = (c*z - c*center_z)/(1+center_z)
    return v
