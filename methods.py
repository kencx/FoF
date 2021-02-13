
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from astropy import units as u
from astropy.constants import G
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord

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

    return (distance.to(u.Mpc, u.with_H0(cosmo.H0)) * cosmo.arcsec_per_kpc_proper(photo_z)).to(u.deg)


def mean_separation(n, radius):
    '''
    Average mean separation of n objects in an area of a given radius. Calculated by taking rho**(-1/3).

    Parameters
    ----------
    n: int
        Number of objects

    radius: float
        Radius of area

    Returns
    -------
    mean_separation: float, array-like
        Average mean separation in units of radius

    '''
    volume = 4/3 * np.pi * radius**3
    density = n/volume
    return 1/(density**(1/3))

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

    c = const.c.to('km/s')
    # v = c*((1+z)**2 - 1)/((1+z)**2 + 1) # relativistic doppler formula
    v = c*z
    return v