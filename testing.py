import pickle
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance, CartesianRepresentation
from astropy.cosmology import LambdaCDM
from params import *

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


with open(fname+'clusters.dat', 'rb') as f:
    virial_clusters = pickle.load(f)

coords = np.array([[c.ra, c.dec] for c in virial_clusters])

test_cluster = virial_clusters[6]
galaxies = test_cluster.galaxies

def radec2xy(coords, r):
    if len(coords) <= 2:
        ra = np.radians(coords[0])
        dec = np.radians(coords[1])
    else:
        ra = np.radians(coords[:,0])
        dec = np.radians(coords[:,1])
    X = np.cos(dec)*np.cos(ra)*r
    Y = np.cos(dec)*np.sin(ra)*r
    Z = np.sin(dec)*r
    return np.column_stack((X,Y,Z))

# galaxies_xy = radec2xy(galaxies[:,:2], r=cosmo.luminosity_distance(np.mean(galaxies[:,2])))

# plt.figure(figsize=(10,8))
# plt.scatter(galaxies_xy[:,0], galaxies_xy[:,1], s=5, alpha=0.5)
# # plt.scatter(test_cluster.ra, test_cluster.dec, s=5, alpha=0.8)
# # plt.axis([min(coords[:,0]), max(coords[:,0]), min(coords[:,1]), max(coords[:,1])])
# plt.show()

dist = cosmo.comoving_distance(test_cluster.z)

radec = SkyCoord(ra=test_cluster.ra*u.degree, dec=test_cluster.dec*u.degree, distance=Distance(dist, unit='Mpc'))
print(radec.represent_as(CartesianRepresentation))

coords = radec2xy([test_cluster.ra,test_cluster.dec], cosmo.comoving_distance(test_cluster.z))[0]
xy = SkyCoord(x=coords[0], y=coords[1], z=coords[2], unit='Mpc', representation_type='cartesian')
print(xy)