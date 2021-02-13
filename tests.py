import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
import astropy.constants as const
from astropy.coordinates import Distance

from FoF_algorithm import linear_to_angular_dist
# from mass_estimator import virial_mass_estimator

cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology


R = 25
D = 4
df = pd.read_csv('derived_datasets\\COSMOS_galaxy_data.csv')
print('Number of galaxies: {len}'.format(len=len(df)))

luminous_df = pd.read_csv('derived_datasets\\luminous_galaxy_redshift0.02.csv')
print('Number of luminous galaxies: {len}'.format(len=len(luminous_df)))

fname = 'derived_datasets\\R{r}_D{d}_0.02\\'.format(r=R, d=D)

candidate_bcg = pd.read_csv(fname+'candidate_bcg.csv')
candidate_members = pd.read_csv(fname+'candidate_members.csv')
print('Number of candidate clusters: {len}'.format(len=len(candidate_bcg)))

bcg_df = pd.read_csv(fname+'filtered_bcg.csv')
member_df = pd.read_csv(fname+'filtered_members.csv')
print('Number of filtered clusters: {len}'.format(len=len(bcg_df)))



