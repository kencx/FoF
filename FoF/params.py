#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import LambdaCDM


##############
# PARAMETERS #
##############

# cosmology
cosmo = LambdaCDM(H0=70*u.km/u.Mpc/u.s, Om0=0.3, Ode0=0.7) # define cosmology

# redshift range
min_redshift = 0.5
max_redshift = 2.5

# FoF key parameters
max_velocity = 2000*u.km/u.s
linking_length_factor = 0.8
virial_radius = 1.5*u.Mpc/u.littleh

# cluster size parameters
richness = 12
D = 2

# path
fname = f'FoF\\analysis\\derived_datasets\\R{richness}_D{D}_vel\\'
# plot = False