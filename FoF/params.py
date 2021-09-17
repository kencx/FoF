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
cosmo = LambdaCDM(H0=70 * u.km / u.Mpc / u.s, Om0=0.3, Ode0=0.7)  # define cosmology

# redshift range
min_redshift = 0.5
max_redshift = 2.5

# FoF key parameters
max_velocity = 2000 * u.km / u.s
linking_length_factor = 0.2  # b
max_radius = 1.5 * u.Mpc / u.littleh
richness = 12
D = 2  # overdensity


# path to save datasets
import os

runname = "COSMOS"  # used to differentiate different datasets
dpath = ".\\analysis\\result_datasets\\"
dname = f"{runname}_R{richness}_D{D}\\"
fname = dpath + dname

os.makedirs(os.path.dirname(fname), exist_ok=True)  # create dir if not already present
