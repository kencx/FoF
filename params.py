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
max_velocity = 2000
linking_length_factor = 0.4
virial_radius = 1.5*u.Mpc/u.littleh

# cluster size parameters
richness = 25
D = 2

# path
fname = f'analysis\\derived_datasets\\' # R{richness}_D{D}_vel\\'
# plot = False