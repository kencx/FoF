import astropy.units as u

##############
# PARAMETERS #
##############
min_redshift = 0.5
max_redshift = 2.5

max_velocity = 2000
linking_length_factor = 0.4
virial_radius = 1.5*u.Mpc/u.littleh

richness = 25
D = 2

fname = f'analysis\\derived_datasets\\' # R{richness}_D{D}_vel\\'
plot = False