import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP5 as cosmo
import astropy.constants as const
from astropy.coordinates import Distance

from FoF_algorithm import linear_to_angular_dist

# Query databases
# conn = sqlite3.connect('galaxy_clusters.db')
# df = pd.read_sql_query('''
#     SELECT ra, dec, redshift, Ngal, R, cluster_id
#     FROM deep_field 
#     WHERE (redshift > 0.1) and (redshift < 0.990)
#     ''', conn)
# conn.close()
# print(df.head())
# print(len(df))


p1 = [149.80304577142851,2.074855220760617,0.9649]
p2 = [149.81160133272112,2.080133371565722,0.9636]
d1 = Distance(unit=u.kpc, z=p1[2], cosmology=cosmo)
d2 = Distance(unit=u.kpc, z=p2[2], cosmology=cosmo)
c1 = SkyCoord(ra=p1[0]*u.degree,dec=p1[1]*u.degree,distance=d1)
c2 = SkyCoord(ra=p2[0]*u.degree,dec=p2[1]*u.degree,distance=d2)

sep = c1.separation_3d(c2).to(u.Mpc)
print(sep)

ca = SkyCoord(ra=p1[0]*u.degree,dec=p1[1]*u.degree)
cb = SkyCoord(ra=p2[0]*u.degree,dec=p2[1]*u.degree)
sep2 = ca.separation(cb)
d_A = cosmo.angular_diameter_distance(z=p1[2])
sep2 = (sep2*d_A).to(u.Mpc, u.dimensionless_angles())
print(sep2)


# if savetxt:
#         np.savetxt(fname=cluster+'_masses.txt', mass.value)

# ----- tested virial mass estimator
# virial_masses (M_sun)
# {(150.10754288169696, 2.5575009868093432, 0.502, -20.68, 831036.0, 115644.98275119808): 5.52462884e+15, 
# (149.5197754535226, 1.83467112187136, 0.5025, -20.505, 359023.0, 115729.9052420429): 6.17663675e+15}

# virial_radius and escape_velocity
# {(150.2773967224679, 1.7571426724087982, 0.5, -20.906, 309409.0, 115304.79153846153): (<Quantity 3.95035035 Mpc>, <Quantity 7097.60005292 km / s>), 
# (150.11257822273575, 2.5560953018745893, 0.5, -23.613, 827001.0, 115304.79153846153): (<Quantity 3.75294887 Mpc>, <Quantity 6572.29537795 km / s>)}