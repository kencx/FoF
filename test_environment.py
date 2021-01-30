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
from mass_estimator import virial_mass_estimator

# Query databases


conn = sqlite3.connect('galaxy_clusters.db')

cosmic_web_bcg = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, cluster_id
    FROM cosmic_web_bcg
    WHERE Ngal >= 20
    ORDER BY redshift
    ''', conn)
cosmic_web_bcg_arr = cosmic_web_bcg.values

cosmic_web_mem = pd.read_sql_query('''
    SELECT ra, dec, redshift, Ngal, cluster_id
    FROM cosmic_web_members
    WHERE Ngal >= 20
    ORDER BY redshift
    ''', conn)
cosmic_web_mem_arr = cosmic_web_mem.values

gal_weight_bcg = pd.read_sql_query('''
    SELECT *
    FROM gal_weight
    WHERE N200 >= 20
    ORDER BY redshift
    ''', conn)
gal_weight_arr = gal_weight_bcg.values

conn.close()

# ids = np.unique(cosmic_web_bcg_arr[:,-1])
# masses = np.zeros(len(ids))

# for idx, i in enumerate(ids):
#     center = cosmic_web_bcg_arr[cosmic_web_bcg_arr[:,-1]==i]
#     cluster = cosmic_web_mem_arr[cosmic_web_mem_arr[:,-1]==i]
#     mass, _, _ = virial_mass_estimator(cluster)
#     masses[idx] = mass.value


# bins = np.arange(0.19, 1.2+0.2, 0.2)
# digitized = np.digitize(cosmic_web_bcg_arr[:,2], bins, right=True)
# mass_curve = []

# for i in range(1,len(bins)+1):
#     bin_mass = masses[np.where(digitized==i)]

#     if len(bin_mass) > 1:
#         median_bin_mass = np.mean(bin_mass)
#         mass_curve.append(median_bin_mass)
#     elif len(bin_mass) == 1:
#         mass_curve.append(bin_mass[0])
#     else:
#         mass_curve.append(0)

# mass_curve = np.array(mass_curve)
# assert len(mass_curve) == len(bins)

plt.figure()
# plt.scatter(cosmic_web_bcg_arr[:,2], np.log10(masses), s=8)
# plt.plot(bins, np.log10(mass_curve), 'r--')
plt.scatter(gal_weight_arr[:,3], np.log10(gal_weight_arr[:,8]), s=8)
plt.show()