#!/usr/bin/env python3

from params import *
import sqlite3
import pandas as pd

import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger("matplotlib.font_manager").disabled = True

"""
Cluster Finding Pipeline
------------------------

1. Import galaxy survey data from source (sqlite db, pandas DataFrame, numpy array)
2. Data must contain columns of ['ra', 'dec', 'z', 'Abs Mag', 'ID']
3. Search for overdensities with luminous_search
4. Search for cluster candidates with FoF. Parameters are found in params.py. Candidates are pickled into fname
5. Perform interloper removal
6. (Optional) Perform manual inspection of clusters.
7. Evaluate cluster properties (incl. virial mass)
8. Final cluster sample is outputed as a pickle into fname.
"""

###########################
# IMPORT RAW DATA FROM DB #
###########################

"""
Import data of galaxies with information of (in this order):
1. RA & DEC (in degrees)
2. z
3. Absolute Magnitude
4. Galaxy ID
"""

print("Selecting galaxy survey...")
conn = sqlite3.connect("FoF\\processing\\datasets\\galaxy_clusters.db")
df = pd.read_sql_query(
    """
    SELECT *
    FROM mag_lim
    WHERE redshift BETWEEN ? AND ?
    ORDER BY redshift""",
    conn,
    params=(min_redshift, max_redshift + 0.2),
)

# print(df.columns)
df = df.loc[
    :, ["ra", "dec", "redshift", "z_lower", "z_upper+", "RMag", "ID"]
]  # select relevant columns
df = df.sort_values("redshift")  # sort by z


#############################
# LUMINOUS CANDIDATE SEARCH #
#############################
from fof import candidate_center_search

galaxy_arr, center_candidates_arr = candidate_center_search(
    df, max_velocity=max_velocity, fname="COSMOS"
)


######################
# FRIENDS-OF-FRIENDS #
######################
from fof import FoF

# replace np loadtxt with faster method for next time
galaxy_arr = np.loadtxt(
    "FoF\\analysis\\derived_datasets\\COSMOS_galaxy_data.csv", delimiter=","
)
center_candidates_arr = np.loadtxt(
    "FoF\\analysis\\derived_datasets\\COSMOS_candidate_centers.csv", delimiter=","
)

galaxy_arr = galaxy_arr[
    (galaxy_arr[:, 2] >= min_redshift) & (galaxy_arr[:, 2] <= max_redshift + 0.02)
]
center_candidates_arr = center_candidates_arr[
    (center_candidates_arr[:, 2] >= min_redshift)
    & (center_candidates_arr[:, 2] <= max_redshift)
]
print(
    f"Number of galaxies: {len(galaxy_arr)}, Number of candidates: {len(center_candidates_arr)}"
)

candidates = FoF(
    galaxy_arr,
    center_candidates_arr,
    max_velocity=max_velocity,
    linking_length_factor=linking_length_factor,
    virial_radius=max_radius,
    richness=richness,
    overdensity=D,
)
print(f"{len(candidates)} candidate clusters found.")

# pickle candidates list
with open(fname + "candidates.dat", "wb") as f:
    pickle.dump(candidates, f)


######################
# INTERLOPER REMOVAL #
######################
with open(fname + "candidates.dat", "rb") as f:
    candidates = pickle.load(f)

cleaned_candidates, number_removed = interloper_removal(candidates)

# check interloper removal
plt.hist(number_removed, bins=10)
plt.show()

with open(fname + "cleaned_candidates.dat", "wb") as f:
    pickle.dump(cleaned_candidates, f)


####################
# CLUSTER FLAGGING #
####################
from helper import plot_clusters

# manually flag clusters near edges or unvirialized (bad clusters)
with open(fname + "cleaned_candidates.dat", "rb") as f:
    cleaned_candidates = pickle.load(f)

plot_clusters(cleaned_candidates, [min_redshift, max_redshift])


####################
# CATALOG MATCHING #
####################
from catalog_matching import compare_clusters

with open(fname + "cleaned_candidates_flagged.dat", "rb") as f:
    cleaned_candidates = pickle.load(f)

conn = sqlite3.connect("FoF\\processing\\datasets\\galaxy_clusters.db")

lensing_df = pd.read_sql_query(
    """
    SELECT RAdeg, DEdeg, z, Rich
    FROM lensing
    WHERE Rich>=? AND z>=0.5 AND (RAdeg BETWEEN ? AND ?) AND (DEdeg BETWEEN ? AND ?)
    ORDER BY z
    """,
    conn,
    params=(richness, lims[0], lims[1], lims[2], lims[3]),
)
lensing_arr = lensing_df.values
conn.close()

lensing_matched, lensing_catalog_idx = compare_clusters(
    virial_clusters, lensing_arr, 0.5 * u.Mpc / u.littleh, richness_plot=False
)


#################
# MASS ANALYSIS #
#################
from mass_analysis import find_masses

with open(fname + "cleaned_candidates_flagged.dat", "rb") as f:
    cleaned_candidates = pickle.load(f)

cleaned_candidates = [c for c in cleaned_candidates if c.flag_poor == False]

virial_clusters = find_masses(cleaned_candidates, "virial")

with open(fname + "clusters.dat", "wb") as f:  # with quantities
    pickle.dump(virial_clusters, f)
