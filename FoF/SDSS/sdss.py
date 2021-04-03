import numpy as np
import pandas as pd
from astroML import datasets

from params import *


def absmag(apparent,z):
    return (apparent-5*np.log10(4.28e8*z))

# -- data cleaning
# data = datasets.fetch_sdss_specgals()
# data = data[['ra','dec','z','zErr','modelMag_r','specObjID']]
# data = pd.DataFrame(data) # turn array into df

# data['absmag'] = absmag(data['modelMag_r'], data['z']) # compute abs mag
# data = data[(data['ra']<=260) & (data['ra']>=100)] # select only legacy region
# data = data[(data['absmag']<=-19.5) & (data['z'] <= 0.2+0.02)] # magnitude cut at same limit as COSMOS
# print(len(data))


# -- search for clusters
from sdss_fof import candidate_cluster_search, FoF, interloper_removal, flag_plots
fname = f'FoF\\analysis\\derived_datasets\\sdss_R{richness}_D{D}\\'

# main_arr, luminous_arr = candidate_cluster_search(data, max_velocity=max_velocity, path=fname)

# main_arr = np.loadtxt(fname+'SDSS_galaxy_data.csv', delimiter=',') 
# luminous_arr = np.loadtxt(fname+'luminous_galaxy_data.csv', delimiter=',')
# print(f'Number of galaxies: {len(main_arr)}, Number of candidates: {len(luminous_arr)}')


# candidates = FoF(main_arr, luminous_arr, max_velocity=max_velocity, linking_length_factor=linking_length_factor, 
#                 virial_radius=virial_radius, richness=richness, overdensity=D)
# print(f'{len(candidates)} candidate clusters found.')

# with open(fname+'sdss_candidates.dat', 'wb') as f:
    # pickle.dump(candidates, f)


# with open(fname+'sdss_candidates.dat', 'rb') as f:
    # candidates = pickle.load(f)
# flag_plots(candidates)

# cleaned_candidates, number_removed = interloper_removal(candidates)

# with open(fname+'sdss_cleaned_candidates.dat', 'wb') as f:
    # pickle.dump(cleaned_candidates, f)

# with open(fname+'sdss_cleaned_candidates.dat', 'rb') as f:
    # cleaned_candidates = pickle.load(f)

# -- mass analysis
from mass_analysis import find_masses, compare_masses, plot_masses

# clusters = find_masses(cleaned_candidates, 'virial')
# projected_clusters = find_masses(cleaned_candidates, 'projected')

with open(fname+'sdss_clusters.dat', 'rb') as f: # with quantities
    clusters = pickle.load(f)

# compare_masses(projected_clusters, clusters)
plot_masses(clusters, None, None)

# --
# clusters = np.array([c for c in clusters if (c.ra <=260) & (c.ra>=100) & (c.z<=0.2)])
# z_arr = np.array([c.z for c in clusters])
# properties = np.array([[c.cluster_mass.value, c.richness] for c in clusters])
# properties = properties[properties[:,0]>0]
# plt.hist(np.log10(properties[:,0]), bins='auto')
# # plt.scatter(properties[:,1], np.log10(properties[:,0]), s=5, alpha=0.75)
# plt.show()