import numpy as np
from astropy.table import Table, setdiff, join, unique, vstack
import virial_mass_estimator as est
import matplotlib.pyplot as plt
from scipy.stats import linregress as lr

'''
Dataset: MASS COMPLETE ENVIRONMENT CATALOG
Attach central galaxy to each member galaxy based on their group_id
Sort the table by groups
To be inserted into virial mass estimator function
'''

# path = 'C:\\Users\\Kenneth\\Desktop\\fyp\\datasets'
data = Table.read('datasets/mass_complete_env_cleaned.csv')

central = np.where(data['galaxy_type'] == 'central')
central_galaxies = data[central] # 333 unique central galaxies

satellite = np.where(data['galaxy_type'] == 'satellite')
satellite_galaxies = data[satellite] # 7221 unique satellite galaxies


# check for missing groups
missing_groups_gal_id = setdiff(central_galaxies, satellite_galaxies, keys='group_id')['id']
mask = np.logical_and.reduce([central_galaxies['id'] != missing_groups_gal_id]) # remove independent central galaxy
central_galaxies = central_galaxies[mask] # 332 central galaxies with groups

joined_satellite_galaxies = join(satellite_galaxies, central_galaxies['id', 'group_id'], keys='group_id') # add central galaxy id column to satellite galaxies
joined_satellite_galaxies.rename_column('id_2', 'ccg')
joined_satellite_galaxies.rename_column('id_1', 'id')
central_galaxies.remove_columns(['col0', 's_cluster', 's_filament', 'galaxy_type', 'environment']) # remove unnecessary columns
joined_satellite_galaxies.remove_columns(['col0', 's_cluster', 's_filament', 'galaxy_type'])

combined_galaxies = vstack([joined_satellite_galaxies, central_galaxies]) # 7426 total galaxies
gal_groups = combined_galaxies.group_by('group_id')
# print(len(gal_groups.groups))


# To-do: Form table of cluster properties

mass_data = np.loadtxt('mass_complete_group_masses.txt')
mask = np.where(mass_data < 1e17)
avg_photoz_data = gal_groups.groups.aggregate(np.mean)['photo_z']

# linear regression
x, y = avg_photoz_data[mask], np.log10(mass_data[mask])
m, c, r_sq, _, stderror = lr(x, y)

print(r_sq)
fig = plt.figure(figsize=(8,6))
plt.plot(x, m*x+c, 'b')
plt.fill_between(x, (m*x+c)-stderror, (m*x+c)+stderror, alpha=0.5)
plt.scatter(x, y, s=5, color='r')
plt.gca().set(xlabel='Redshift', ylabel='Mass (solar masses)')#, yscale='log')
plt.show()


