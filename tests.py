import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from fof_kdtree import FoF



# --- testing xray groups
gal_data = pd.read_csv('datasets/xray_group_members_cleaned.csv')
gal_data = gal_data.loc[:,['gal_id', 'ra', 'dec', 'zphot', 'group_id', 'mmggs']]

group_data = pd.read_csv('datasets/xray_group_catalog_tbl.csv')
group_data = group_data.loc[:,['group_id', 'ra', 'dec', 'redshift', 'm200c', 'r200c_mpc', 'nmem']]

gal_arr = np.asarray(gal_data.loc[:,['ra', 'dec', 'zphot', 'gal_id']])
group_arr = np.asarray(group_data.loc[:,['ra', 'dec', 'redshift', 'group_id']])

gal_arr = gal_arr[gal_arr[:,2].argsort()] # sort by redshift
group_arr = group_arr[group_arr[:,2].argsort()] # sort by redshift

gal_arr = np.hstack((gal_arr, np.zeros((gal_arr.shape[0], 1)))) # add column for galaxy velocity
group_arr = np.hstack((group_arr, np.zeros((group_arr.shape[0], 1)))) # add column for galaxy velocity

plt.hist(group_data['redshift'], bins='auto')
plt.show()

# candidate_list = FoF(gal_arr, group_arr, max_velocity=2000, linking_length_factor=0.4, virial_radius=2)

# max_nmem = group_data['nmem'].max()
# max_group = group_data[group_data['nmem'] == max_nmem]
# max_group_id = max_group['group_id'].iloc[0]
# max_gal = gal_data.loc[gal_data.group_id == max_group_id, 'ra':'dec']
# max_candidate_gal = candidate_gal_data.loc[candidate_gal_data.group_id == max_group_id, 'ra':'dec']

# plt.scatter(max_gal['ra'], max_gal['dec'], color='orange', alpha=0.5)
# plt.scatter(max_candidate_gal['ra'], max_candidate_gal['dec'], color='blue', alpha=0.7)
# plt.scatter(max_group['ra'], max_group['dec'], color='red')
# plt.show()







# ----- tested virial mass estimator
# virial_masses (M_sun)
# {(150.10754288169696, 2.5575009868093432, 0.502, -20.68, 831036.0, 115644.98275119808): 5.52462884e+15, 
# (149.5197754535226, 1.83467112187136, 0.5025, -20.505, 359023.0, 115729.9052420429): 6.17663675e+15}

# virial_radius and escape_velocity
# {(150.2773967224679, 1.7571426724087982, 0.5, -20.906, 309409.0, 115304.79153846153): (<Quantity 3.95035035 Mpc>, <Quantity 7097.60005292 km / s>), 
# (150.11257822273575, 2.5560953018745893, 0.5, -23.613, 827001.0, 115304.79153846153): (<Quantity 3.75294887 Mpc>, <Quantity 6572.29537795 km / s>)}