import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from virial_mass_estimator import virial_mass_estimator as vm_estimator
from virial_mass_estimator import redshift_to_velocity
from fof_kdtree_3 import FoF
from astropy.table import Table

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

candidate_list = FoF(gal_arr, group_arr, max_velocity=3000, linking_length_factor=0.4, virial_radius=2)

# form FoF table of group and galaxy data
gal_columns = ['gal_id', 'ra', 'dec', 'zphot', 'group_id']
group_columns = ['group_id', 'ra', 'dec', 'redshift', 'nmem']
candidate_gal_data = pd.DataFrame(columns=gal_columns).fillna(0)
candidate_group_data = pd.DataFrame(columns=group_columns).fillna(0)

for i, (k,v) in enumerate(candidate_list.items()):
    candidate_group_data.loc[i, ['ra', 'dec', 'redshift']] = k[0:3]
    candidate_group_data.loc[i, 'group_id'] = k[3]
    candidate_group_data.loc[i, 'nmem'] = len(v)

    temp_arr = np.zeros((v.shape))
    temp_arr[:,0] = v[:,3]
    temp_arr[:,1:4] = v[:,:3]
    temp_arr[:,-1] = k[3]
    temp = pd.DataFrame(data=temp_arr ,columns=gal_columns)
    candidate_gal_data = candidate_gal_data.append(temp)


accuracy = []
for idx, row in group_data.iterrows():
    temp_row = candidate_group_data[candidate_group_data['group_id'] == row['group_id']]
    if len(temp_row) != 0:    
        candidate_num = temp_row.iloc[0]['nmem']
        num = row['nmem']
        acc = (candidate_num/num)*100
        accuracy.append(acc)

print(np.mean(accuracy))

# plt.hist(accuracy, bins='auto')
# plt.show()
# print(1)

max_nmem = group_data['nmem'].max()
max_group = group_data[group_data['nmem'] == max_nmem]
max_group_id = max_group['group_id'].iloc[0]
max_gal = gal_data.loc[gal_data.group_id == max_group_id, 'ra':'dec']
max_candidate_gal = candidate_gal_data.loc[candidate_gal_data.group_id == max_group_id, 'ra':'dec']

plt.scatter(max_gal['ra'], max_gal['dec'], color='orange', alpha=0.5)
plt.scatter(max_candidate_gal['ra'], max_candidate_gal['dec'], color='blue', alpha=0.7)
plt.scatter(max_group['ra'], max_group['dec'], color='red')
plt.show()

