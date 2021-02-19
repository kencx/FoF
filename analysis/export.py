import numpy as np
import pandas as pd

def create_df(lst, mode):

    if mode == 'before_interloper':
        columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'LR', 'N(0.5)']
        member_columns = ['ra', 'dec', 'redshift', 'brightness', 'gal_id', 'LR', 'cluster_id']

    elif mode == 'after_interloper':
        columns = ['ra', 'dec', 'redshift', 'brightness', 'LR', 'N(0.5)', 'total_N', 'gal_id', 'cluster_id']
        member_columns = ['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'cluster_id']
    
    else:
        return Exception('No mode stated')

    bcg_df = pd.DataFrame(columns=columns).fillna(0)
    member_df = pd.DataFrame(columns=columns).fillna(0)

    for i, c in lst:
        bcg_df.loc[i, columns] = c.center_attributes()
        bcg_df.loc[i, 'total_N'] = c.richness()
        if mode == 'before_interloper':
            bcg_df.loc[i, 'cluster_id'] = i
        
        N = c.galaxies.shape
        temp_arr = np.zeros((N[0], N[1]))
        if mode == 'before_interloper':
            temp_arr[:,:-1] = c.galaxies[:,:-1]
            temp_arr[:,-1] = i
            temp = pd.DataFrame(data=temp_arr, columns=member_columns)

        elif mode == 'after_interloper':
            temp_arr = c.galaxies
            temp = pd.DataFrame(data=temp_arr, columns=member_columns)

        member_df = member_df.append(temp)
        
    
    bcg_df = bcg_df.sort_values('redshift')
    bcg_df = bcg_df[['ra', 'dec', 'redshift', 'brightness', 'LR', 'N(0.5)', 'total_N', 'gal_id', 'cluster_id']]
    member_df = member_df[['ra', 'dec', 'redshift', 'brightness', 'LR', 'gal_id', 'cluster_id']]

    return bcg_df, member_df

