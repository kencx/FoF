from methods import fits_to_df, add_data

# DATA PIPELINE FOR FITS -> DF -> SQLITE3 DATABASE


# --- import fits files to df
import pandas as pd
test_df = pd.read_csv('FoF\\processing\\datasets\\cosmos2015_dataset.csv')
print(test_df.columns)
# cosmic_web_df = fits_to_df('J_ApJ_837_16_table1.dat.gz.fits')
# deep_field_df = fits_to_df('J_ApJ_734_68_table2.dat.fits')
# ultra_deep_df = fits_to_df('J_MNRAS_423_2436_table1.dat.fits')


# --- Custom data cleaning and filtering 
# KEY INFO : ['ra', 'dec', 'redshift', 'number of members/richness', 'group/galaxy id']


# --- cosmic_web
# cosmic_web_df = cosmic_web_df[(cosmic_web_df['Ngroup'] != -99)]
# cosmic_web_df.drop(['SCl', 'SFil', 'Flag'], axis=1, inplace=True)
# cosmic_web_df.columns = ['COSMOS2015_id', 'ra', 'dec', 'redshift', 'density', 'overdensity', 'environment', 'cluster_id', 'Ngal', 'type']
# cosmic_web_df['type'] = cosmic_web_df['type'].str.rstrip()
# cosmic_web_members_df = cosmic_web_df[cosmic_web_df['type']!='isolated'] # member gal array
# cosmic_web_df = cosmic_web_df[cosmic_web_df['type']=='central'] # center gal array


# --- deep_field
# deep_field_df = deep_field_df[deep_field_df['Field']=='COSMOS']
# deep_field_df.drop(['D', 'Field', 'Cat'], axis=1, inplace=True)
# deep_field_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'zs', 'mag', 'Ngal', 'R', 'Ltot']


# --- ultra_deep
# ultra_deep_df.drop(['e_z', 'Area'], axis=1, inplace=True)
# ultra_deep_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'R', 'number_density', 'L4']


# --- Add datasets to database, form cleaned csv file
# add_data(cosmic_web_df, 'cosmic_web_bcg')
# add_data(cosmic_web_members_df, 'cosmic_web_members')
# add_data(deep_field_df, 'deep_field')
# add_data(ultra_deep_df, 'ultra_deep')