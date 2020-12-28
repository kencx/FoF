import sqlite3
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ---
# Import astropy fits datasets to pandas df

def fits_to_df(fname):
    d = fits.open('datasets\\' + fname)
    print(d.info())
    col_num = int(input('Choose the table to import: '))
    t = Table(d[col_num].data)
    df = t.to_pandas()
    d.close()
    print('Dataframe of table ' + str(col_num) + ' initialized.')
    print(df.head())
    return df

# cosmic_web_df = fits_to_df('J_ApJ_837_16_table1.dat.gz.fits')
# deep_field_df = fits_to_df('J_ApJ_734_68_table2.dat.fits')
# xray_df = fits_to_df('')


# --- Custom data cleaning and filtering 
# KEY INFO : ['ra', 'dec', 'redshift', 'number of members/richness', 'group/galaxy id']

def drop_columns(df, col_list):
    df = df.drop(col_list, axis=1)
    return df

def check_null(df):
    return df.isnull().sum()

def drop_null(df):
    if sum(check_null(df)):
        return df.dropna()

# -- cosmic_web
# cosmic_web_df = cosmic_web_df[(cosmic_web_df['Flag']==0) & (cosmic_web_df['Ngroup'] != -99)]
# cosmic_web_df.drop(['Den', 'Oden', 'SCl', 'SFil', 'CWE', 'Flag'], axis=1, inplace=True)
# cosmic_web_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'group', 'Ngal', 'type']
# cosmic_web_df['type'] = cosmic_web_df['type'].str.rstrip()
# cosmic_web_members_df = cosmic_web_df[cosmic_web_df['type']=='satellite'] # member gal array
# cosmic_web_df = cosmic_web_df[cosmic_web_df['type']=='central'] # center gal array


# -- deep_field
# deep_field_df = deep_field_df[deep_field_df['Field']=='COSMOS']
# deep_field_df.drop(['D', 'Field', 'Cat'], axis=1, inplace=True)
# deep_field_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'zs', 'mag', 'Ngal', 'R', 'Ltot']


# -- xray
# xray_df = xray_df[]



# --- Add datasets to database, form cleaned csv file
def df_to_csv(df, fname):
    df.to_csv('datasets\\' + fname + '_cleaned.csv')
    print(fname + ' CSV file added.')


def add_to_db(df, table_name):
    conn = sqlite3.connect('galaxy_clusters.db')
    c = conn.cursor()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print('Table added to SQL DB.')

def add_data(df, name):
    df_to_csv(df, name)
    add_to_db(df, name)


# add_data(cosmic_web_df, 'cosmic_web')
# add_data(cosmic_web_members_df, 'cosmic_web_members')
# add_data(deep_field_df, 'deep_field')
# add_data(xray_df, 'xray')
# add_data(xray_members_df, 'xray_members')

