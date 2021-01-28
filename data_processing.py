import sqlite3
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table


def fits_to_df(fname):
    ''' Imports table from fits files and converts them to Pandas dataframes.

    Parameters
    ----------
    fname: str
        FITS file name

    Returns
    -------
    df: pd.Dataframe
        Pandas Dataframe
    '''

    d = fits.open('datasets\\' + fname)
    print(d.info())
    col_num = int(input('Choose the table to import: '))
    t = Table(d[col_num].data)
    df = t.to_pandas()
    d.close()
    print('Dataframe of table ' + str(col_num) + ' initialized.')
    print(df.head())
    return df


def split_df_into_groups(df, column, n):
    ''' Splits df of group members into groups

    Parameters
    ----------
    df: pd.Dataframe
        df to be split, where cluster_id must be last column (-1)

    column: str
        Name of column to sort and group by
    
    n: int
        Index of column to group by

    Returns
    -------
    arr: array-like
        Array of df rows
    
    group_n: array-like
        Array of group ids

    '''
    arr = df.sort_values(column).values
    group_n = np.unique(arr[:,n])
    
    return arr, group_n


# cosmic_web_df = fits_to_df('J_ApJ_837_16_table1.dat.gz.fits')
# deep_field_df = fits_to_df('J_ApJ_734_68_table2.dat.fits')
# gal_weight_df = fits_to_df('J_ApJS_246_2_galwcls.dat.gz.fits')

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
# cosmic_web_df = cosmic_web_df[(cosmic_web_df['Ngroup'] != -99)]
# cosmic_web_df.drop(['SCl', 'SFil', 'Flag'], axis=1, inplace=True)
# cosmic_web_df.columns = ['COSMOS2015_id', 'ra', 'dec', 'redshift', 'density', 'overdensity', 'environment', 'cluster_id', 'Ngal', 'type']
# cosmic_web_df['type'] = cosmic_web_df['type'].str.rstrip()
# cosmic_web_members_df = cosmic_web_df[cosmic_web_df['type']!='isolated'] # member gal array
# cosmic_web_df = cosmic_web_df[cosmic_web_df['type']=='central'] # center gal array


# -- deep_field
# deep_field_df = deep_field_df[deep_field_df['Field']=='COSMOS']
# deep_field_df.drop(['D', 'Field', 'Cat'], axis=1, inplace=True)
# deep_field_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'zs', 'mag', 'Ngal', 'R', 'Ltot']


# -- galweight
# gal_weight_df = gal_weight_df[['ClID', 'RAdeg', 'DEdeg', 'z', 'RV', 'r200', 'N200', 'sig200', 'M200', 'Rs', 'Ms', 'conc']]
# gal_weight_df.columns = ['cluster_id', 'ra', 'dec', 'redshift', 'RV', 'r200', 'N200', 'sig200', 'M200', 'Rs', 'Ms', 'conc']


# --- Add datasets to database, form cleaned csv file
def df_to_csv(df, fname):
    df.to_csv('datasets\\' + fname + '_cleaned.csv')
    print(fname + ' CSV file added.')


def add_to_db(df, table_name):
    ''' Adds df to sqlite3 database. Table is created if it does not exists. If table exists, data is replaced.

    Parameters
    ----------
    df: pd.Dataframe
    
    table_name: str
        Table to be added to
    
    '''
    conn = sqlite3.connect('galaxy_clusters.db')
    c = conn.cursor()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print('Table added to SQL DB.')
    conn.commit()
    conn.close()

def add_data(df, name):
    df_to_csv(df, name)
    add_to_db(df, name)


# add_data(cosmic_web_df, 'cosmic_web_bcg')
# add_data(cosmic_web_members_df, 'cosmic_web_members')
# add_data(deep_field_df, 'deep_field')
# add_data(gal_weight_df, 'gal_weight')


# --- database management

def print_tables():
    conn = sqlite3.connect('galaxy_clusters.db')
    c = conn.cursor()
    c.execute('''SELECT name FROM sqlite_master WHERE type="table"''')
    print(c.fetchall())
    conn.close()

# print_tables()