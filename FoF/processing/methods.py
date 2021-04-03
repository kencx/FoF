#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

# ----- IMPORT DATA ------
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

    d = fits.open('FoF\\processing\\datasets\\' + fname)
    print(d.info())
    col_num = int(input('Choose the table to import: '))
    t = Table(d[col_num].data)
    df = t.to_pandas()
    d.close()
    print('Dataframe of table ' + str(col_num) + ' initialized.')
    print(df.head())
    return df


# ------- PROCESS DATA -------
def drop_columns(df, col_list):
    df = df.drop(col_list, axis=1)
    return df

def check_null(df):
    return df.isnull().sum()

def drop_null(df):
    if sum(check_null(df)):
        return df.dropna()



# ----- EXPORT DATA --------
def df_to_csv(df, fname):
    df.to_csv('FoF\\processing\\datasets\\' + fname + '_cleaned.csv')
    print(fname + ' CSV file added.')


def add_to_db(df, table_name):
    ''' Adds df to sqlite3 database. Table is created if it does not exists. If table exists, data is replaced.

    Parameters
    ----------
    df: pd.Dataframe
    
    table_name: str
        Table to be added to
    
    '''
    conn = sqlite3.connect('FoF\\processing\\datasets\\galaxy_clusters.db')
    c = conn.cursor()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print('Table added to SQL DB.')
    conn.commit()
    conn.close()


def add_data(df, name):
    df_to_csv(df, name)
    add_to_db(df, name)


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



# --- DB MANAGEMENT ---
path = 'FoF\\processing\\datasets\\galaxy_clusters.db'
def print_tables(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''SELECT name FROM sqlite_master WHERE type="table"''')
    print(c.fetchall())
    conn.close()

def drop_table(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    data3 = str(input('Please enter name: '))
    mydata = c.execute("DELETE FROM sqlite_master WHERE Name=?", (data3,))
    conn.commit()
    c.close

# print_tables(path)
# drop_table(path)
# print(1)