'''
This script performs data cleaning on the raw census data, and saves the cleaned
data into the data/ folders.

Author: Gian Atmaja
Created: 4 May 2023
'''

# Import required libraries
import numpy as np
import pandas as pd
import logging

# Function to clean data
def clean_data(filepath_str):
    '''
    Input:
        - filepath_str (str): string of filepath containing csv file
    Output:
        - df: cleaned data in Pandas DataFrame
    '''

    df = read_data(filepath_str)
    logging.info('Cleaning data...')

    logging.info('Renaming columns...')
    df.columns = [
        'age', 'workclass', 'fnlgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race','sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'salary'
    ]

    cols_to_strip = return_cols_to_strip()

    logging.info('Stripping whitespace...')
    for col in cols_to_strip:
        df[col] = df[col].str.strip()
    
    logging.info('Saving cleaned data to data/')
    df.to_csv('../data/census_intermediate.csv', index = False)
    logging.info('Clean data saved to data/')

    return df

# Function to read data
def read_data(filepath_str):
    '''
    Input:
        - filepath_str (str): string of filepath containing csv file
    Output:
        - df: data in Pandas DataFrame
    '''

    logging.info('Reading data...')
    df = pd.read_csv(filepath_str)

    return df

# Function to obtain column names where whitespaces are to be stripped from
def return_cols_to_strip():
    '''
    Output:
        - cols_to_strip_list (list): List of column names
    '''

    cols_to_strip_list = ['workclass', 'education', 'marital-status',
        'occupation', 'relationship', 'race', 'native-country','salary'
        ]

    return cols_to_strip_list

# Run
if __name__ == "__main__":
    data_filepath = '../data/census.csv'
    clean_data(data_filepath)