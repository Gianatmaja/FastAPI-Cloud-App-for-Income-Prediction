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

from utils import read_data, return_cols_to_strip

# Setup logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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
    df.to_csv('data/census_intermediate.csv', index = False)
    logging.info('Clean data saved to data/')

    return df


if __name__ == "__main__":
    data_filepath = 'data/census.csv'
    clean_data(data_filepath)