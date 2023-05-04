import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from pickle import load


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


def return_le_features():

    le_features_list = ['workclass', 'marital-status', 'occupation', 
               'relationship', 'race','sex', 'native-country']
    
    return le_features_list


def preprocess_target(df):

    df['salary'] = df['salary'].replace({'<=50K': 0, '>50K':1})

    return df


def return_cat_features():

    cat_features_list = ['workclass', 'education', 'marital-status', 'occupation', 
        'relationship', 'race','sex', 'native-country']

    return cat_features_list
