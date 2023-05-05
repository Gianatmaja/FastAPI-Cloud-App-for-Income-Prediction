'''
This script contains small functions used throughout the data_cleaning.py and
model_runner.py files.

Author: Gian Atmaja
Created: 4 May 2023
'''

# Import required libraries
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from pickle import load

# Setup logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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


# Function to return list of features to be encoded
def return_le_features():
    '''
    Output:
        - le_features_list (list): list of features to be encoded
    '''

    le_features_list = ['workclass', 'marital-status', 'occupation', 
               'relationship', 'race','sex', 'native-country']
    
    return le_features_list


# Function to preprocess target value
def preprocess_target(df):
    '''
    Input:
        - df (Pandas DF): Input DF
    Output:
        - df (Pandas DF): Pandas DF after target preprocessing
    '''

    df['salary'] = df['salary'].replace({'<=50K': 0, '>50K':1})

    return df


# Function to return categorical features list
def return_cat_features():
    '''
    Output:
        - cat_features_list (list): list of categorical features
    '''

    cat_features_list = ['workclass', 'education', 'marital-status', 'occupation', 
        'relationship', 'race','sex', 'native-country']

    return cat_features_list


# Function to load model
def load_model(model_path):
    '''
    Input:
        - model_path (str): string value of model filepath
    Output:
        - model (obj): model Python object
    '''
    model = load(open(model_path, 'rb'))

    return model