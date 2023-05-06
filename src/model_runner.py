'''
This script contains functions that perform model training, inference, as well as checking the
performance on each categorical slices from the data.

Author: Gian Atmaja
Created: 4 May 2023
'''

# Import required libraries
import numpy as np
import pandas as pd
import warnings
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from pickle import dump
from pickle import load

from src.utils import read_data, return_le_features, preprocess_target, return_cat_features

# Setup options & warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Function to train model
def train_model(filepath_str):
    '''
    Input:
        - filepath_str (str): data filepath
    Output:
        - model (obj): model Python object
    '''

    df = read_data(filepath_str)
    df = preprocess_target(df)

    LE_features = return_le_features()
    df = encode_features(df, LE_features, 'train')

    X_train, X_test, y_train, y_test = split_data(df)
    X_train = scale_data(X_train, 'train')
    X_test = scale_data(X_test, 'test')

    model = get_model()

    logging.info('Fitting model...')
    model.fit(X_train, y_train)

    logging.info('Predicting on test set...')
    y_preds = model.predict(X_test)

    acc = accuracy_score(y_preds, y_test)
    prec = precision_score(y_preds, y_test, zero_division = 0)
    rec = recall_score(y_preds, y_test, zero_division = 0)
    f1 = f1_score(y_preds, y_test, zero_division = 0)

    logging.info('Model accuracy: %.3f' % acc)
    logging.info('Model precision: %.3f' % prec)
    logging.info('Model recall: %.3f' % rec)
    logging.info('Model f1 score: %.3f' % f1)

    filepath = 'model/xgb_model.pkl'

    logging.info('Saving model...')
    dump(model, open(filepath, 'wb'))
    logging.info('Model saved to model/')

    return model


# Function to obtain slice-specific performance
def assess_slice_performance(model, data):
    '''
    Input:
        - model (obj): model Python object
        - data (Pandas DF): Pandas DF of data
    Output:
        - slice_performance_df (Pandas DF): Pandas DF containing
        model performance on slices
    '''

    cols = ['Feature', 'Slice', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    slice_performance_df = pd.DataFrame(columns = cols)

    Cat_Features = return_cat_features()

    logging.info('Assessing performance on slices...')
    for feature in Cat_Features:
        for slc in data[feature].unique():
            logging.info('Feature: {}'.format(feature))
            logging.info('Slice: {}'.format(slc))

            df_slice = data[data[feature] == slc]
            X_slice, y_slice = process_data(df_slice, 'test')

            logging.info('Predicting on data slice...')
            y_pred_slice = model.predict(X_slice)

            logging.info('Recording performance metrics on data slice...')
            acc = accuracy_score(y_pred_slice, y_slice)
            prec = precision_score(y_pred_slice, y_slice, zero_division = 0)
            rec = recall_score(y_pred_slice, y_slice, zero_division = 0)
            f1 = f1_score(y_pred_slice, y_slice, zero_division = 0)
            
            row_data = {'Feature':feature, 'Slice':slc, 'Accuracy':acc,
                        'Precision':prec, 'Recall':rec, 'F1 Score':f1}

            slice_performance_df = slice_performance_df.append(row_data,
                                                            ignore_index = True)
            logging.info('Performance metrics on data slice saved.')

    with open('slice_output.txt', 'a') as f:
        df_string = slice_performance_df.to_string(header=True, index=False)
        f.write(df_string)

    logging.info('Performance on data slices stored in slice_output.txt')

    return slice_performance_df


# Function to use model to predict on data
def model_inference(data, model_path):
    '''
    Input:
        - data (Pandas DF): Pandas DF of data
        - model_path (str): model filepath
    Output:
        - acc (float): accuracy score
    '''
    logging.info('Loading model...')
    model = load(open(model_path, 'rb'))

    X, y = process_data(data, 'test')

    logging.info('Predicting on data...')
    y_preds = model.predict(X)
    data['Predicted'] = y_preds

    acc = accuracy_score(y_preds, y)
    logging.info('Model accuracy on prediction data: %.3f' % acc)

    data.to_csv('Predictions.csv')
    logging.info('Predictions saved to Predictions.csv')

    return acc


# Function to process data
def process_data(df, process_mode):
    '''
    Input:
        - df (Pandas DF): Pandas DF to be processed
        - process_mode: train or test
    Output:
        - X (Numpy array): X features
        - y (Numpy array): y values
    '''

    df = preprocess_target(df)
    LE_Features = return_le_features()

    df = encode_features(df, LE_Features, process_mode)

    logging.info('Splitting X and y features...')
        
    y = df['salary']
    X = df.drop('salary', axis = 1)
    
    X = scale_data(X, process_mode)
    
    return X,y


# Function to process data (if target is not present)
def process_inference_data(df):
    '''
    Input:
        - df (Pandas DF): Pandas DF to be processed
    Output:
        - X (Numpy array): X features
    '''

    LE_Features = return_le_features()

    df = encode_features(df, LE_Features, 'test')
    
    X = scale_data(df, 'test')
    
    return X


# Function to encode categorical features
def encode_features(df, le_features_list, process_mode):
    '''
    Input:
        - df (Pandas DF): Pandas DF to be encoded
        - le_features_list (list): list of features to be encoded
        - process_mode: train or test
    Output:
        - df (Pandas DF: encoded Pandas DF
    '''
    
    if process_mode == 'train':
        for feature in le_features_list:
            logging.info('Encoding feature:{}'.format(feature))
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            logging.info('{} feature encoded'.format(feature))
            
            filepath = 'model/{}_encoder.pkl'.format(feature)
            dump(le, open(filepath, 'wb'))
            logging.info('Saved encoder to model/')
    else:
        for feature in le_features_list:
            logging.info('Loading encoder for {}...'.format(feature))
            filepath = 'model/{}_encoder.pkl'.format(feature)
            le = load(open(filepath, 'rb'))
            logging.info('Encoding feature:{}'.format(feature))
            df[feature] = le.transform(df[feature])

            logging.info('{} feature encoded'.format(feature))

    df.drop('education', axis = 1, inplace = True)

    return df


# Function to split data
def split_data(df):
    '''
    Input:
        - df (Pandas DF): Pandas DF to be split
    Output:
        - X_train (Numpy array)
        - X_test (Numpy array)
        - y_train (Numpy array)
        - y_test (Numpy array)
    '''

    logging.info('Splitting X & y features...')

    y = df['salary']
    X = df.drop('salary', axis = 1)

    logging.info('Splitting train & test sets...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    return X_train, X_test, y_train, y_test


# Function to scale numerical features
def scale_data(data, process_mode):
    '''
    Input:
        - data (Pandas DF): Pandas DF to be scaled
        - process_mode: train or test
    Output:
        - data (Pandas DF): scaled data
    '''

    filepath = 'model/robust_scaler.pkl'

    if process_mode == 'train':
        logging.info('Fitting scaler...')
        scaler = RobustScaler().fit(data)
        data = scaler.transform(data)
        logging.info('Data scaled.')
        dump(scaler, open(filepath, 'wb'))
        logging.info('Scaler saved to model/')
    else:
        logging.info('Loading scaler...')
        scaler = load(open(filepath, 'rb'))
        data = scaler.transform(data)
        logging.info('Data scaled.')

    return data


# Function to obtain XGBoost model
def get_model():
    '''
    Output:
        - xgb (obj): XGBoost Python object
    '''

    logging.info('Getting XGBoost model...')
    xgb = XGBClassifier(max_depth = 6, n_estimators = 30)

    return xgb


if __name__ == "__main__":
    data_filepath = 'data/census_intermediate.csv'
    model_filepath = 'model/xgb_model.pkl'

    model = train_model(data_filepath)

    df_full = read_data(data_filepath)
    X_train, X_test, y_train, y_test = split_data(df_full)

    data = X_test.join(y_test)
    assess_slice_performance(model, data)

    model_inference(df_full, model_filepath)