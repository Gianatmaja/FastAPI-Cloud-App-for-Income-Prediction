'''
This script performs 

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

from utils import read_data, return_le_features, preprocess_target, return_cat_features

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.getLogger().setLevel(logging.INFO)

def train_model(filepath_str):

    df = read_data(filepath_str)
    df = preprocess_target(df)

    LE_features = return_le_features()
    df = encode_features(df, LE_features, 'train')

    X_train, X_test, y_train, y_test = split_data(df)
    X_train = scale_data(X_train, 'train')
    X_test = scale_data(X_test, 'test')

    model = get_model()

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    acc = accuracy_score(y_preds, y_test)

    logging.info('Model accuracy: %.3f' % acc)

    filepath = 'model/xgb_model.pkl'
    dump(model, open(filepath, 'wb'))

    return model


def assess_slice_performance(model, data):

    cols = ['Feature', 'Slice', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    slice_performance_df = pd.DataFrame(columns = cols)

    Cat_Features = return_cat_features()

    for feature in Cat_Features:
        for slc in data[feature].unique():
            df_slice = data[data[feature] == slc]
            X_slice, y_slice = process_data(df_slice, 'test')
            y_pred_slice = model.predict(X_slice)

            acc = accuracy_score(y_pred_slice, y_slice)
            prec = precision_score(y_pred_slice, y_slice, zero_division = 0)
            rec = recall_score(y_pred_slice, y_slice, zero_division = 0)
            f1 = f1_score(y_pred_slice, y_slice, zero_division = 0)
            
            row_data = {'Feature':feature, 'Slice':slc, 'Accuracy':acc,
                        'Precision':prec, 'Recall':rec, 'F1 Score':f1}

            slice_performance_df = slice_performance_df.append(row_data,
                                                            ignore_index = True)

    slice_performance_df.to_csv('Performance_on_slices.csv', index = False)

    return slice_performance_df


def process_data(df, process_mode):
    df = preprocess_target(df)
    LE_Features = return_le_features()

    df = encode_features(df, LE_Features, process_mode)
        
    y = df['salary']
    X = df.drop('salary', axis = 1)
    
    X = scale_data(X, process_mode)
    
    return X,y

def encode_features(df, le_features_list, process_mode):
    
    if process_mode == 'train':
        for feature in le_features_list:
            logging.info('Encoding feature:{}'.format(feature))
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            
            filepath = 'model/{}_encoder.pkl'.format(feature)
            dump(le, open(filepath, 'wb'))
            logging.info('Saved encoder to model/')
    else:
        for feature in le_features_list:
            filepath = 'model/{}_encoder.pkl'.format(feature)
            le = load(open(filepath, 'rb'))
            df[feature] = le.transform(df[feature])

    df.drop('education', axis = 1, inplace = True)

    return df


def split_data(df):

    y = df['salary']
    X = df.drop('salary', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    return X_train, X_test, y_train, y_test


def scale_data(data, process_mode):

    filepath = 'model/robust_scaler.pkl'

    if process_mode == 'train':
        scaler = RobustScaler().fit(data)
        data = scaler.transform(data)
        dump(scaler, open(filepath, 'wb'))
    else:
        scaler = load(open(filepath, 'rb'))
        data = scaler.transform(data)

    return data


def get_model():

    xgb = XGBClassifier(max_depth = 6, n_estimators = 30)

    return xgb


if __name__ == "__main__":
    data_filepath = 'data/census_intermediate.csv'
    model = train_model(data_filepath)

    df_full = read_data(data_filepath)
    X_train, X_test, y_train, y_test = split_data(df_full)

    data = X_test.join(y_test)
    assess_slice_performance(model, data)