'''
This script performs unit testing on the ML pipeline components.

Author: Gian Atmaja
Created: 5 May 2023
'''

# Import required libraries
import logging
import pandas as pd
import pytest

from pathlib import Path
from xgboost import XGBClassifier
from pickle import load
from src import utils

# Define data and model paths
data_path = 'data/census_intermediate.csv'
model_path = 'model/xgb_model.pkl'


# Data to be used in testing
@pytest.fixture(name='data')
def data():
    
    data = utils.read_data(data_path)

    yield data


# Test read_data function from utils.py
def test_read_data(data):

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


# Test get_model function from model_runner.py
def test_preprocess_target(data):

    df = utils.preprocess_target(data)

    assert set(df['salary']) == {0, 1}


# Test process_data function from model_runner.py
def test_load_model():

    model = utils.load_model(model_path)

    assert isinstance(model, XGBClassifier)