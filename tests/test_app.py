'''
This script contains the unit tests for the FAST API app.

Author: Gian Atmaja
Created: 6 May 2023
'''

# Import required libraries
from fastapi.testclient import TestClient
from main import app

# Instantiate the testing client with the app.
client = TestClient(app)


# Test GET
def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Hi": "This app predicts whether the input person's annual income exceeds $50 000."}


# Test POST for '<=50K' class
def test_post_predict_lower():
    """ Test an example when income is less than 50K """

    r = client.post("/predict-income", json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": " Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}


# Test POST for '>50K' class
def test_post_predict_higher():
    """ Test an example when income is higher than 50K """
    r = client.post("/predict-income", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 338409,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": " Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "Cuba"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": ">50K"}