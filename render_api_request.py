'''
This script tests the live API by posting a set of input data, and checking whether the Render App
responses correctly by returning the predicted annual income category (over/under $50K).

Author: Gian Atmaja
Created: 6 May 2023
'''

# Import required libraries
import requests
import logging


# Set logging configurations
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# Input data
input_features = {
    "age": 34,
    "workclass": "Private",
    "fnlgt": 302146,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Divorced",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "Black",
    "sex": " Male",
    "capital_gain": 2135,
    "capital_loss": 0,
    "hours_per_week": 42,
    "native_country": "United-States"
}


# POST to API
app_url = "https://fastapi-cloudapp.onrender.com/predict-income"

r = requests.post(app_url, json=input_features)
#assert r.status_code == 200

# Log response
logging.info("Testing Render app")
logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")