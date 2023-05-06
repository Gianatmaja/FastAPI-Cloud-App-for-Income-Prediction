# Put the code for your API here.
import os
from fastapi import FastAPI
from typing import Literal
import pandas as pd
import numpy as np
import uvicorn
from pydantic import BaseModel
from src.model_runner import process_inference_data
from src.utils import load_model

'''
# Set up DVC on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
'''


# Create app
app = FastAPI()

# POST Input Schema
class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal[" Female", " Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": 32,
                "workclass": 'Self-emp-not-inc',
                "fnlgt": 83311,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Sales",
                "relationship": "Husband",
                "race": "White",
                "sex": " Male",
                "capital_gain": 2500,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": 'United-States'
            }
        }


# Load model
model = load_model('model/xgb_model.pkl')


# Root path
@app.get("/")
async def root():
    return {
        "Hi": "This app predicts whether the input person's annual income exceeds $50 000."}

# Prediction path
@app.post("/predict-income")
async def predict(input: ModelInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    original_col_names = [
        'age', 'workclass', 'fnlgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race','sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]

    input_df = pd.DataFrame(data=input_data, columns=original_col_names)

    X = process_inference_data(input_df)
    y = model.predict(X)

    if y == 0:
        pred = '<=50K'
    elif y == 1:
        pred = '>50K'
    else:
        pred = 'Check model output'

    return {"Income prediction": pred}

'''
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''