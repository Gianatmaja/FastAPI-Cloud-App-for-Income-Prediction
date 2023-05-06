# FASTAPI Cloud App
A classification model on publicly available Census Bureau data, deployed using FastAPI & Render, with CI/CD incorporated using GitHub Actions.

## Project Structure
The main files of this repository follow the following structure

    .
    ├── .dvc/                                   # DVC Configs
    ├── .github/workflows                       # CI/CD (with Github Actions) Codes
    ├── data/                                   # DVC Metadata for input data
    ├── model/                                  # Model-related files
    ├── notebooks/                              # Jupyter Notebooks 
    │  ├── EDA.ipynb
    │  ├── Model Building.ipynb
    │  ├── Model Bias & Fairness.ipynb
    ├── screenshots/                            # Screenshots for Documentation
    ├── src/                                    # ML Pipeline Codes                
    │  ├── __init__.py                                  
    │  ├── data_cleaning.py
    │  ├── model_runner.py
    │  ├── utils.py  
    ├── tests/                                             
    │  ├── __init__.py                          # Unit Tests for FastAPI App & ML Pipeline
    │  ├── test_app.py
    │  ├── test_ml_pipeline.py                                
    ├── main.py                                 # API Codes                     
    ├── sanitycheck.py                          # API Checking
    ├── setup.py
    ├── requirements.txt                      
    ├── slice_output.txt
    ├── xgboost_model_card.md
    └── README.md

## Running the Project
To install the requirements, run

    pip install -r requirements.txt

To run the web app locally, run the following command:

    uvicorn main:app --reload

Next, to run the unit tests, run the following command:

    pytest tests/ -vv

Finally, to run individual scripts, run the following command:

    python src/{SCRIPT_NAME}.py

## CI/CD Using GitHub Actions
CI (Continuous Integration) and CD (Continuous Deployment) is integrated to this repository through
the use of GitHub Actions. 

## ML Model Building
An XGBoost model was trained on the Census Bureau Dataset, obtained from the UCL ML Repository, to predict
whether a person's annual income is over or under $50K.

More details on the model can be found in the [model card](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/xgboost_model_card.md)

## API Creation & Deployment