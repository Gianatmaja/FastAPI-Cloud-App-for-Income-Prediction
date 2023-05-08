# FASTAPI Cloud App
A classification model on publicly available Census Bureau data, deployed using FastAPI & Render, with CI/CD incorporated using GitHub Actions.

## Project Structure
The main files of this repository follow the following structure.

    .
    ├── .dvc/                                   # DVC Configs
    ├── .github/workflows                       # CI/CD (with Github Actions) Codes
    ├── data/                                   # Metadata for Input Data (DVC Enabled)
    ├── model/                                  # Metadata for Model-Related Files (DVC Enabled)
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
    ├── render_api_request.py                   # live API POST method                
    ├── sanitycheck.py                          # API Checking
    ├── setup.py
    ├── requirements.txt                      
    ├── slice_output.txt
    ├── xgboost_model_card.md
    └── README.md

## Running the Project
To install the requirements, run the following command:

    pip install -r requirements.txt

To run the app locally, run the following command:

    uvicorn main:app --reload

To run the unit tests, run the following command:

    pytest tests/ -vv

To run individual scripts inside `src/`, run the following command:

    python src/{SCRIPT_NAME}.py

## CI/CD Using GitHub Actions
CI (Continuous Integration) and CD (Continuous Deployment) is integrated to this repository through
the use of GitHub Actions.

![CI/CD Pipeline After Each Commit](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/ci_cd.png)

After each commit to the `main` branch, GitHub Actions will initiate the build job, check for syntax errors using Flake8, and run a series of tests (refer to scripts in `tests/`) using Pytest. These processes will serve as the Continuous Integration (CI) step before deployment.

![CI Steps After Each Commit](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/continuous_integration.png)

Once the build process is completed (and all tests have been passed), the deployment job will start. This will ensure Continuous
Deployment (CD) to the Render cloud app.

![CD Steps After CI](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/continuous_deployment.png)

For more information on GitHub Actions, refer to the [GitHub Actions documentation](https://docs.github.com/en/actions).

## ML Model Building
An XGBoost model was trained on the Census Bureau Dataset, obtained from the UCL ML Repository, to predict
whether a person's annual income is over or under $50K. More details on the model (as well as the EDA, training, and model fairness assessment process) can be found in the [model card](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/xgboost_model_card.md).

The data and model-related (label encoder, scaler, and model pkl) files are version-controlled using DVC. The metadata for these files can be found inside `data/` and `model/`. 

For more information on DVC, refer to the [DVC documentation](https://dvc.org/doc).

## API Creation & Deployment
The API is built using FastAPI, using the codes found in `main.py`. A snippet of the automatically-generated documentation by FastAPI can be found below.

![FastAPI App Docs](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/example.png)

For API deployment, Render is used. A snippet of the live page receiving the contents of the GET method can be found below.

![FastAPI App Docs](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/live_get.png)

The POST method can be used to obtain a prediction from the trained ML model. To do this, input the desired feature values inside the `render_api_request.py` file. Then, run the following command:

    python render_api_request.py

An example of this process, and the response generated from the live API, can be found below.

![FastAPI App Docs](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/live_post.png)

For more information on FastAPI and Render, refer to the [FastAPI](https://fastapi.tiangolo.com/) and [Render](https://render.com/docs) documentations.