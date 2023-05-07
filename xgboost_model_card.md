# Model Card
For additional information, refer to the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf.

## Model Details
The model used is an XGBoost, with a max_depth hyperparameter of 6, and an n_estimators hyperparameter of 30. These
set of hyperparemeters were chosen as they were the best performing combination, from a range of other values, trained 
with the use of a stratified 5-fold validation on the training dataset.

More information on the model building process can be found in the [model building notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/notebooks/Model%20Building.ipynb).

![Model Building Notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/ml_nb.png)

For more information on the algorithm used, refer to the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/).

## Intended Use
The objective of the model is to predict whether a person is earning more or less than $50 000.

## Training Data
The training data used is the publicly available [Census Bureau dataset](https://archive.ics.uci.edu/ml/datasets/census+income), 
obtained from the [UCL ML Repository](https://archive.ics.uci.edu/ml/index.php). More information on the dataset can be found on 
the dataset page.

An EDA of the dataset can be found in the [EDA notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/notebooks/EDA.ipynb).

![EDA Notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/eda_nb.png)

## Evaluation Data
80% of the data was used for training purposes, whereas 20% was set aside to evaluate the trained models.

## Metrics
The overall (test) metrics of the XGBoost model are as follows:
- Accuracy: 0.872
- Precision: 0.657
- Recall: 0.751
- F1-Score: 0.701

Performance on specific data-slices can be found [here](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/slice_output.txt).

## Ethical Considerations
Given that the data contains attributes such as occupation, workclass, sex, race, etc., consideration on how the 
model performs accross different subgroups must be given.

Our assessment of the model's fairness across different population subgroups can be found in the
[model bias & fairness notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/notebooks/Model%20Bias%20%26%20Fairness.ipynb).

![Model Fairness Notebook](https://github.com/Gianatmaja/fastapi-cloud-app/blob/main/screenshots/fairness.png)

## Caveats and Recommendations
Caveats:
- Limited feature engineering and hyperparameter tuning was utilized in the model building process.
- The model shows some disparities, particulary related to false positive and negative rates, on different occupation,
workclass, education, and marital-status subgroups, raising concerns on its fairness.

Recommendations:
- A more thorough hyperparameter tuning, as well as feature engineering & EDA process might lead to a better-performing
model.
- Slice-based learning could be considered to achieve better model fairness.