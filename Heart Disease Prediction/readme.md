# Heart Disease Prediction

Experiment with various algorithms and methodology of machine learrning, I was able to achieve 151 rank (Private Notebook) out of 4370 team.

## 03. predicting-heart-disease-ensemble_v3.ipynb

### Summary: 
This notebook develops a machine learning pipeline to predict the likelihood of heart disease using an ensemble of gradient boosting models.

The workflow begins with data loading and exploratory data analysis (EDA) to understand relationships between clinical variables and the target outcome. Important observations include correlations between chest pain type, thallium stress test results, and maximum heart rate with heart disease presence.

During preprocessing, the notebook performs data cleaning and missing value handling, filling numerical values with the median and categorical values with the mode. The target variable is encoded into binary classes, and categorical features are transformed using one-hot encoding. Feature scaling is also applied to prepare the data for modeling.

The modeling stage uses a stacking ensemble architecture with three powerful gradient boosting algorithms as base learners:
- XGBoost
- CatBoost
- LightGBM

Their predictions are combined using a meta-learner in a stacking ensemble, which improves predictive performance by leveraging strengths of each model.

Model performance is evaluated using cross-validation, after which the final ensemble is trained on the full dataset. The notebook then generates probability predictions for heart disease presence and saves them in a submission CSV file.

Key Techniques Used
- Exploratory Data Analysis (EDA)
- Missing value imputation
- One-hot encoding for categorical variables
- Feature scaling
- Gradient boosting models
- Stacking ensemble learning
- Cross-validation evaluation

Overall Goal:
Build a robust ensemble model for heart disease prediction by combining multiple boosting algorithms within a stacking framework to improve classification accuracy.
