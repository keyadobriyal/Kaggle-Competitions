# Heart Disease Prediction

Experiment with various algorithms and methodology of machine learrning, I was able to achieve 151 rank (Private Notebook) out of 4370 team.

***CatBoost KFold V2*** was the best notebook with ***Public Score*** of ***0.95530*** and ***Private Score*** of ***0.95387***

## 1. predictingheart-disease-elasticnet_v4.ipynb

### Summary

This notebook builds a heart disease prediction model using ElasticNet-based regression/classification techniques. The workflow focuses on preparing clinical data and training a regularized linear model that balances L1 (Lasso) and L2 (Ridge) penalties to control overfitting and improve generalization.

The data preprocessing stage includes handling missing values, separating features and target variables, and preparing the dataset for model training. The model is evaluated using K-Fold and Stratified K-Fold cross-validation, ensuring that the distribution of the target class is preserved across folds.

The main goal of this notebook is to demonstrate how regularized linear models can perform robust classification for medical datasets, while maintaining interpretability.

**Key Techniques:**
- ElasticNet regularization
- Stratified K-Fold cross-validation
- Clinical feature-based prediction
- Model evaluation using cross-validation

## 2. catboost-kfold_v3.ipynb

### Summary

This notebook develops a heart disease prediction model using CatBoost, a gradient boosting algorithm that handles categorical features efficiently.

The dataset is processed and then trained using CatBoost with K-Fold / Stratified K-Fold cross-validation, allowing robust estimation of model performance across multiple folds. CatBoost automatically handles categorical feature encoding and reduces the need for extensive preprocessing.

The notebook emphasizes boosting-based modeling and cross-validation stability, producing predictions based on the averaged results across folds.

***Key Techniques***
- CatBoost gradient boosting model
- Stratified K-Fold cross-validation
- Automated categorical feature handling
- Cross-validation based prediction averaging

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

## 4. 5-model-stacking-pipeline_v3.ipynb

### Summary

This notebook implements an advanced ensemble learning pipeline for heart disease prediction using stacking. Multiple machine learning models are trained as base learners, including:
- Ridge Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The workflow includes data preprocessing, feature scaling using StandardScaler, and model training using Stratified K-Fold cross-validation.

Hyperparameters are optimized using Optuna, and out-of-fold predictions from the base models are used to train a meta-model in a stacking architecture. This approach improves predictive accuracy by combining the strengths of multiple algorithms.

The final predictions are generated using the optimized stacked ensemble.

**Key Techniques**
- Multi-model stacking ensemble
- Optuna hyperparameter optimization
- Stratified K-Fold cross-validation
- StandardScaler preprocessing
- Gradient boosting + linear + tree models

