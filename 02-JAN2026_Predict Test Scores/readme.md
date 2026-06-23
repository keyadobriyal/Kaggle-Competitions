# Test Score Prediction Challenge

Here are four methodology amongst many of learning proccess to alevate my machine learning skills;

## 1. exam-score-prediction-fe-and-ensemble.ipynb

### Summary

This notebook builds an exam score prediction model using feature engineering and ensemble learning techniques. It begins with exploratory data preparation and introduces engineered features such as study efficiency and total engagement to better capture student performance patterns.

Categorical variables are encoded using OrdinalEncoder, which is suitable for tree-based algorithms. A preprocessing pipeline is created using ColumnTransformer and StandardScaler for numerical features.

The model training stage uses gradient boosting algorithms, primarily:
- XGBoost
- LightGBM

Hyperparameters are optimized using Optuna, and model performance is evaluated using cross-validation. Predictions from multiple models are combined through ensemble techniques to improve overall accuracy and robustness.

**Key Techniques**
- Feature engineering
- Ordinal encoding for categorical features
- Pipeline-based preprocessing
- Optuna hyperparameter optimization
- Ensemble modeling

## 2. exam-score-prediction-bayesian-cv-rmse-notebook.ipynb

### Summary

This notebook focuses on model comparison and optimization using Bayesian-style hyperparameter tuning and cross-validation for exam score prediction.

The workflow includes:
* Data preprocessing with ColumnTransformer
* Encoding categorical variables using OneHotEncoder
* Standardizing numerical variables using StandardScaler
* Multiple regression models are trained and compared:
  *  Ridge Regression
  *  Random Forest
  *  Gradient Boosting
  *  XGBoost
  *  LightGBM
* Hyperparameters are tuned using Optuna, and model performance is evaluated using cross-validated RMSE (Root Mean Squared Error).

The notebook emphasizes systematic model evaluation and Bayesian-style optimization to identify the best-performing model.

**Key Techniques**
- Bayesian hyperparameter tuning
- Cross-validation RMSE evaluation
- Multi-model comparison
- Pipeline-based preprocessing

## 3. test-score-prediction_v3.ipynb

### Summary

This notebook implements a stacked regression approach for exam score prediction using feature engineering and model blending.

Categorical variables are encoded using Target Encoding, which captures relationships between categories and the target variable more effectively than standard encoding.

The modeling stage uses multiple algorithms:
- Ridge Regression
- ElasticNet
- XGBoost
- LightGBM

Out-of-fold predictions are generated using Stratified K-Fold cross-validation, and model outputs are combined using weight optimization via numerical minimization to reduce prediction error.

The final predictions are produced through a weighted ensemble of multiple models.

Key Techniques
- Target encoding
- Ridge / ElasticNet linear models
- Gradient boosting models
- Out-of-fold stacking
- Weight optimization using scipy.optimize

## 4. test-score-prediction_v6.ipynb

### Summary

This notebook is an enhanced version of the previous pipeline, focusing on improving ensemble performance and training stability.

It retains the same core structure:
- Feature engineering
- Target encoding
- Multiple regression models (Ridge, ElasticNet, XGBoost, LightGBM)

However, improvements are made in:
- Cross-validation strategy
- Model training consistency
- Ensemble weight optimization
- Prediction stability

The final model uses optimized blending of multiple algorithms to achieve better predictive performance.

**Key Techniques**
- Advanced ensemble blending
- Target encoding
- Cross-validation improvements
- Optimized model weighting

