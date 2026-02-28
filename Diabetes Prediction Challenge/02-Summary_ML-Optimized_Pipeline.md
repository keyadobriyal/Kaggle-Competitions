#🩺 Diabetes Prediction Challenge – Optimized ML Pipeline with Automated Tuning & Stacking#

## 1. Problem Overview ##

The goal of this competition was to predict the probability of diabetes using structured clinical and demographic data. Since the target variable is binary, the task was formulated as a binary classification problem, with ROC-AUC as the evaluation metric.

The focus of this submission was not only accuracy, but also:

Automated hyperparameter optimization

Robust cross-validation

Ensemble stacking

Calibrated probability outputs

## 2.0 Data Exploration & Preprocessing ##
Exploratory Data Analysis (EDA)

Initial steps included:

Checking missing values

Inspecting feature distributions

Understanding class imbalance

Identifying categorical vs numerical variables

Key insights:

Moderate class imbalance

Mixed data types requiring structured preprocessing

Certain clinical features strongly associated with diabetes risk

## 3.0 Modular ML Pipeline Architecture ##

A fully automated and reusable pipeline was constructed using:

Pipeline

ColumnTransformer

Scikit-learn compatible models

Feature Handling
Feature Type	Processing
Numerical	Imputation + StandardScaler
Categorical	OneHotEncoder (handle_unknown="ignore")

This ensured:

No data leakage

Reproducibility

Fair comparison across models

## 4.0 Base Models ##

Multiple high-performance algorithms were implemented:

Logistic Regression

Random Forest

XGBoost

LightGBM

Tree-based gradient boosting models showed superior baseline performance.

## 5.0 Automated Hyperparameter Optimization (Optuna) ##

To maximize model performance, Optuna-based Bayesian optimization was implemented.

Optimization Highlights:

Stratified K-Fold cross-validation

ROC-AUC as objective

Automated search over:

Learning rate

Number of estimators

Max depth

Subsampling parameters

Regularization terms

This significantly improved model stability and performance compared to manual tuning.

## 6.0 Stacking Ensemble with Weight Optimization ##

Instead of selecting a single best model, an ensemble approach was adopted:

### Stacking Strategy ###

Level-0 models: Tuned base learners

Out-of-fold predictions generated

Weighted blending optimized using Optuna

This automated weight search improved generalization by leveraging complementary model strengths.

### Probability Calibration ###

Since the competition required probability outputs:

Platt scaling (sigmoid calibration) was applied

Calibration improved probability reliability without sacrificing AUC

This is critical for medical risk prediction applications where probability interpretation matters.

### Model Performance ###

The final optimized stacked ensemble achieved:

Strong cross-validated ROC-AUC

Improved stability across folds

Better leaderboard performance compared to single-model approaches

Gradient boosting models (LightGBM/XGBoost) contributed most significantly to final performance.
