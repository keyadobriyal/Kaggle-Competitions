This challenge I decided to progress from simple machine learning models to complex and analyse how it influences ROC-AUC.  Jupyter notebooks demonstrate an iterative approach to predicting customer churn, evolving from a basic model to sophisticated ensembles. Below is a summary of the machine learning algorithms and key techniques used in each version.

This report summarizes the evolution of the machine learning models used in the Customer Churn Prediction project across 15 versions. The project demonstrates a clear progression from basic single-model architectures to high-complexity stacking ensembles, with a direct correlation between model complexity and the improvement in ROC-AUC (Area Under the Receiver Operating Characteristic Curve).

| Submission and Description	| Public Score |
| --- | --- |
| customer-churn-RF - Version 27	| 0.91436 |
| customer-churn-RF - Version 25	| 0.91401 |
| customer-churn-RF - Version 21	| 0.91419 |
| customer-churn-RF - Version 19	| 0.91417 |
| customer-churn-RF - Version 16	| 0.91396 |
| customer-churn-RF - Version 14	| 0.91401 |
| customer-churn-RF - Version 13	| 0.91393 |
| customer-churn-RF - Version 11	| 0.91390 |
| customer-churn-RF - Version 10	| 0.91385 |
| customer-churn-RF - Version 9	| 0.91086 |
| customer-churn-RF - Version 7	| 0.91003 |
| customer-churn-RF - Version 6	| 0.91005 |
| customer-churn-RF - Version 4 |	0.89906 |
| customer-churn-RF - Version 2 |	0.89858 |
| customer-churn-RF - Version 1	| 0.89269 |

<img width="591" height="641" alt="image" src="https://github.com/user-attachments/assets/eec1496a-1d9b-48e3-9ce1-e55475411f36" />


## Phase 1: Foundation & Calibration (v1 – v2)Algorithms: 
Random Forest Classifier.Key Complexity: Introduced Probability Calibration (Platt Scaling).AUC Impact: Initial models suffered from "center-pushed" probabilities. By applying calibration, the model began providing more reliable financial metrics, slightly improving the AUC by refining the decision boundary between churners and non-churners.

## Phase 2: Optimization & Class Balancing (v4 – v7)Algorithms: 
Random Forest, Balanced Random Forest.Key Complexity: Transitioned to GridSearchCV for hyperparameter tuning and Internal Resampling to handle the imbalanced nature of churn data.AUC Impact: These versions maximized the potential of a single-algorithm approach. By balancing the classes, the model became much more sensitive to the minority "Churn" class, leading to a noticeable jump in AUC.

## Phase 3: Multi-Model Blending (v9 – v14)Algorithms: 
XGBoost, LightGBM, CatBoost, and Random Forest.Key Complexity: Introduced Weighted Averaging (Blending). Instead of relying on one "perspective," these versions combined the strengths of Gradient Boosting (XGB/LGBM) with the stability of Bagging (Random Forest).AUC Impact: Significant improvement. Blending different "learning styles" smoothed out individual model errors, leading to a more robust and generalized prediction.

## Phase 4: Multi-Seed & Rank Refinement (v16 – v21)Algorithms: 
Four-Model Ensemble (XGB, LGBM, CatBoost, RF).Key Complexity: Implemented Multi-Seed Averaging and Rank Blending.AUC Impact: By running models across multiple random seeds, the "luck" factor was removed from training. Rank blending ensured that models with different output scales (e.g., CatBoost vs. Random Forest) were combined fairly based on relative probability, further squeezing out marginal AUC gains.

## Phase 5: Advanced Stacking & Meta-Modeling (v25 – v27)Algorithms: 
Stacking Ensemble with an optimized Meta-Learner.Key Complexity: Moved from simple averaging to Two-Level Stacking. Base model predictions (Level 1) were used as features for a Meta-Model (Level 2), which was optimized using a weight-discovery algorithm.AUC Impact: This phase reached the project peak (final OOF AUC ~0.917). The complexity of the meta-learner allowed the system to learn which model to trust for specific types of customers, providing the most accurate and nuanced churn probabilities.
Summary of Improvement
| Stage | Model Complexity | Primary Technique | AUC Trend |
| --- | --- | --- | --- |
| Baseline | Low | Single Random Forest | Entry level |
| Optimization | Medium | GridSearchCV & Balancing | Moderate Increase |
| Ensemble | High | Blending (XGB + RF + CatBoost) | Significant Jump |
| Stacking | Very High | Meta-Modeling & Rank Blending | Peak Performance |

Conclusion: The increase in complexity was not merely "adding more models," but rather adding diversity. Moving from a single tree-based model to a stacked ensemble of different boosting architectures allowed the system to capture complex, non-linear churn patterns that simpler models missed, resulting in a superior ROC-AUC.


# Evolution of Models and Algorithms
## 1. Basic Random Forest (Version 1.0)
Algorithm: Random Forest Classifier.

Summary: This initial version establishes a baseline using a standard Random Forest to predict churn probability. It includes fundamental data preprocessing like LabelEncoder for categorical variables and uses standard metrics like accuracy and a confusion matrix for evaluation.

## 2. Calibrated Random Forest (Version 2.0)
Algorithm: Random Forest Classifier with Platt Scaling (Sigmoid Calibration).

Summary: Recognizing that Random Forests often produce probabilities pushed toward the center (away from 0 and 1), this version introduces calibration. By applying CalibratedClassifierCV, it transforms raw outputs into more reliable probability estimates, which is critical for financial churn prevention strategies.

## 3. Optimized Random Forest (Version 3.0 & 4.0)
Algorithm: Random Forest Classifier with GridSearchCV.

Summary: This version moves away from manual hyperparameter guessing. It utilizes GridSearchCV to systematically test different numbers of trees (ranging from 50 to 300) to find the most accurate estimator for the dataset.

## 4. Balanced Random Forest (Version 6.0 & 7.0)
Algorithm: Balanced Random Forest Classifier (from imblearn).

Summary: To specifically address class imbalance (where non-churners far outnumber churners), these versions adopt the BalancedRandomForestClassifier. This model is designed to improve recall for churners and increase sensitivity to churn probability, while continuing to use stratified cross-validation and probability calibration.

## 5. Multi-Model Ensembles (Versions 9.0 – 14.0)
Later versions transition into ensemble methods, combining multiple high-performance gradient boosting and forest-based algorithms:

Algorithms Used:
- XGBoost: A high-performance gradient boosting framework.
- LightGBM: A fast, distributed gradient boosting framework focused on efficiency.
- CatBoost: A gradient boosting library that handles categorical features automatically.
- Random Forest: Retained as a base learner in the ensemble.

Summary of Techniques:
- Weighted Averaging: Versions like 9.0 and 10.0 blend predictions from XGBoost, Random Forest, and CatBoost using optimized weights to minimize loss.
- Strategic Ensembling: Version 14.0 represents the most advanced iteration, combining XGBoost, LightGBM, CatBoost, and Random Forest. This "ensemble of ensembles" leverages the unique strengths of each algorithm to provide the final churn probability prediction.

## 6. Advanced Ensemble Methods (Versions 16.0 – 21.0)
These versions focus on refining a multi-seed, four-model ensemble to maximize predictive "perspectives".
- XGBoost & LightGBM (Gradient Boosted Decision Trees):

Summary: These models build trees sequentially, where each new tree attempts to correct the errors of its predecessor.

Purpose: They are highly aggressive and excel at identifying complex, non-linear patterns in the data.

- CatBoost:

Summary: Utilizes a specialized technique called "Ordered Boosting".

Purpose: Specifically designed to reduce bias when handling categorical variables, often finding insights that XGBoost or LightGBM might overlook.

- Random Forest (Bagging Model):

Summary: Builds many independent trees in parallel and averages their results.

Purpose: Acts as a "perfect anchor" for the ensemble. Because it is harder to overfit, it prevents the more volatile boosting models from destabilizing the final prediction.

Refinement Techniques:

Multi-Seed Averaging: Running models across different seeds to ensure results are not due to random chance.

Rank Blending: Combining model outputs based on their relative ranks rather than raw probabilities to normalize different output scales.

## 7.0 Stacking and Meta-Modeling (Versions 25.0 – 27.0)
The final stages of the analysis move from simple blending to Stacking, where model outputs become inputs for a final decision-maker.

Algorithm: Stacking Classifier (Meta-Modeling):

Summary: This approach uses the predictions from the base models (XGBoost, LightGBM, CatBoost, and Random Forest) as "features" for a second-level model.

Meta-Learner: A final model (often a simple Logistic Regression or a weighted optimizer) learns how to best combine the strengths of each base model based on their Out-Of-Fold (OOF) performance.

Objective: These versions are heavily optimized for ROC-AUC and Log-Loss, ensuring that the final "Churn Probability" is a mathematically sound financial metric for churn prevention.
