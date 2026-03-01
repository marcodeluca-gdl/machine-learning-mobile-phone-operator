# Machine Learning for Churn Prediction -- Mobile Phone Operator

## Overview

This project develops a machine learning framework to predict customer
churn for a mobile phone operator.

The dataset contains 8000 customers and includes demographic, financial,
and behavioral variables. Given the class imbalance (approximately 20%
churn rate), the analysis prioritizes ROC-AUC as the primary metric,
with particular attention to Recall and F1-score for the minority class
(churners).

In addition to model selection, the project includes a fairness analysis
with respect to the sensitive attribute Gender, evaluating Independence,
Separation, and Sufficiency criteria.

------------------------------------------------------------------------

## Dataset Description

-   Total observations: 8000
-   Target variable: Churn (1 = client leaves, 0 = client stays)
-   Features:
    -   CreditScore
    -   Geography (one-hot encoded)
    -   Gender (binary encoded)
    -   Age
    -   Tenure
    -   Balance
    -   NumOfProducts
    -   HasCrCard
    -   IsActiveMember
    -   Salary

Preprocessing steps: - Gender encoded as binary variable. - Geography
one-hot encoded. - Stratified 80/20 train-test split. - Feature scaling
applied where required.

The dataset presents a clear imbalance (around 20% churners), making
accuracy alone insufficient as an evaluation metric.

------------------------------------------------------------------------

## Machine Learning Models

The following classification models were implemented and compared:

-   Gaussian Naive Bayes
-   LDA, QDA
-   Logistic Regression
-   KNN
-   Decision Tree
-   Bagging (Decision Trees)
-   Random Forest
-   Extra Trees
-   AdaBoost
-   Gradient Boosting
-   Stacking Classifier

To address class imbalance: - class_weight="balanced" was used for
tree-based models. - Sample weights were applied in boosting models. -
Evaluation emphasized Recall for churners and ROC-AUC.

------------------------------------------------------------------------

## Training Strategies

Three evaluation approaches were used:

### 1. Fixed Parameters Cross-Validation

-   5-fold cross-validation with default hyperparameters.
-   Baseline comparison across models.

### 2. Hyperparameter Tuning

-   RandomizedSearchCV with 5-fold cross-validation.
-   Optimization of ensemble models.
-   Exploration of different stacking configurations.

### 3. Anomaly Feature Engineering

-   Local Outlier Factor (LOF) used to detect anomalies.
-   Binary anomaly feature added to the dataset.
-   Tested whether churners behave as statistical outliers.

All models were evaluated using: - Accuracy - Precision - Recall -
F1-score - ROC-AUC

------------------------------------------------------------------------

## Results

### Recommended Model: Gradient Boosting (Fixed Parameters)

  Metric      Cross-Validation   Test Set
  ----------- ------------------ ----------
  Accuracy    0.801              0.806
  Precision   0.505              0.512
  Recall      0.732              0.748
  F1-score    0.597              0.608
  ROC-AUC     0.861              0.866

Reasons for selection:

1.  Strong generalization performance (stable CV and Test metrics).
2.  Highest Recall for churners (\~0.75), minimizing false negatives.
3.  Strong discriminative power (ROC-AUC approximately 0.87).
4.  Balanced trade-off between Recall and Precision.
5.  Lower complexity compared to stacking models.

Hyperparameter tuning and anomaly feature engineering did not produce
significant improvements over the fixed-parameter Gradient Boosting
model.

------------------------------------------------------------------------

## Fairness Analysis

The final Gradient Boosting model was evaluated with respect to Gender
as a sensitive attribute.

### Independence

Predicted churn rates differ across genders.\
Conclusion: Violated.

### Separation

True Positive Rates are similar across genders, but False Positive Rates
differ significantly.\
Conclusion: Partially violated.

### Sufficiency

Precision is nearly identical across genders.\
Conclusion: Satisfied.

The results align with the fairness trade-off theorem: when the
sensitive attribute is correlated with the target variable, it is
generally impossible to satisfy Independence, Separation, and
Sufficiency simultaneously.

------------------------------------------------------------------------

## Project Structure
'''
├── celldata.csv 
│ 
├── mlProject_mobile_phone_operator.ipynb 
│ 
├── requirements.txt  
│
└── README.md
'''
------------------------------------------------------------------------

## Technologies

-   Python 3.x
-   scikit-learn
-   pandas
-   numpy
-   matplotlib
-   seaborn

------------------------------------------------------------------------

## Authors

Marco De Luca\
Jacopo Spandri

