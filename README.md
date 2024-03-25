# Description

## What is ML Olympiad

The Google ML Community Network (AI/ML GDEs, TFUGs, and 3rd-party ML communities) will be hosting Community Competitions and participating in active competitions on Kaggle. The campaign is supported by Google for Developers.

Launching and participating in Kaggle competitions will create hands-on learning opportunities for AI/ML communities and developers. This is the 3rd round of ML Olympiad operated by AI/ML Developer Programs team to help build and nurture strong AI/ML communities.

## Datasets

Anonymised Digital Bank Marketing Funnel dataset is used for this hackathon.

## Task

In this competition, I was given a Digital bank Marketing Funnel for micro loan application with some missing values. I required to build a model to predict Loan approval status.

## Evaluation

Evaluation is based on Probabilistic F-Score Beta (Micro). This leaderboard is calculated with approximately 40% of the test data. The final results will be based on the other 60%, so the final standings may be different. The final winners will be determined by combining both public (40%) and private (60%) results.

## Methodology

Details implementation and explaination is available on each notebook under /research folder.

For this project, the main challenges are as the followings:

- Highly imbalance dataset
- Highly missing values on 3 major columns
- High outliers on 3 numerical columns

#### Handle Missing Values:

- Using simple imputer for numerical and categorical features.
- Impute with mode for categorical features.
- KNN imputation for numerical features.

#### Handle Outliers:

- Remove outliers using z-score with threshold = 3 for numerical features.
- Also tested with IQR method for moving outliers.
- IQR method removed a lot more dataset compare to z-score.

#### Handle Imbalance Dataset:

- Using SMOTE for upsampling dataset
- USING RUS or NearMiss for undersampling dataset

### Feature Engineering

- Create Age column which is the difference between Leading Creation Date - Birth Date
- Create debt to income ratio
- create_monthly_loan_repayment_column
- create_payment_to_income_ratio_column
- create_loan_to_income_ratio_column

### Model Training

For this project, I tried to train the model with various model architecture including:

- Random Forest Classifier
- Gradient Boost Classifier
- XGboost Classifier
- Catboost Classifier
- Ensemble Model
- Tensorflow GradientBoosted Classifier
- Deep Neural Network

### Evaluation Metrics Selection

Before building the model, I need to decide the performance metric I would like to optimize towards.

The most critical performance metric for the rare events modeling is usually the minority class recall or precision values. For example, in the context of fraud detection, I would like to maximize the True Positive Rate (Sensitivity) and capture as many fraud cases as possible. I would like the model NOT to predict Fraud as not-Fraud. False Negative will have a higher cost than False Positive. Hence, I would monitor and optimize RECALL.

While in context of spam email classification, I would like to minimize the False Positive Rate and not misclassify any important email as spam, so the precision for the minority class is the metric I would like to optimize. False Positive will have a higher cost than False Negative. Hence, I would monitor and optimize PRECISION.

In this project (loan approval classification), I would like the model NOT to predict approve for unqualified application. False Negative will have a higher cost than False Positive. Hence, I would like a model to have LOW RECALL and optimize PRECISION.

However I would still need to find the best precision-recall values. Therefore I would also look into F2-score since the competition will evaluate from F-beta (=2)

### Summary

1. TFDF Gradientboosted model gives the best competition score. It can handle training with missing values which gives the advantage to train the features has high % of missing values such as "Interest Rate" in this case. With other model, I have to impute the data first. Imputation high missing values might not be representing the real world dataset.
2. TFDF also can train on highly imbalance dataset right away.
3. TFDF is so easy to use and implement.
4. Downside of TFDF is that it is limited to customization of hyperparameter tuning.
