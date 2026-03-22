# Income Prediction Classifier

This project builds a Machine Learning model to predict whether a person's income is greater than 50K or less than 50K per year.

The model is trained on the Adult Census Income Dataset and implemented using Scikit-learn.

## Dataset

- Dataset used: Adult Census Income Dataset

The dataset contains features such as:

- Age
- Workclass
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

Target variable:

income → <=50K or >50K

## Task

- Handling Missing Values and feature cleaning
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- DecisionTreeClassifier for prediction
- GridSearchCV for hyperparameter tuning

## Result 

- Accuracy ≈ 0.80
- F1 Score ≈ 0.68

## Library Used
- numpy
- pandas
- scikit-learn
