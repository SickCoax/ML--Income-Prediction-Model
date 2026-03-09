import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , f1_score

df_train = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Income Prediction Model\dataset\adult train.csv" , names=["age" , "workclass" , "fnlwgt" , "education" , "education-num" , "marital-status" , "occupation" , "relationship" , "race" , "sex" , "capital-gain" , "capital-loss" , "hours-per-week" , "native-country" , "income"])
df_train = df_train.replace(" ?" , np.nan)
df_train = df_train.drop(["education"] , axis=1)
df_train["native-country"] = df_train["native-country"].fillna(df_train["native-country"].mode()[0])
df_train = df_train.dropna(thresh=13)
df_train["occupation"] = df_train["occupation"].fillna(df_train["occupation"].mode()[0])

df_test = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Income Prediction Model\dataset\adult test.csv" , names=["age" , "workclass" , "fnlwgt" , "education" , "education-num" , "marital-status" , "occupation" , "relationship" , "race" , "sex" , "capital-gain" , "capital-loss" , "hours-per-week" , "native-country" , "income"])
df_test = df_test.replace(" ?" , np.nan)
df_test = df_test.drop(["education"] , axis=1)
df_test = df_test.replace(" <=50K." , " <=50K")
df_test = df_test.replace(" >50K." , " >50K")
df_test["native-country"] = df_test["native-country"].fillna(df_test["native-country"].mode()[0])
df_test = df_test.dropna(thresh=13)
df_test["occupation"] = df_test["occupation"].fillna(df_test["occupation"].mode()[0])


xtrain = df_train.drop(["income"] , axis=1)
ytrain = df_train["income"]

xtest = df_test.drop(["income"] , axis=1)
ytest = df_test["income"]

le = LabelEncoder()

ytrain = le.fit_transform(ytrain)
ytest = le.transform(ytest)

num_cols = xtrain.select_dtypes(include="number").columns
cat_cols = xtrain.select_dtypes(include=["object", "string"]).columns

preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprocess" , preprocess) , ("DT" , DecisionTreeClassifier(max_depth=8 , class_weight="balanced"))])

param_grid = {"DT__max_depth" : [4,5,6,7,8,9,10]}

grid = GridSearchCV(pipeline , param_grid , cv=5 , scoring="f1_weighted")
grid.fit(xtrain , ytrain)
model = grid.best_estimator_
ypred = model.predict(xtest)


# --------------------
# Evaluation Matric
# --------------------
print(f"Accuracy : {accuracy_score(ytest , ypred)}")
print(f"F1 Score : {f1_score(ytest , ypred)}")