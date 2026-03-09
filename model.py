import pandas as pd
import numpy as np

df_train = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Income Prediction Model\dataset\adult train.csv" , names=["age" , "workclass" , "fnlwgt" , "education" , "education-num" , "maratial-status" , "occupation" , "relationship" , "race" , "sex" , "capital-gain" , "capital-loss" , "hours-per-week" , "native-country" , "income"])
df_train = df_train.drop(["education"] , axis=1)
df_train = df_train.replace(" ?" , np.nan)
df_train["native-country"] = df_train["native-country"].fillna(df_train["native-country"].mode()[0])
df_train = df_train.dropna

df_test = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Income Prediction Model\dataset\adult test.csv" , names=["age" , "workclass" , "fnlwgt" , "education" , "education-num" , "maratial-status" , "occupation" , "relationship" , "race" , "sex" , "capital-gain" , "capital-loss" , "hours-per-week" , "native-country" , "income"])
df_test = df_test.drop(["education"] , axis=1)


xtrain = df_train.drop(["income"] , axis=1)
ytrain = df_train["income"]

xtest = df_test.drop(["income"] , axis=1)
ytest = df_test["income"]

print(df_train.isnull().sum())