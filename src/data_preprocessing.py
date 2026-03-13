import pandas as pd
import numpy as np

def mean_imputation(df):
    return df.fillna(df.mean())

def median_imputation(df):
    return df.fillna(df.median())

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

def normalize(X):
    return (X - X.mean()) / X.std()
    