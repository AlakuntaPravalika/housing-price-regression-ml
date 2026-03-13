import numpy as np
import pandas as pd
from src.multiple_linear_regression import MultipleLinearRegression
from src.metrics import rmse

def forward_selection(X, y, feature_names):
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    best_score = float("inf")
    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_subset = X[:, features_to_test]
            model = MultipleLinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            score = rmse(y, y_pred)
            scores.append((score, feature))
        scores.sort()
        best_new_score, best_feature = scores[0]
        if best_new_score < best_score:
            best_score = best_new_score
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    selected_feature_names = [feature_names[i] for i in selected_features]
    print("\nSelected Features (Forward Selection):")
    print(selected_feature_names)
    return selected_features

def backward_elimination(X, y, feature_names):
    features = list(range(X.shape[1]))
    best_score = float("inf")
    while len(features) > 1:
        scores = []
        for feature in features:
            temp_features = [f for f in features if f != feature]
            X_subset = X[:, temp_features]
            model = MultipleLinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            score = rmse(y, y_pred)
            scores.append((score, feature))
        scores.sort()
        best_new_score, worst_feature = scores[0]
        if best_new_score < best_score:
            best_score = best_new_score
            features.remove(worst_feature)
        else:
            break

    selected_feature_names = [feature_names[i] for i in features]
    print("\nSelected Features (Backward Elimination):")
    print(selected_feature_names)
    return features

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [1 / (1 - df.corr()[col].drop(col).pow(2).max())
        for col in df.columns
    ]

    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)

    return vif_data