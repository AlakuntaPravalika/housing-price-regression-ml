import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_preprocessing import mean_imputation,median_imputation, remove_outliers_iqr, normalize
from src.simple_linear_regression import SimpleLinearRegression
from src.multiple_linear_regression import MultipleLinearRegression
from src.metrics import r2_score, rmse, mae, mape
from src.feature_seletion import forward_selection, backward_elimination, calculate_vif
from src.cross_validation import run_k_fold_cv
def load_data():

    df = pd.read_csv("data/housing_data.csv")

    print("Dataset Loaded")
    print(df.head())

    return df


def preprocess_data(df):

    # Handle missing values
    df = mean_imputation(df)

    # Remove price outliers
    df = remove_outliers_iqr(df, "price")

    return df


def simple_regression(df):

    print("\nRunning Simple Linear Regression (area → price)")

    X = df["area"].values
    y = df["price"].values

    model = SimpleLinearRegression(lr=0.00000001, epochs=1000)

    model.fit(X, y)

    y_pred = model.predict(X)

    print("R2:", r2_score(y, y_pred))
    print("RMSE:", rmse(y, y_pred))
    print("MAE:", mae(y, y_pred))
    print("MAPE:", mape(y, y_pred))

    # Plot regression line
    # Plot regression line with confidence interval
    sorted_idx = np.argsort(X)
    X_sorted = X[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    std_error = np.std(y - y_pred)
    upper = y_pred + 2 * std_error
    lower = y_pred - 2 * std_error
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X, y_pred, color="red")
    plt.fill_between(X, lower, upper, color="gray", alpha=0.2)
    
    plt.title("Regression with Confidence Interval")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.show()
    

    #residual plot
    residuals = y - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
    
    # Plot cost convergence
    plt.plot(model.cost_history)
    plt.title("Cost Function Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

def multiple_regression(df):

    print("\nRunning Multiple Linear Regression")

    X = df.drop("price", axis=1).values
    y = df["price"].values

    calculate_vif(X)#multicollinearity check
    feature_names = X.columns.tolist()
    X = X.values

    selected_features = forward_selection(X, y, feature_names)
    X_selected = X[:, selected_features]

    model = MultipleLinearRegression()
    model.fit(X_selected, y)

    y_pred = model.predict(X_selected)

    print("R2:", r2_score(y, y_pred))
    print("RMSE:", rmse(y, y_pred))
    print("MAE:", mae(y, y_pred))
    print("MAPE:", mape(y, y_pred))
    print("\nRunning K-Fold Cross Validation")
    run_k_fold_cv(X_selected, y, k=5)

def compare_imputation(df):

    mean_df = mean_imputation(df.copy())
    median_df = median_imputation(df.copy())

    print("\nMissing values handled with MEAN")
    simple_regression(mean_df)

    print("\nMissing values handled with MEDIAN")
    simple_regression(median_df)

def compare_outliers(df):

    print("\nWITH OUTLIERS")
    simple_regression(df)

    from src.data_preprocessing import remove_outliers_iqr
    clean_df = remove_outliers_iqr(df, "price")
    print("\nWITHOUT OUTLIERS")
    simple_regression(clean_df)

def main():

    df = load_data()
    compare_imputation(df)

    df = preprocess_data(df)

    simple_regression(df)

    multiple_regression(df)


if __name__ == "__main__":
    main()