import numpy as np

def k_fold_split(X, y, k=5):

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    fold_size = len(X) // k

    folds = []

    for i in range(k):

        test_idx = indices[i*fold_size:(i+1)*fold_size]

        train_idx = np.concatenate(
            (indices[:i*fold_size], indices[(i+1)*fold_size:])
        )

        folds.append((train_idx, test_idx))

    return folds

from src.multiple_linear_regression import MultipleLinearRegression
from src.metrics import rmse


def run_k_fold_cv(X, y, k=5):
    folds = k_fold_split(X, y, k)
    scores = []

    for train_idx, test_idx in folds:
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model = MultipleLinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = rmse(y_test, y_pred)
        scores.append(score)
        
    print("\nK-Fold Cross Validation RMSE:")
    print(scores)
    print("Average RMSE:", np.mean(scores))