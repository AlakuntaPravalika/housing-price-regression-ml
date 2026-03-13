import numpy as np

class MultipleLinearRegression:

    def fit(self, X, y):

        X = np.c_[np.ones(X.shape[0]), X]

        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):

        X = np.c_[np.ones(X.shape[0]), X]

        return X @ self.theta