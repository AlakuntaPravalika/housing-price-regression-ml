import numpy as np

class PolynomialRegression:

    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):

        X_poly = X.copy()

        for d in range(2, self.degree+1):
            X_poly = np.c_[X_poly, X**d]

        return X_poly