import numpy as np

class SimpleLinearRegression:

    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):

        n = len(X)

        self.w = 0
        self.b = 0

        self.cost_history = []

        for i in range(self.epochs):

            y_pred = self.w * X + self.b

            error = y_pred - y

            cost = (1/(2*n)) * np.sum(error**2)
            self.cost_history.append(cost)

            dw = (1/n) * np.sum(error * X)
            db = (1/n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return self.w * X + self.b