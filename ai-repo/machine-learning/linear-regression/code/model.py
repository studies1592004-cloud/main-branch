import numpy as np


class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=10000, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = None

    def fit(self, X, y, method="bgd"):
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            if method == "bgd":
                self._bgd(X_shuffled, y_shuffled)
            elif method == "sgd":
                self._sgd(X_shuffled, y_shuffled)
            elif method == "mbgd":
                self._mbgd(X_shuffled, y_shuffled)
            else:
                raise ValueError("Invalid gradient type")

    def _bgd(self, X, y):
        error = (X @ self.weights) - y
        grad = (2 / len(X)) * (X.T @ error)
        self.weights -= self.lr * grad

    def _sgd(self, X, y):
        for i in range(len(X)):
            xi = X[i:i+1]
            yi = y[i:i+1]
            error = (xi @ self.weights) - yi
            grad = 2 * xi.T @ error
            self.weights -= self.lr * grad

    def _mbgd(self, X, y):
        for i in range(0, len(X), self.batch_size):
            xb = X[i:i+self.batch_size]
            yb = y[i:i+self.batch_size]
            error = (xb @ self.weights) - yb
            grad = (2 / len(xb)) * (xb.T @ error)
            self.weights -= self.lr * grad

    def predict(self, X):
        return X @ self.weights
