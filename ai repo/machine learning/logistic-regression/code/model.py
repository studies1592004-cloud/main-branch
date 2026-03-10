import numpy as np


class LogisticRegressionGD:
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


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def _bgd(self, X, y):
        y_hat = self.sigmoid(X @ self.weights)
        error = y_hat - y
        grad = (1 / len(X)) * (X.T @ error)
        self.weights -= self.lr * grad


    def _sgd(self, X, y):
        for i in range(len(X)):
            xi = X[i:i+1]
            yi = y[i:i+1]

            y_hat = self.sigmoid(xi @ self.weights)
            error = y_hat - yi
            grad = xi.T @ error
            self.weights -= self.lr * grad


    def _mbgd(self, X, y):
        for i in range(0, len(X), self.batch_size):
            xb = X[i:i+self.batch_size]
            yb = y[i:i+self.batch_size]

            y_hat = self.sigmoid(xb @ self.weights)
            error = y_hat - yb
            grad = (1 / len(xb)) * (xb.T @ error)
            self.weights -= self.lr * grad

    # return probabilities
    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights)
    
    # return class labels
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
