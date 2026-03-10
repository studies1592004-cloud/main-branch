import numpy as np
import pandas as pd


class DataPipeline:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.features = None
        self.mini = None
        self.maxi = None

    def preprocess(self):
        self.df = pd.get_dummies(
            self.df,
            columns=["Building Type", "Day of Week"],
            dtype=int
        )

        target = "Energy Consumption"
        self.features = [c for c in self.df.columns if c != target]

        X = self.df[self.features].values
        y = self.df[target].values

        split = int(0.7 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def normalize(self, X_train, X_test):
        self.mini = X_train.min(axis=0)
        self.maxi = X_train.max(axis=0)

        denom = np.where(self.maxi - self.mini == 0, 1, self.maxi - self.mini)

        X_train = (X_train - self.mini) / denom
        X_test = (X_test - self.mini) / denom

        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        return X_train, X_test

    def prepare_single_input(self, x):
        denom = np.where(self.maxi - self.mini == 0, 1, self.maxi - self.mini)
        x = (x - self.mini) / denom
        x = np.hstack([np.ones((1, 1)), x])
        return x
