import numpy as np
import pandas as pd


class DataPipeline:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.features = None
        self.mean = None
        self.std = None

    def preprocess(self):

        self.df.drop(columns = ['id', 'Unnamed: 32'], inplace = True)

        self.df.diagnosis = [1 if value == 'M' else 0 for value in self.df.diagnosis]
        # self.df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

        target = "diagnosis"
        self.features = [c for c in self.df.columns if c != target]

        X = self.df[self.features].values
        y = self.df[target].values

        # Shuffle before split
        # indices = np.random.permutation(len(X))
        # X = X[indices]
        # y = y[indices]

        split = int(0.7 * len(X))
        return X[:split], X[split:], y[:split], y[split:]

    def standardize(self, X_train, X_test):
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)

        # Prevent division by zero
        # self.std[self.std == 0] = 1

        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std

        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        return X_train, X_test

    def prepare_single_input(self, x):
        x = (x - self.mean) / self.std
        x = np.hstack([np.ones((1, 1)), x])
        return x
