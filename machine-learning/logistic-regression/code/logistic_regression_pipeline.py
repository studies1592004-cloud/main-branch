import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import LogisticRegressionGD

MISSING_VALUE_THRESHOLD = 0.2

class DatasetLoader:
    """
    Loads dataset from a CSV file
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return pd.read_csv(self.file_path)


class DatasetVerifier:
    """
    Performs basic dataset validation
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def verify(self):
        if self.dataframe.empty:
            raise ValueError("Dataset is empty")
        return True


class DatasetPreprocessor:
    """
    Handles missing values and categorical encoding
    """

    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column

    def drop_fully_na_columns(self, df):
        """
        Drops columns where ALL values are NA
        """

        # Identify columns where all values are NA
        fully_na_columns = df.columns[df.isna().all()]

        if len(fully_na_columns) > 0:

            print("\nDropping fully NA columns:")

            for col in fully_na_columns:
                print(col)

            df = df.drop(columns=fully_na_columns)

        return df

    def handle_missing_values(self):
        """
        Drops rows if missing values ≤ 20%
        Otherwise applies forward-fill and backward-fill
        """
        df = self.dataframe.copy()
        max_null_percentage = df.isnull().mean().max()

        if max_null_percentage <= MISSING_VALUE_THRESHOLD:
            df.dropna(inplace=True)
        else:
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

        return df

    def encode_categorical(self, df):
        """
        Applies one-hot encoding only if
        unique values in column ≤ 10
        """
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:
            if df[column].nunique() <= 10:
                df = pd.get_dummies(
                    df,
                    columns=[column],
                    drop_first=True
                )

        return df

    def encode_target(self, y):
        """
        Encodes target labels if they are categorical (e.g., B, M → 0, 1)
        """

        if y.dtype == "object" or str(y.dtype) == "category":

            print("\nEncoding target labels...")

            encoder = LabelEncoder()

            y = encoder.fit_transform(y)

            print("Label mapping:")
            for i, label in enumerate(encoder.classes_):
                print(f"{label} → {i}")

        return y


    def preprocess(self):
        """
        Executes full preprocessing pipeline
        """
        df = self.handle_missing_values()
        df = self.drop_fully_na_columns(df)
        y = df[self.target_column]
        y = self.encode_target(y)
        df = self.encode_categorical(df.drop(columns=[self.target_column]))
        x = df

        return x, y


class DatasetConverter:
    """
    Performs feature scaling
    """

    def __init__(self, method="standardize"):
        self.method = method
        self.scaler = None

    def fit_transform(self, x_train):
        if self.method == "normalize":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        return self.scaler.fit_transform(x_train)

    def transform(self, x):
        return self.scaler.transform(x)


class DatasetVisualizer:
    """
    Handles visualization for both regression and classification datasets
    including pairplot and correlation comparison
    """

    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column

    def pairplot(self):
        """
        Generates pairplot showing relationships between all columns
        including the target column
        """
        sns.pairplot(self.dataframe)
        plt.show()

    def correlation_heatmap(self):
        """
        Displays correlation matrix heatmap for all numeric columns
        including correlation with target column
        Works for both regression and classification targets
        """

        # Compute correlation matrix
        correlation_matrix = self.dataframe.corr(numeric_only=True)

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title(
            "Correlation Matrix (All Features vs Each Other including Target)"
        )

        plt.show()

    def target_correlation(self):
        """
        Shows sorted correlation of all features with target column
        Helps identify most important predictors
        """

        correlation_matrix = self.dataframe.corr(numeric_only=True)

        if self.target_column not in correlation_matrix.columns:
            print("Target column is not numeric, correlation not possible.")
            return

        target_corr = correlation_matrix[self.target_column].sort_values(
            ascending=False
        )

        print("\nCorrelation of each feature with target:\n")
        print(target_corr)


class Model:
    """
    Wrapper class for Logistic Regression
    """

    def __init__(self, penalty="l2", c_value=1.0):
        self.model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            solver="liblinear",
            max_iter=1000
        )
        # self.model = LogisticRegressionGD()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]


class DatasetTrainer:
    """
    Handles model training
    """

    def __init__(self, model):
        self.model = model

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)


class DatasetValidator:
    """
    Handles validation predictions
    """

    def __init__(self, model):
        self.model = model

    def validate(self, x_val):
        return self.model.predict(x_val), self.model.predict_proba(x_val)


class DatasetTester:
    """
    Handles test predictions
    """

    def __init__(self, model):
        self.model = model

    def test(self, x_test):
        return self.model.predict(x_test), self.model.predict_proba(x_test)


class Metrics:
    """
    Computes and prints classification metrics
    """

    @staticmethod
    def evaluate(y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_prob)
        }

    @staticmethod
    def pretty_print(title, metrics):
        print(title)
        print("-" * len(title))
        for key, value in metrics.items():
            print(f"{key:<10}: {value:.4f}")
        print()


class LogisticRegressionPipeline:
    """
    End-to-end Logistic Regression pipeline
    """

    def __init__(
        self,
        dataset_path,
        target_column,
        scaling_method="standardize",
        test_size=0.2,
        val_size=0.1
    ):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.scaling_method = scaling_method
        self.test_size = test_size
        self.val_size = val_size

    def run(self):
        """
        Executes the complete ML pipeline
        """

        loader = DatasetLoader(self.dataset_path)
        df = loader.load()

        verifier = DatasetVerifier(df)
        verifier.verify()

        # visualizer = DatasetVisualizer(df, self.target_column)
        # visualizer.pairplot()
        # visualizer.correlation_heatmap()
        # visualizer.target_correlation()

        preprocessor = DatasetPreprocessor(df, self.target_column)
        x, y = preprocessor.preprocess()

        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=self.test_size + self.val_size,
            random_state=42,
            stratify=y
        )

        relative_val_size = self.val_size / (
            self.test_size + self.val_size
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=1 - relative_val_size,
            random_state=42,
            stratify=y_temp
        )

        converter = DatasetConverter(self.scaling_method)
        x_train = converter.fit_transform(x_train)
        x_val = converter.transform(x_val)
        x_test = converter.transform(x_test)

        hyperparameters = [
            {"penalty": "l2", "C": 0.1},
            {"penalty": "l2", "C": 1.0},
            {"penalty": "l1", "C": 0.1},
            {"penalty": "l1", "C": 1.0}
        ]

        for params in hyperparameters:
            print("\n==============================================")
            print(f"Model Configuration: {params}")
            print("==============================================\n")

            model = Model(
                penalty=params["penalty"],
                c_value=params["C"]
            )

            trainer = DatasetTrainer(model)
            trainer.train(x_train, y_train)

            validator = DatasetValidator(model)
            y_val_pred, y_val_prob = validator.validate(x_val)

            tester = DatasetTester(model)
            y_test_pred, y_test_prob = tester.test(x_test)

            val_metrics = Metrics.evaluate(
                y_val, y_val_pred, y_val_prob
            )
            test_metrics = Metrics.evaluate(
                y_test, y_test_pred, y_test_prob
            )

            Metrics.pretty_print("Validation Metrics", val_metrics)
            Metrics.pretty_print("Test Metrics", test_metrics)


if __name__ == "__main__":
    pipeline = LogisticRegressionPipeline(
        dataset_path="data.csv",
        target_column="diagnosis",
        scaling_method="standardize"
    )

    pipeline.run()
