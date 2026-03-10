import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ================= GLOBAL VARIABLES =================

RANDOM_STATE = 42
MISSING_VALUE_THRESHOLD = 0.2
MAX_CATEGORICAL_UNIQUE = 10


# ================= DATASET LOADER =================

class DatasetLoader:
    """
    Loads dataset from CSV file
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return pd.read_excel(self.file_path, header=1)


# ================= DATASET VERIFIER =================

class DatasetVerifier:
    """
    Verifies dataset validity
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def verify(self):
        print(self.dataframe.info())
        if self.dataframe.empty:
            raise ValueError("Dataset is empty")
        return True


# ================= DATASET PREPROCESSOR =================

class DatasetPreprocessor:
    """
    Handles missing values and categorical encoding
    """

    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column

    def handle_missing_values(self):

        df = self.dataframe.copy()

        max_null_percentage = df.isnull().mean().max()

        if max_null_percentage <= MISSING_VALUE_THRESHOLD:

            df.dropna(inplace=True)

        else:

            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

        return df

    def encode_categorical(self, df):

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in categorical_columns:

            if df[column].nunique() <= MAX_CATEGORICAL_UNIQUE:

                df = pd.get_dummies(
                    df,
                    columns=[column],
                    drop_first=True
                )

        return df

    def preprocess(self):

        df = self.handle_missing_values()

        df = self.encode_categorical(df)

        x = df.drop(columns=[self.target_column])

        y = df[self.target_column]

        return x, y, df


# ================= DATASET CONVERTER =================

class DatasetConverter:
    """
    Performs scaling or normalization
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


# ================= DATASET VISUALIZER =================

class DatasetVisualizer:
    """
    Handles pairplot and correlation visualization
    """

    def __init__(self, dataframe, target_column):

        self.dataframe = dataframe

        self.target_column = target_column

    def pairplot(self):

        sns.pairplot(self.dataframe)

        plt.show()

    def correlation_heatmap(self):

        correlation_matrix = self.dataframe.corr(
            numeric_only=True
        )

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Matrix")

        plt.show()

    def target_correlation(self):

        correlation_matrix = self.dataframe.corr(
            numeric_only=True
        )

        if self.target_column in correlation_matrix:

            print("\nTarget Correlation:\n")

            print(
                correlation_matrix[self.target_column]
                .sort_values(ascending=False)
            )


# ================= SVM MODEL =================

class SVMModel:
    """
    Wrapper class to handle both SVC and SVR
    """

    def __init__(
        self,
        task_type,
        kernel="rbf",
        c_value=1.0,
        gamma="scale"
    ):

        self.task_type = task_type

        if task_type == "classification":

            self.model = SVC(
                kernel=kernel,
                C=c_value,
                gamma=gamma,
                probability=True,
                random_state=RANDOM_STATE
            )

        else:

            self.model = SVR(
                kernel=kernel,
                C=c_value,
                gamma=gamma
            )

    def train(self, x_train, y_train):

        self.model.fit(x_train, y_train)

    def predict(self, x):

        return self.model.predict(x)


# ================= METRICS =================

class Metrics:

    @staticmethod
    def classification_metrics(y_true, y_pred):

        return {

            "Accuracy": accuracy_score(y_true, y_pred),

            "Precision": precision_score(
                y_true,
                y_pred,
                average="weighted"
            ),

            "Recall": recall_score(
                y_true,
                y_pred,
                average="weighted"
            ),

            "F1 Score": f1_score(
                y_true,
                y_pred,
                average="weighted"
            )
        }

    @staticmethod
    def regression_metrics(y_true, y_pred):

        return {

            "R2 Score": r2_score(y_true, y_pred),

            "MAE": mean_absolute_error(y_true, y_pred),

            "RMSE": np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        }

    @staticmethod
    def pretty_print(title, metrics):

        print(title)

        print("-" * len(title))

        for key, value in metrics.items():

            print(f"{key:<10}: {value:.4f}")

        print()


# ================= PIPELINE =================

class SVMPipeline:

    def __init__(
        self,
        dataset_path,
        target_column,
        scaling_method="standardize"
    ):

        self.dataset_path = dataset_path

        self.target_column = target_column

        self.scaling_method = scaling_method

    def detect_task(self, y):

        if y.nunique() <= 10:

            return "classification"

        return "regression"

    def run(self):

        loader = DatasetLoader(self.dataset_path)

        df = loader.load()

        verifier = DatasetVerifier(df)

        verifier.verify()

        preprocessor = DatasetPreprocessor(
            df,
            self.target_column
        )

        x, y, processed_df = preprocessor.preprocess()

        # visualizer = DatasetVisualizer(
        #     processed_df,
        #     self.target_column
        # )

        # visualizer.pairplot()

        # visualizer.correlation_heatmap()

        # visualizer.target_correlation()

        task_type = self.detect_task(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

        converter = DatasetConverter(self.scaling_method)

        x_train = converter.fit_transform(x_train)

        x_test = converter.transform(x_test)

        hyperparameters = [

            {"kernel": "linear", "C": 0.1},

            {"kernel": "linear", "C": 1},

            {"kernel": "rbf", "C": 1},

            {"kernel": "rbf", "C": 10}

        ]

        for params in hyperparameters:

            print("\n================================")

            print(f"SVM Params: {params}")

            print("================================\n")

            model = SVMModel(
                task_type,
                kernel=params["kernel"],
                c_value=params["C"]
            )

            model.train(x_train, y_train)

            predictions = model.predict(x_test)

            if task_type == "classification":

                metrics = Metrics.classification_metrics(
                    y_test,
                    predictions
                )

            else:

                metrics = Metrics.regression_metrics(
                    y_test,
                    predictions
                )

            Metrics.pretty_print(
                "Test Metrics",
                metrics
            )


# ================= MAIN =================

if __name__ == "__main__":

    pipeline = SVMPipeline(

        dataset_path="default of credit card clients.xlsx",

        target_column="default payment next month",

        scaling_method="standardize"

    )

    pipeline.run()
