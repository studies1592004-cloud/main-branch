import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ================= GLOBAL CONFIGURATION =================

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
        return pd.read_csv(self.file_path)


# ================= DATASET VERIFIER =================

class DatasetVerifier:
    """
    Verifies dataset integrity
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def verify(self):
        if self.dataframe.empty:
            raise ValueError("Dataset is empty")
        return True


# ================= DATASET PREPROCESSOR =================

class DatasetPreprocessor:
    """
    Handles missing values, column dropping, and encoding
    """

    def __init__(
        self,
        dataframe,
        target_column,
        drop_columns=None
    ):
        self.dataframe = dataframe
        self.target_column = target_column
        self.drop_columns = drop_columns or []

    def drop_user_columns(self, df):
        """
        Drops user-specified columns safely
        """

        valid_columns = [
            col for col in self.drop_columns
            if col in df.columns
        ]

        if valid_columns:

            print("\nDropping user-specified columns:")

            for col in valid_columns:
                print(col)

            df = df.drop(columns=valid_columns)

        return df


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

        # Drop user specified columns first
        df = self.drop_user_columns(df)

        df = self.encode_categorical(df)

        x = df.drop(columns=[self.target_column])

        y = df[self.target_column]

        return x, y, df


# ================= DATASET CONVERTER =================

class DatasetConverter:
    """
    Performs scaling only on numeric columns.
    Object and category columns are left unchanged.
    """

    def __init__(self, method="standardize"):

        self.method = method

        self.scaler = None

        self.numeric_columns = None

    def fit_transform(self, x_train):
        """
        Fits scaler on numeric columns and transforms them
        """

        # Convert to DataFrame if numpy array
        if isinstance(x_train, np.ndarray):
            x_train = pd.DataFrame(x_train)

        # Identify numeric columns only
        self.numeric_columns = x_train.select_dtypes(
            include=[np.number]
        ).columns

        # Initialize scaler
        if self.method == "normalize":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Scale numeric columns
        x_train_scaled = x_train.copy()

        x_train_scaled[self.numeric_columns] = self.scaler.fit_transform(
            x_train[self.numeric_columns]
        )

        return x_train_scaled

    def transform(self, x):
        """
        Transforms numeric columns using previously fitted scaler
        """

        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)

        x_scaled = x.copy()

        x_scaled[self.numeric_columns] = self.scaler.transform(
            x[self.numeric_columns]
        )

        return x_scaled


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

        correlation_matrix = self.dataframe.corr(numeric_only=True)

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

        correlation_matrix = self.dataframe.corr(numeric_only=True)

        if self.target_column in correlation_matrix:

            print("\nTarget Correlation:\n")

            print(
                correlation_matrix[self.target_column]
                .sort_values(ascending=False)
            )


# ================= MODEL CLASS =================

class TreeModel:
    """
    Wrapper class to choose Decision Tree or Random Forest
    and Classification or Regression automatically
    """

    def __init__(
        self,
        model_type,
        task_type,
        max_depth=None,
        n_estimators=100
    ):

        self.model_type = model_type

        self.task_type = task_type

        if task_type == "classification":

            if model_type == "decision_tree":

                self.model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=RANDOM_STATE
                )

            else:

                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE
                )

        else:

            if model_type == "decision_tree":

                self.model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    random_state=RANDOM_STATE
                )

            else:

                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE
                )

    def train(self, x_train, y_train):

        self.model.fit(x_train, y_train)

    def predict(self, x):

        return self.model.predict(x)


# ================= METRICS =================

class Metrics:
    """
    Computes and prints metrics
    """

    @staticmethod
    def regression_metrics(y_true, y_pred):

        return {

            "R2": r2_score(y_true, y_pred),

            "MAE": mean_absolute_error(y_true, y_pred),

            "RMSE": np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        }

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

            "F1": f1_score(
                y_true,
                y_pred,
                average="weighted"
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

class TreePipeline:

    def __init__(
        self,
        dataset_path,
        target_column,
        model_type,
        drop_columns=None,
        scaling_method="standardize"
    ):

        self.dataset_path = dataset_path

        self.target_column = target_column

        self.model_type = model_type

        self.drop_columns = drop_columns or []

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
            self.target_column,
            drop_columns=self.drop_columns
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

            {"max_depth": 3, "n_estimators": 50},

            {"max_depth": 5, "n_estimators": 100},

            {"max_depth": 10, "n_estimators": 200}

        ]

        for params in hyperparameters:

            print("\n================================")

            print(
                f"Model: {self.model_type} | Params: {params}"
            )

            print("================================\n")

            model = TreeModel(
                self.model_type,
                task_type,
                params["max_depth"],
                params["n_estimators"]
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

    pipeline = TreePipeline(

        dataset_path="churn raw data.csv",

        target_column="Exited",

        model_type="random_forest",  # or "decision_tree"

        drop_columns=[
            "Surname",
            "Unnamed: 32",
        ],

        scaling_method="standardize"

    )

    pipeline.run()
