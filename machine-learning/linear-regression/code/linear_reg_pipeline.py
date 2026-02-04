import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DatasetLoader:
    """
    Responsible for loading dataset from a given file path
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """
        Reads CSV file and returns a pandas DataFrame
        """
        return pd.read_csv(self.file_path)


class DatasetVerifier:
    """
    Performs basic dataset validation checks
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def verify(self):
        """
        Ensures dataset is not empty
        """
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

    def handle_missing_values(self):
        """
        Drops rows if missing values are less than or equal to 20%
        Otherwise applies forward-fill followed by backward-fill
        """
        df = self.dataframe.copy()

        # Maximum null percentage across all columns
        null_percentage = df.isnull().mean().max()

        if null_percentage <= 0.2:
            df.dropna(inplace=True)
        else:
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

        return df

    def encode_categorical(self, df):
        """
        Applies one-hot encoding only if a categorical column
        has 10 or fewer unique values
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

    def preprocess(self):
        """
        Executes full preprocessing pipeline
        """
        df = self.handle_missing_values()
        df = self.encode_categorical(df)

        # Separate features and target
        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        return x, y


class DatasetConverter:
    """
    Performs feature scaling
    """

    def __init__(self, method="standardize"):
        self.method = method
        self.scaler = None

    def fit_transform(self, x_train):
        """
        Fits scaler on training data and transforms it
        """
        if self.method == "normalize":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        return self.scaler.fit_transform(x_train)

    def transform(self, x):
        """
        Transforms validation and test data
        """
        return self.scaler.transform(x)


class DatasetVisualizer:
    """
    Handles dataset visualization
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def pairplot(self):
        """
        Generates pairplot for feature relationships
        """
        sns.pairplot(self.dataframe)
        plt.show()


class Model:
    """
    Wrapper class for regression models
    """

    def __init__(self, model_type="linear", alpha=1.0):
        if model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()

    def train(self, x_train, y_train):
        """
        Trains regression model
        """
        self.model.fit(x_train, y_train)

    def predict(self, x):
        """
        Generates predictions
        """
        return self.model.predict(x)


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
        return self.model.predict(x_val)


class DatasetTester:
    """
    Handles test predictions
    """

    def __init__(self, model):
        self.model = model

    def test(self, x_test):
        return self.model.predict(x_test)


class Metrics:
    """
    Computes regression evaluation metrics
    """

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def pretty_print(title, metrics_dict):
        print(title)
        print("-" * len(title))
        for key, value in metrics_dict.items():
            print(f"{key:<5}: {value:.4f}")
        print()


class LinearRegressionPipeline:
    """
    End-to-end pipeline orchestrating all components
    """

    def __init__(
        self,
        dataset_path,
        target_column,
        scaling_method,
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
        Executes full ML pipeline
        """

        # Load dataset
        loader = DatasetLoader(self.dataset_path)
        df = loader.load()

        # Verify dataset
        verifier = DatasetVerifier(df)
        verifier.verify()

        # Visualize dataset
        visualizer = DatasetVisualizer(df)
        visualizer.pairplot()

        # Preprocess dataset
        preprocessor = DatasetPreprocessor(df, self.target_column)
        x, y = preprocessor.preprocess()

        # Split dataset into train, validation, and test
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=self.test_size + self.val_size,
            random_state=42
        )

        relative_val_size = self.val_size / (
            self.test_size + self.val_size
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=1 - relative_val_size,
            random_state=42
        )

        # Scale features
        converter = DatasetConverter(self.scaling_method)
        x_train = converter.fit_transform(x_train)
        x_val = converter.transform(x_val)
        x_test = converter.transform(x_test)

        # Hyperparameter configurations to validate
        hyperparameters = [
            {"model_type": "linear"},
            {"model_type": "ridge", "alpha": 0.1},
            {"model_type": "ridge", "alpha": 1.0},
            {"model_type": "lasso", "alpha": 0.1},
            {"model_type": "lasso", "alpha": 1.0}
        ]

        results = {}

        # Train, validate, and test each configuration
        for params in hyperparameters:
            model = Model(
                model_type=params.get("model_type"),
                alpha=params.get("alpha", 1.0)
            )

            trainer = DatasetTrainer(model)
            trainer.train(x_train, y_train)

            validator = DatasetValidator(model)
            y_val_pred = validator.validate(x_val)

            tester = DatasetTester(model)
            y_test_pred = tester.test(x_test)

            # results[str(params)] = {
            #     "validation_metrics": Metrics.evaluate(
            #         y_val, y_val_pred
            #     ),
            #     "test_metrics": Metrics.evaluate(
            #         y_test, y_test_pred
            #     )
            # }

            print("\n==============================================")
            print(f"Model Configuration: {params}")
            print("==============================================\n")

            val_metrics = Metrics.evaluate(y_val, y_val_pred)
            test_metrics = Metrics.evaluate(y_test, y_test_pred)

            Metrics.pretty_print("Validation Metrics", val_metrics)
            Metrics.pretty_print("Test Metrics", test_metrics)

        return results


if __name__ == "__main__":
    pipeline = LinearRegressionPipeline(
        dataset_path="test_energy_data.csv",
        target_column="Energy Consumption",
        scaling_method="normalize"
    )

    metrics_output = pipeline.run()
    # print(metrics_output)
