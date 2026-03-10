import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# ================= GLOBAL CONFIG =================

MISSING_VALUE_THRESHOLD = 0.2
MAX_CATEGORICAL_UNIQUE = 10
RANDOM_STATE = 42


# ================= DATASET LOADER =================

class DatasetLoader:
    """
    Loads dataset from CSV
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return pd.read_csv(self.file_path)


# ================= DATASET PREPROCESSOR =================

class DatasetPreprocessor:
    """
    Handles missing values, column dropping, and encoding
    """

    def __init__(
        self,
        dataframe,
        drop_columns=None
    ):
        self.dataframe = dataframe
        self.drop_columns = drop_columns or []

    def handle_missing_values(self, df):

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

    def preprocess(self):

        df = self.dataframe.copy()

        # Drop user defined columns first
        df = self.drop_user_columns(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical columns
        df = self.encode_categorical(df)

        return df


# ================= DATASET CONVERTER =================

class DatasetConverter:
    """
    Handles scaling
    """

    def __init__(self, method="standardize"):
        self.method = method
        self.scaler = None

    def fit_transform(self, x):

        if self.method == "normalize":
            self.scaler = MinMaxScaler()

        else:
            self.scaler = StandardScaler()

        return self.scaler.fit_transform(x)


# ================= DATASET VISUALIZER =================

class DatasetVisualizer:
    """
    Handles visualizations
    """

    def __init__(self, dataframe):

        self.dataframe = dataframe

    def pairplot(self):

        sns.pairplot(self.dataframe)

        plt.show()

    def correlation_heatmap(self):

        corr = self.dataframe.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Matrix")

        plt.show()

    def plot_clusters(self, x, labels, title):
        """
        Uses PCA to reduce dimensions to 2D for proper visualization
        """

        # Reduce dimensions to 2 using PCA
        pca = PCA(n_components=2)

        x_pca = pca.fit_transform(x)

        # Plot PCA components
        plt.figure(figsize=(8, 6))

        plt.scatter(
            x_pca[:, 0],
            x_pca[:, 1],
            c=labels,
            cmap="viridis",
            s=50
        )

        plt.title(title)

        plt.xlabel("Principal Component 1")

        plt.ylabel("Principal Component 2")

        plt.colorbar(label="Cluster")

        plt.show()


    def plot_dendrogram(self, x):

        linkage_matrix = linkage(x, method="ward")

        plt.figure(figsize=(12, 6))

        dendrogram(linkage_matrix)

        plt.title("Hierarchical Clustering Dendrogram")

        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        plt.show()

    def plot_elbow_method(self, x, max_k=10):
        """
        Plots elbow curve to determine optimal number of clusters
        """

        inertia_values = []

        k_range = range(1, max_k + 1)

        for k in k_range:

            kmeans = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10
            )

            kmeans.fit(x)

            inertia_values.append(kmeans.inertia_)

        print("\nK vs Inertia:")
        for k, inertia in zip(k_range, inertia_values):
            print(f"K={k}, Inertia={inertia:.2f}")

        plt.figure(figsize=(8, 6))

        plt.plot(
            k_range,
            inertia_values,
            marker="o"
        )

        plt.title("Elbow Method for Optimal Number of Clusters")

        plt.xlabel("Number of Clusters (K)")

        plt.ylabel("Inertia (WCSS)")

        plt.xticks(k_range)

        plt.grid(True)

        plt.show()



# ================= CLUSTER MODELS =================

class ClusteringModel:
    """
    Supports KMeans, Hierarchical, DBSCAN
    """

    def __init__(
        self,
        algorithm,
        n_clusters=3,
        eps=0.5,
        min_samples=5
    ):

        self.algorithm = algorithm

        if algorithm == "kmeans":

            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE
            )

        elif algorithm == "hierarchical":

            self.model = AgglomerativeClustering(
                n_clusters=n_clusters
            )

        elif algorithm == "dbscan":

            self.model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

        else:

            raise ValueError("Invalid algorithm")

    def train_predict(self, x):

        return self.model.fit_predict(x)


# ================= PIPELINE =================

class ClusteringPipeline:
    """
    Complete clustering pipeline
    """

    def __init__(
        self,
        dataset_path,
        algorithm="kmeans",
        drop_columns=None,
        scaling_method="standardize"
    ):

        self.dataset_path = dataset_path

        self.algorithm = algorithm

        self.drop_columns = drop_columns or []

        self.scaling_method = scaling_method

    def run(self):

        loader = DatasetLoader(self.dataset_path)

        df = loader.load()

        preprocessor = DatasetPreprocessor(
            df,
            drop_columns=self.drop_columns
        )

        processed_df = preprocessor.preprocess()

        visualizer = DatasetVisualizer(processed_df)

        # visualizer.pairplot()

        # visualizer.correlation_heatmap()

        converter = DatasetConverter(self.scaling_method)

        x_scaled = converter.fit_transform(processed_df)

        if self.algorithm == "hierarchical":

            visualizer.plot_dendrogram(x_scaled)

        hyperparameters = []

        if self.algorithm == "kmeans":

            visualizer.plot_elbow_method(x_scaled)

            hyperparameters = [

                {"n_clusters": 2},

                {"n_clusters": 3},

                {"n_clusters": 4},

                {"n_clusters": 5}

            ]

        elif self.algorithm == "hierarchical":

            hyperparameters = [

                {"n_clusters": 2},

                {"n_clusters": 3},

                {"n_clusters": 4}

            ]

        elif self.algorithm == "dbscan":

            hyperparameters = [

                {"eps": 0.3, "min_samples": 5},

                {"eps": 0.5, "min_samples": 5},

                {"eps": 0.7, "min_samples": 10}

            ]

        for params in hyperparameters:

            print("\n================================")

            print(f"Algorithm: {self.algorithm}")

            print(f"Params: {params}")

            print("================================\n")

            model = ClusteringModel(
                algorithm=self.algorithm,
                n_clusters=params.get("n_clusters", 3),
                eps=params.get("eps", 0.5),
                min_samples=params.get("min_samples", 5)
            )

            labels = model.train_predict(x_scaled)

            visualizer.plot_clusters(
                x_scaled,
                labels,
                f"{self.algorithm} Clusters {params}"
            )


# ================= MAIN =================

if __name__ == "__main__":

    pipeline = ClusteringPipeline(

        dataset_path="Customer_Data.csv",

        algorithm="hierarchical",  # kmeans, hierarchical, dbscan

        drop_columns=[
            "CUST_ID"
        ],

        scaling_method="standardize"

    )

    pipeline.run()
