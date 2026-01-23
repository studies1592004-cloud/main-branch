import numpy as np
from model import LinearRegressionGD
from pipeline import DataPipeline


def take_user_input(features):
    values = []
    for feature in features:
        val = float(input(f"Enter {feature}: "))
        values.append(val)
    return np.array(values).reshape(1, -1)


def main():
    method = input("Enter gradient type (bgd/sgd/mbgd): ")

    pipeline = DataPipeline("test_energy_data.csv")
    X_train, X_test, y_train, y_test = pipeline.preprocess()
    X_train, X_test = pipeline.normalize(X_train, X_test)

    model = LinearRegressionGD(lr=0.01, epochs=10000)
    model.fit(X_train, y_train, method=method)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
    print("\nRMSE:", rmse)

    print("\n--- Predict Your Own Energy Consumption ---")
    x_user = take_user_input(pipeline.features)
    x_user = pipeline.prepare_single_input(x_user)

    prediction = model.predict(x_user)
    print("\nPredicted Energy Consumption:", prediction[0])


if __name__ == "__main__":
    main()
