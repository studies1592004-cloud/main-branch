import numpy as np
from model import LogisticRegression
from pipeline import DataPipeline
from sklearn.metrics import confusion_matrix, classification_report


def take_user_input(features):
    values = []
    for feature in features:
        val = float(input(f"Enter {feature}: "))
        values.append(val)
    return np.array(values).reshape(1, -1)


def main():
    method = input("Enter gradient type (bgd/sgd/mbgd): ")

    pipeline = DataPipeline("data.csv")
    X_train, X_test, y_train, y_test = pipeline.preprocess()
    X_train, X_test = pipeline.normalize(X_train, X_test)

    model = LogisticRegression(lr=0.01, epochs=10000)
    model.fit(X_train, y_train, method=method)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nPredict Your Own Input:")
    x_user = take_user_input(pipeline.features)
    x_user = pipeline.prepare_single_input(x_user)

    prob = model.predict_proba(x_user)[0]
    label = model.predict(x_user)[0]

    print(f"\nProbability of Malignant (1): {prob:.4f}")
    print("Predicted Diagnosis:", "Malignant" if label == 1 else "Benign")


if __name__ == "__main__":
    main()
