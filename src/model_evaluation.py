import os
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "models/model.pkl"
TEST_DATA_PATH = "data/features/test.csv"
METRICS_DIR = "metrics"
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    # load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # load test data
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]

    # predictions
    y_pred = model.predict(X_test)

    # metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics saved:")
    print(metrics)


if __name__ == "__main__":
    main()
