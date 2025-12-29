import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "data/features/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def train_xgboost(X, y):
    """
    Trains XGBoost using GridSearchCV.
    """

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.01],
        "n_estimators": [100, 200]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    # --- create model directory (MANDATORY for DVC) ---
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- load training data ---
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # --- split X and y ---
    X_train = train_df.drop(columns=["diagnosis"])
    y_train = train_df["diagnosis"]

    # --- train model ---
    best_model, best_params = train_xgboost(X_train, y_train)

    # --- save model ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("Model trained successfully")
    print("Best parameters:", best_params)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
