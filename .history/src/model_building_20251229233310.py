# src/model_building.py

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os
import os
import pickle
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH= "data/processed/test.csv"
# OUTPUT_DIR = "data/features"


def train_xgboost(X_train, y_train):
    """
    Trains XGBoost using GridSearchCV.
    """

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    param_grid = {
        "max_depth": [3, 5, 7],
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

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_

def main():
    

   
    train_df = pd.DataFrame(TRAIN_DATA_PATH)
    x_train= train_df.iloc(:,:-1)
    y_train= train_df.iloc()
    # test_df = pd.DataFrame(test_data)
    
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
