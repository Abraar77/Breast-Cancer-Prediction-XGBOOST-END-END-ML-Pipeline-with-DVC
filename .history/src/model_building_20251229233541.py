# src/model_building.py

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os
import os
import pickle
TRAIN_DATA_PATH = "data/features/train.csv"
TEST_DATA_PATH= "data/features/test.csv"



def train_xgboost(x_train, y_train):
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
    x_train= train_df.iloc(:,-1)
    y_train= train_df.iloc(:,-1)

    model = XGBClassifier(x_train,y_train)
    
    
    

if __name__ == "__main__":
    main()
