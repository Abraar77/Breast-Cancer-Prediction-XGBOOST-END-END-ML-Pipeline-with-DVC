# src/feature_engineering.py

import numpy as np
import pandas as pd
import os

TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH= "data/processed/test.csv"
OUTPUT_DIR = "data/features"

def engineer_features(dff):
    df= pd.read_csv(dff)
    """
    Creates domain-inspired engineered features.
    """

    df = df.copy()

    # Ratio features (mean vs worst)
    df["radius_ratio"] = df["radius_mean"] / (df["radius_worst"] + 1e-6)
    df["texture_ratio"] = df["texture_mean"] / (df["texture_worst"] + 1e-6)
    df["perimeter_ratio"] = df["perimeter_mean"] / (df["perimeter_worst"] + 1e-6)
    df["area_ratio"] = df["area_mean"] / (df["area_worst"] + 1e-6)

    # Variance-based feature
    df["radius_variance"] = df["radius_worst"] - df["radius_mean"]
    df["area_variance"] = df["area_worst"] - df["area_mean"]

    # Composite tumor severity score
    df["tumor_severity_score"] = (
        df["radius_mean"] +
        df["perimeter_mean"] +
        df["area_mean"] +
        df["concavity_mean"]
    )


def main():
     # --- create output directory (MANDATORY for DVC) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data=engineer_features(TRAIN_DATA_PATH)
    test_data=engineer_features(TEST_DATA_PATH)
    train_df = pd.DataFrame(X_t, columns=X.columns)
    
    
    train_data.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_data.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
