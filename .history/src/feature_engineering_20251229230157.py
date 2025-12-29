# src/feature_engineering.py

import numpy as np
import pandas as pd

TRAIN_DATA_PATH = "data/preprossed/train.csv"
TEST_DATA_PATH= "data/preprossed/test.csv"
OUTPUT_DIR = "data/processed"

def engineer_features() -> pd.DataFrame:
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

    return df
