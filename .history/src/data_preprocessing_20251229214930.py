# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Cleans data, encodes labels, splits, scales and applies SMOTE.
    """

    # Drop ID column
    df = df.drop(columns=["id"], errors="ignore")

    # Encode target
    encoder = LabelEncoder()
    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled, y_train
    )

    return (
        X_train_smote,
        X_test_scaled,
        y_train_smote,
        y_test,
        scaler,
        encoder
    )
