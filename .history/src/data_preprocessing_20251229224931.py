import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

RAW_DATA_PATH = "data/raw/wdbc.csv"
OUTPUT_DIR = "data/processed"

def main():
    # --- create output directory (MANDATORY for DVC) ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- load data ---
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop ID column if exists
    df = df.drop(columns=["id"], errors="ignore")

    # Encode target
    encoder = LabelEncoder()
    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

    
    y = df["diagnosis"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle imbalance (ONLY on training data — correct)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled, y_train
    )

    # --- save outputs ---
    train_df = pd.DataFrame(X_train_smote, columns=X.columns)
    train_df["diagnosis"] = y_train_smote

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["diagnosis"] = y_test.values

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved:")
    print(" - data/processed/train.csv")
    print(" - data/processed/test.csv")

if __name__ == "__main__":
    main()
