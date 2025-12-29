# src/data_ingestion.py

import pandas as pd

path="../data/wdbc"

def load_data(: str) -> pd.DataFrame:
    """
    Loads raw breast cancer dataset.
    """
    df = pd.read_csv(file_path)

    # basic validation
    if df.empty:
        raise ValueError("Dataset is empty")

    return df
