# src/data_ingestion.py

import pandas as pd

path="../data/wdbc.data"

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw breast cancer dataset.
    """
    df = pd.read_csv(path)

    # basic validation
    if df.empty:
        raise ValueError("Dataset is empty")

    return df
