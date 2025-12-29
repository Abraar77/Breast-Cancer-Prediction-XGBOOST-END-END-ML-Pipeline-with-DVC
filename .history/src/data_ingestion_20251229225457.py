import os
import pandas as pd

INPUT_PATH = "data/source/wdbc.data"
OUTPUT_DIR = "data/raw"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "wdbc.csv")

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    
    if df.empty:
        raise ValueError("Dataset is empty")

    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
