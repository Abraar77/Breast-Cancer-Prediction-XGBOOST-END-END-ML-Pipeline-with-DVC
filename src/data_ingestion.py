import os
import pandas as pd

INPUT_PATH = "data/source/wdbc.data"
OUTPUT_DIR = "data/raw"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "wdbc.csv")

COLUMN_NAMES = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = COLUMN_NAMES
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
