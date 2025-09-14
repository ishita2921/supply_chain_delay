# scripts/train.py
import argparse
from pathlib import Path
import joblib
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.supply_chain import utils
from glob import glob

def find_data():
    patterns = [
        "data/processed/*.parquet", "data/processed/*.csv",
        "data/interim/*.parquet", "data/interim/*.csv",
    ]
    for p in patterns:
        files = glob(p)
        if files:
            return files[0]
    raise FileNotFoundError("No training data found in data/processed or data/interim")

def main(seed=42):
    utils.set_seed(seed)
    data_path = find_data()
    print(f"Using training data: {data_path}")

    # Load data
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df = utils.standardize_columns(df)

    # Target
    y = df["delay_flag"].astype(int)
    X = df.drop(columns=["delay_flag"])

    # Feature groups
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessing
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ],
        remainder="drop",
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    preproc.fit(X_train)
    X_train_t = preproc.transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_t, y_train)

    # Save artifacts
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "artifacts/best_model.pkl")
    joblib.dump(preproc, "models/feature_pipeline.pkl")

    with open("models/raw_feature_names.json", "w") as f:
        json.dump(numeric_cols + cat_cols, f)

    print("✅ Training complete — artifacts written to artifacts/ and models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
