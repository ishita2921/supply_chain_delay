import glob
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# File handling
# -------------------------------------------------------------------
def candidate_files() -> list:
    """
    Search for candidate data files in standard folders.
    """
    patterns = [
        "data/interim/*.parquet", "data/interim/*.csv",
        "data/processed/*.parquet", "data/processed/*.csv",
        "data/raw/*.parquet", "data/raw/*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


# -------------------------------------------------------------------
# Data preprocessing
# -------------------------------------------------------------------
def to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names and ensure required fields exist.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "supplier" not in df.columns:
        df["supplier"] = "DefaultSupplier"
    if "mode" not in df.columns:
        df["mode"] = "Road"
    if "region" not in df.columns:
        df["region"] = "UnknownRegion"
    if "event_date" not in df.columns:
        df["event_date"] = pd.Timestamp.today().normalize()

    df["event_date"] = to_datetime(df["event_date"])
    df["month"] = df["event_date"].dt.month

    # 🚨 Fix leakage
    if "logistics_delay" in df.columns:
        df["delay_flag"] = df["logistics_delay"].astype(bool)
        df = df.drop(columns=["logistics_delay"])  # prevent leakage
    else:
        df["delay_flag"] = False

    return df



# -------------------------------------------------------------------
# Model + pipeline helpers
# -------------------------------------------------------------------
def load_model_and_pipeline():
    """
    Load trained model, preprocessing pipeline, and threshold.
    """
    model = joblib.load("artifacts/best_model.pkl")
    pipeline = joblib.load("models/feature_pipeline.pkl")

    with open("models/raw_feature_names.json", "r") as f:
        raw_feature_names = json.load(f)

    threshold = 0.5
    thr_path = Path("artifacts/threshold.json")
    if thr_path.exists():
        with open(thr_path, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    return model, pipeline, threshold, raw_feature_names


def predict_probability(model, pipeline, X_raw: pd.DataFrame) -> float:
    """
    Return the probability of delay for a given input row.
    """
    Xt = pipeline.transform(X_raw)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xt)[:, 1][0])
    return float(model.predict(Xt)[0])


# -------------------------------------------------------------------
# Feature name utilities
# -------------------------------------------------------------------
def get_transformed_feature_names(pipeline: Pipeline, raw_features: list) -> list:
    out = []
    for name, trans, cols in pipeline.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        else:
            names = cols
        out.extend(names)
    return out


def clean_feature_name(name: str) -> str:
    """
    Convert transformed feature names into human-friendly labels.
    Example:
        ohe__region_North -> "region = North"
        numeric__month    -> "month"
    """
    if "__" in name:
        base = name.split("__", 1)[-1]
        # OneHotEncoded case: e.g. region_North
        if "_" in base:
            parts = base.split("_", 1)
            if len(parts) == 2:
                return f"{parts[0]} = {parts[1]}"
        # Numeric/scalar case: e.g. month
        return base
    return name


def input_value_for_transformed_feature(name: str, X_input: pd.DataFrame):
    if "=" in name:  # OHE feature
        feat, cat = name.split("=", 1)
        feat = feat.strip()
        if feat in X_input.columns:
            return str(X_input.iloc[0][feat])
        return cat.strip()
    else:
        base = name.split("__")[-1]
        if base in X_input.columns:
            return X_input.iloc[0][base]
        return None
