# scripts/train_model.py
"""
Train a model, save:
 - models/feature_pipeline.pkl       (ColumnTransformer)
 - models/raw_feature_names.json
 - artifacts/best_model.pkl
 - artifacts/threshold.json
 - artifacts/shap_background.pkl
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# -------------------
# Paths
# -------------------
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH   = Path("data/processed/val.csv")
TEST_PATH  = Path("data/processed/test.csv")


OUT_MODELS = Path("models")
OUT_ARTIFACTS = Path("artifacts")
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# -------------------
# Load data
# -------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Standardize column names
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_")
val_df.columns = val_df.columns.str.strip().str.lower().str.replace(" ", "_")

# -------------------
# Target variable
# -------------------
TARGET = "logistics_delay"   # change if your column is named differently
y_train = train_df[TARGET].astype(int)
y_val = val_df[TARGET].astype(int)

# -------------------
# Features
# -------------------
raw_numeric = train_df.select_dtypes(include=[np.number]).columns.drop([TARGET]).tolist()
raw_categorical = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
raw_categorical = [c for c in raw_categorical if train_df[c].nunique() < 200]

FEATURES = raw_numeric + raw_categorical
X_train = train_df[FEATURES].copy()
X_val = val_df[FEATURES].copy()

# Save raw feature list for the app
with open(OUT_MODELS / "raw_feature_names.json", "w") as f:
    json.dump(FEATURES, f)

# -------------------
# Preprocessor
# -------------------
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, raw_numeric),
    ("cat", cat_transformer, raw_categorical),
], remainder="drop")

# -------------------
# Train classifier
# -------------------
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_trans, y_train)

# -------------------
# Save artifacts
# -------------------
joblib.dump(preprocessor, OUT_MODELS / "feature_pipeline.pkl")
joblib.dump(clf, OUT_ARTIFACTS / "best_model.pkl")

# Background sample for SHAP
bg = X_train.sample(n=min(200, len(X_train)), random_state=42)
joblib.dump(bg, OUT_ARTIFACTS / "shap_background.pkl")

# -------------------
# Threshold (maximize F1)
# -------------------
probs = clf.predict_proba(X_val_trans)[:, 1]
best_f1 = -1
best_thr = 0.5
for thr in np.linspace(0.01, 0.99, 99):
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

with open(OUT_ARTIFACTS / "threshold.json", "w") as f:
    json.dump({"threshold": float(best_thr), "f1": float(best_f1)}, f)

print("✅ Saved artifacts:")
print(" -", OUT_MODELS / "feature_pipeline.pkl")
print(" -", OUT_ARTIFACTS / "best_model.pkl")
print(" -", OUT_ARTIFACTS / "threshold.json")
print(" -", OUT_ARTIFACTS / "shap_background.pkl")
