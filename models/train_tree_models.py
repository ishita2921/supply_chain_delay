"""
Day 8 — Tree Models (RandomForest / XGBoost)

- Time-based split
- Baselines (Dummy + Logistic Regression)
- RandomForest + XGBoost with small grid search (TimeSeriesSplit CV)
- Evaluation on time-based test set
- Metric table, plots, and best model saved
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# ------------------ Helpers ------------------ #
def ensure_dirs(base: Path):
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train tree models with time-based split")
    parser.add_argument("--data", type=str, default="data/interim/smart_logistics_cleaned.csv")
    parser.add_argument("--target", type=str, default="logistics_delay")
    parser.add_argument("--timestamp_col", type=str, default="timestamp")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def time_based_split(df: pd.DataFrame, timestamp_col: str, test_frac: float):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["__date"] = df[timestamp_col].dt.date
    uniq_dates = np.array(sorted(df["__date"].unique()))
    n_test = max(1, int(len(uniq_dates) * test_frac))
    cutoff_date = uniq_dates[-n_test]
    train_idx = df["__date"] < cutoff_date
    test_idx = df["__date"] >= cutoff_date
    return df.loc[train_idx].drop(columns="__date"), df.loc[test_idx].drop(columns="__date")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    return ColumnTransformer([
        ("cat", categorical, cat_cols),
        ("num", numeric, num_cols)
    ])


def evaluate(model, X_test, y_test, title: str, out_dir: Path):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{title} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}_cm.png")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} — PR Curve (AP={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}_pr.png")
    plt.close()

    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": title,
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "accuracy": float(report["accuracy"]),
        "precision_0": float(report["0"]["precision"]),
        "recall_0": float(report["0"]["recall"]),
        "f1_0": float(report["0"]["f1-score"]),
        "precision_1": float(report["1"]["precision"]),
        "recall_1": float(report["1"]["recall"]),
        "f1_1": float(report["1"]["f1-score"]),
    }

def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Extract feature names after preprocessing (OneHot + numeric)."""
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols))
        elif name == "num":
            names.extend(cols)
    return np.array(names)


def plot_feature_importance(model, preprocessor, title: str, out_dir: Path, top_n: int = 20):
    """Plot top-N feature importances for RF/XGB."""
    try:
        clf = model.named_steps["clf"]  # classifier inside pipeline

        if hasattr(clf, "feature_importances_"):  # RF / XGB
            importances = clf.feature_importances_
            feat_names = get_feature_names(preprocessor)
        else:
            return  # skip if not supported

        order = np.argsort(importances)[::-1][:top_n]
        top_feats = feat_names[order]
        top_vals = importances[order]

        plt.figure(figsize=(8, 6))
        plt.barh(top_feats, top_vals)
        plt.gca().invert_yaxis()
        plt.title(f"{title} — Top {top_n} Features")
        plt.tight_layout()
        plt.savefig(out_dir / f"{title.replace(' ', '_')}_feature_importances.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Feature importance plot failed for {title}: {e}")



def compute_scale_pos_weight(y):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return max(1.0, neg / max(1, pos))


# ------------------ Main ------------------ #
def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    ensure_dirs(here)

    df = pd.read_csv(args.data)
    train_df, test_df = time_based_split(df, args.timestamp_col, args.test_frac)

    drop_cols = [args.timestamp_col, "asset_id", "shipment_status", "logistics_delay_reason", args.target]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[args.target]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[args.target]

    preprocessor = build_preprocessor(X_train)

    metrics = []

    # Dummy Baseline
    dummy = Pipeline([("prep", preprocessor), ("clf", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    metrics.append(evaluate(dummy, X_test, y_test, "Dummy Baseline", here / "plots"))

    # Logistic Regression Baseline
    logreg = Pipeline([("prep", preprocessor),
                       ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear"))])
    logreg.fit(X_train, y_train)
    metrics.append(evaluate(logreg, X_test, y_test, "LogReg Baseline", here / "plots"))

    # RandomForest
    rf = Pipeline([("prep", preprocessor),
                   ("clf", RandomForestClassifier(random_state=args.random_state,
                                                  class_weight="balanced_subsample"))])
    rf_params = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [None, 10],
        "clf__max_features": ["sqrt", 0.5],
    }
    rf_search = GridSearchCV(rf, rf_params, cv=TimeSeriesSplit(n_splits=3),
                             scoring="average_precision", n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    metrics.append(evaluate(best_rf, X_test, y_test, "RandomForest (best)", here / "plots"))
    plot_feature_importance(best_rf, preprocessor, "RandomForest (best)", here / "plots")


    # XGBoost
    best_xgb = None
    try:
        from xgboost import XGBClassifier
        spw = compute_scale_pos_weight(y_train)
        xgb = Pipeline([("prep", preprocessor),
                        ("clf", XGBClassifier(
                            random_state=args.random_state,
                            eval_metric="logloss",
                            n_jobs=-1,
                            scale_pos_weight=spw
                        ))])
        xgb_params = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.05, 0.1],
        }
        xgb_search = GridSearchCV(xgb, xgb_params, cv=TimeSeriesSplit(n_splits=3),
                                  scoring="average_precision", n_jobs=-1)
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        metrics.append(evaluate(best_xgb, X_test, y_test, "XGBoost (best)", here / "plots"))
        plot_feature_importance(best_xgb, preprocessor, "XGBoost (best)", here / "plots")

    except ImportError:
        print("[WARN] XGBoost not installed")

    # Save metrics table
    metrics_df = pd.DataFrame(metrics).sort_values("pr_auc", ascending=False)
    metrics_df.to_csv(here / "reports" / "metrics_day8.csv", index=False)

    # Save best model by PR-AUC
    best_name = metrics_df.iloc[0]["model"]
    name_to_model = {
        "Dummy Baseline": dummy,
        "LogReg Baseline": logreg,
        "RandomForest (best)": best_rf,
    }
    if best_xgb: name_to_model["XGBoost (best)"] = best_xgb

    best_model = name_to_model[best_name]
    joblib.dump(best_model, here / "models" / "best_model.pkl")

    print(f"[OK] Best model ({best_name}) saved to models/best_model.pkl")
    print("Metrics:\n", metrics_df)


if __name__ == "__main__":
    main()
