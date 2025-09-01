# models/day9_thresholding.py
import argparse
import json
import os
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Costs:
    fn: float  # cost of missed delay
    fp: float  # cost of false alert

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_model(path: str):
    return joblib.load(path)

def load_table(path: str, label_col: str, drop_cols):
    df = pd.read_csv(path)
    y = df[label_col].astype(int).values
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [label_col])
    return X, y, df

def get_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        # convert scores to [0,1] via logistic link if uncalibrated
        from scipy.special import expit
        p = expit(estimator.decision_function(X))
    else:
        raise ValueError("Estimator must implement predict_proba or decision_function.")
    return p

def maybe_calibrate(estimator, method, X_cal=None, y_cal=None):
    if method is None or method.lower() == "none":
        return estimator, None
    method = method.lower()
    if X_cal is None or y_cal is None:
        raise ValueError("Calibration requested but no calibration set provided. Pass --val-csv.")
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("Calibration method must be 'sigmoid' (Platt) or 'isotonic'.")
    calib = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calib.fit(X_cal, y_cal)
    return calib, method

def metrics_at_threshold(y_true, p, t):
    yhat = (p >= t).astype(int)
    if yhat.sum() == 0:
        prec = 0.0
    else:
        prec = precision_score(y_true, yhat, zero_division=0)
    rec = recall_score(y_true, yhat, zero_division=0)
    f1 = f1_score(y_true, yhat, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return prec, rec, f1, tp, fp, tn, fn, tpr, fpr

def scan_thresholds(y_true, p, step=0.001):
    thresholds = np.arange(0.0, 1.0 + step, step)
    rows = []
    for t in thresholds:
        prec, rec, f1, tp, fp, tn, fn, tpr, fpr = metrics_at_threshold(y_true, p, t)
        rows.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr,
        })
    return pd.DataFrame(rows)

def choose_by_precision_floor(df, min_precision=0.5):
    ok = df[df["precision"] >= min_precision]
    if len(ok) == 0:
        # Fallback: maximize F-beta with beta=5 (recall-heavy) to get a workable threshold.
        beta = 5.0
        df = df.copy()
        df["f_beta"] = (1 + beta**2) * (df["precision"] * df["recall"]) / (
            (beta**2) * df["precision"] + df["recall"] + 1e-12
        )
        cand = df.sort_values(["f_beta", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
        return float(cand["threshold"]), False  # False => did not meet precision floor
    # among those meeting the precision floor, pick max recall; break ties by lower threshold
    cand = ok.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
    return float(cand["threshold"]), True

def choose_by_cost(df, costs: Costs):
    # Expected misclassification cost per threshold:  C = FN*cost_fn + FP*cost_fp
    # We don't normalize by N, because argmin is unaffected by scaling.
    df = df.copy()
    df["expected_cost"] = df["fn"] * costs.fn + df["fp"] * costs.fp
    cand = df.sort_values(["expected_cost", "threshold"], ascending=[True, True]).iloc[0]
    return float(cand["threshold"]), float(cand["expected_cost"])

def plot_pr_vs_threshold(df, outpath):
    plt.figure(figsize=(8,5))
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_calibration(y_true, p_raw, p_cal, used_method, outpath):
    plt.figure(figsize=(6,6))
    CalibrationDisplay.from_predictions(y_true, p_raw, n_bins=10, name="Pre-calibration")
    if p_cal is not None:
        CalibrationDisplay.from_predictions(y_true, p_cal, n_bins=10, name=f"Calibrated ({used_method})")
    plt.title("Reliability (Calibration) Plot")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Day 9 — Thresholding + Cost Sensitivity + Calibration")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, default=None, help="For calibration (recommended).")
    parser.add_argument("--label-col", type=str, default="delayed")
    parser.add_argument("--id-cols", type=str, nargs="*", default=["shipment_id", "order_id", "date"])
    parser.add_argument("--min-precision", type=float, default=0.55, help="Business floor for precision.")
    parser.add_argument("--cost-fn", type=float, default=5.0, help="Cost weight for missed delay (FN).")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Cost weight for false alert (FP).")
    parser.add_argument("--calibration", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--out-dir", type=str, default="reports/day9")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    # Load model
    est = load_model(args.model)

    # Optional calibration set
    X_cal = y_cal = None
    if args.val_csv:
        X_cal, y_cal, _ = load_table(args.val_csv, args.label_col, args.id_cols)

    # Test/holdout set
    X_test, y_test, _ = load_table(args.test_csv, args.label_col, args.id_cols)

    # Pre-calibration probabilities (for reliability comparison)
    p_raw = get_proba(est, X_test)

    # Calibrate if requested
    used_method = None
    if args.calibration != "none":
        est_cal, used_method = maybe_calibrate(est, args.calibration, X_cal, y_cal)
        p_cal = get_proba(est_cal, X_test)
        brier_raw = brier_score_loss(y_test, p_raw)
        brier_cal = brier_score_loss(y_test, p_cal)
        p_final = p_cal
    else:
        est_cal = est
        p_cal = None
        brier_raw = brier_score_loss(y_test, p_raw)
        brier_cal = None
        p_final = p_raw

    # Scan thresholds
    df_thr = scan_thresholds(y_test, p_final, step=0.001)

    # Business choice: maximize recall subject to precision floor
    t_business, met_floor = choose_by_precision_floor(df_thr, args.min_precision)

    # Cost-minimizing choice
    costs = Costs(fn=args.cost_fn, fp=args.cost_fp)
    t_cost, min_expected_cost = choose_by_cost(df_thr, costs)

    # Build plots
    plot_pr_vs_threshold(df_thr, os.path.join(args.out_dir, "precision_recall_vs_threshold.png"))
    plot_calibration(y_test, p_raw, p_cal, used_method, os.path.join(args.out_dir, "calibration_plot.png"))

    # Persist threshold metrics table
    df_thr.to_csv(os.path.join(args.out_dir, "threshold_metrics.csv"), index=False)

    # Snapshot for the two recommended thresholds
    def snapshot_at(t):
        row = df_thr.iloc[(df_thr["threshold"]-t).abs().argsort()[:1]].iloc[0]
        cm = {"tp": int(row["tp"]), "fp": int(row["fp"]), "tn": int(row["tn"]), "fn": int(row["fn"])}
        return {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "confusion_matrix": cm
        }

    summary = {
        "calibration": {
            "used": used_method if used_method else "none",
            "brier_pre": float(brier_raw),
            "brier_post": float(brier_cal) if brier_cal is not None else None
        },
        "business_threshold": {
            "min_precision": args.min_precision,
            "met_precision_floor": bool(met_floor),
            **snapshot_at(t_business)
        },
        "cost_threshold": {
            "costs": {"FN": args.cost_fn, "FP": args.cost_fp},
            "expected_cost_at_optimum": float(min_expected_cost),
            **snapshot_at(t_cost)
        },
        "artifacts": {
            "precision_recall_vs_threshold": os.path.join(args.out_dir, "precision_recall_vs_threshold.png"),
            "calibration_plot": os.path.join(args.out_dir, "calibration_plot.png"),
            "threshold_metrics_csv": os.path.join(args.out_dir, "threshold_metrics.csv")
        }
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Findings.md snippet (print to console for copy-paste)
    print("\n================ Copy below into findings.md (Day 9) ================\n")
    print(f"""## Day 9 — Business-Oriented Threshold & Cost Sensitivity

**Objective.** Pick a decision threshold that maximizes **Recall** subject to a **Precision floor** and document our cost assumptions. Also verify and improve **probability calibration**.

### Calibration
- Method: **{summary['calibration']['used']}**
- Brier score (lower is better): pre = **{summary['calibration']['brier_pre']:.4f}**{'' if brier_cal is None else f', post = **{summary["calibration"]["brier_post"]:.4f}**'}
- Reliability plot: see `{summary['artifacts']['calibration_plot']}`.

### Business-Oriented Threshold (primary)
- Precision floor: **≥ {args.min_precision:.2f}**
- Selected threshold: **p* = {summary['business_threshold']['threshold']:.3f}** (floor met: **{summary['business_threshold']['met_precision_floor']}**)
- Metrics @ p*: Precision = **{summary['business_threshold']['precision']:.3f}**, Recall = **{summary['business_threshold']['recall']:.3f}**, F1 = **{summary['business_threshold']['f1']:.3f}**
- Confusion Matrix @ p*: TP={summary['business_threshold']['confusion_matrix']['tp']}, FP={summary['business_threshold']['confusion_matrix']['fp']}, TN={summary['business_threshold']['confusion_matrix']['tn']}, FN={summary['business_threshold']['confusion_matrix']['fn']}

**Rationale.** We maximize the capture of true delays while enforcing a minimum alert quality (Precision floor) to control operational noise.

### Cost-Sensitive Threshold (secondary, for reference)
- Cost matrix (unitless weights): **FN = {args.cost_fn:.2f}**, **FP = {args.cost_fp:.2f}**
- Selected threshold: **t_cost = {summary['cost_threshold']['threshold']:.3f}**
- Expected misclassification cost @ t_cost: **{summary['cost_threshold']['expected_cost_at_optimum']:.2f}**
- Metrics @ t_cost: Precision = **{summary['cost_threshold']['precision']:.3f}**, Recall = **{summary['cost_threshold']['recall']:.3f}**

**Interpretation.** We assume a missed delay (FN) is ~{args.cost_fn/args.cost_fp:.1f}× more costly than a false alert (FP). If business priorities shift (e.g., expediting budget tightens), adjust these weights and recompute.

### Plots & Artifacts
- Precision/Recall vs Threshold: `{summary['artifacts']['precision_recall_vs_threshold']}`
- Calibration (reliability) plot: `{summary['artifacts']['calibration_plot']}`
- Full per-threshold table: `{summary['artifacts']['threshold_metrics_csv']}`

""")
    print("====================================================================\n")

if __name__ == "__main__":
    main()
