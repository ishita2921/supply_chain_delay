import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- Paths ---------------- #
TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"
MODEL_PATH_CANDIDATES = ["artifacts/best_model.pkl"]
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load Data ---------------- #
train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)

# ---------------- Detect Columns ---------------- #
LABEL_COL = "logistics_delay" if "logistics_delay" in train.columns else train.columns[-1]

# Only detect true ID columns; remove "humidity" from id detection
id_cols = [c for c in train.columns if ("id" in c.lower() or "date" in c.lower()) and c != "humidity"]
print("Detected id cols:", id_cols)

# Detect supplier column
supplier_col = None
for c in train.columns:
    if "supplier" in c.lower() or "vendor" in c.lower():
        supplier_col = c
        break
print("Detected supplier column:", supplier_col)

# ---------------- Load Model ---------------- #
model = None
pipeline_model = None
for p in MODEL_PATH_CANDIDATES:
    if os.path.exists(p):
        print("Loading model from", p)
        model = joblib.load(p)
        break

if model is None:
    print("No model found. Training RandomForest baseline.")
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    estimator = model
else:
    if isinstance(model, SKPipeline):
        pipeline_model = model
        estimator = list(model.named_steps.values())[-1]
    else:
        estimator = model
        pipeline_model = None

# ---------------- Prepare Data ---------------- #
def prepare_xy(df, drop_humidity=False):
    drop_cols = id_cols.copy()
    if drop_humidity and "humidity" in df.columns:
        drop_cols.append("humidity")
    if LABEL_COL in df.columns:
        drop_cols.append(LABEL_COL)
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[LABEL_COL].astype(int) if LABEL_COL in df.columns else None
    return X, y

# Drop humidity only if we are NOT using pipeline model
drop_humidity = pipeline_model is None
X_train, y_train = prepare_xy(train, drop_humidity=drop_humidity)
X_val, y_val = prepare_xy(val, drop_humidity=drop_humidity)
X_test, y_test = prepare_xy(test, drop_humidity=drop_humidity)

# ---------------- Preprocessing ---------------- #
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()

num_pipe = SKPipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = SKPipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop"
)

X_train_proc = preprocessor.fit_transform(X_train)
ohe_names = list(preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols)) if cat_cols else []
feature_names = list(num_cols) + ohe_names
X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)

X_test_proc = preprocessor.transform(X_test)
X_test_proc = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

# ---------------- Predictions ---------------- #
if pipeline_model is not None:
    probs = pipeline_model.predict_proba(X_test)
else:
    probs = estimator.predict_proba(X_test_proc)

# ---------------- SHAP Explainability ---------------- #
import shap

is_tree_model = hasattr(estimator, "feature_importances_") or estimator.__class__.__name__.lower().startswith(
    ("random","gradient","xgboost","lgbm","catboost","hist"))

if pipeline_model is not None:
    background = X_train.sample(min(200, len(X_train)), random_state=42)
    background_proc = preprocessor.transform(background)
    background_proc = pd.DataFrame(background_proc, columns=feature_names)

    if is_tree_model:
        try:
            explainer = shap.TreeExplainer(estimator, data=background_proc)
            shap_vals = explainer.shap_values(X_test_proc)
            shap_for_delay = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            base_values = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        except:
            explainer = shap.Explainer(pipeline_model.predict_proba, background_proc)
            shap_exp = explainer(X_test_proc)
            shap_for_delay = shap_exp.values[:,1,:] if shap_exp.values.ndim==3 else shap_exp.values
            base_values = shap_exp.base_values[:,1] if shap_exp.values.ndim==3 else shap_exp.base_values
    else:
        explainer = shap.Explainer(pipeline_model.predict_proba, background_proc)
        shap_exp = explainer(X_test_proc)
        shap_for_delay = shap_exp.values[:,1,:] if shap_exp.values.ndim==3 else shap_exp.values
        base_values = shap_exp.base_values[:,1] if shap_exp.values.ndim==3 else shap_exp.base_values

else:
    background = X_train_proc.sample(min(200, len(X_train_proc)), random_state=42)
    if is_tree_model:
        try:
            explainer = shap.TreeExplainer(estimator, data=background)
            shap_vals = explainer.shap_values(X_test_proc)
            shap_for_delay = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            base_values = explainer.expected_value[1] if isinstance(explainer.expected_value, (list,np.ndarray)) else explainer.expected_value
        except:
            explainer = shap.Explainer(estimator.predict_proba, background)
            shap_exp = explainer(X_test_proc)
            shap_for_delay = shap_exp.values[:,1,:] if shap_exp.values.ndim==3 else shap_exp.values
            base_values = shap_exp.base_values[:,1] if shap_exp.values.ndim==3 else shap_exp.base_values
    else:
        explainer = shap.Explainer(estimator.predict_proba, background)
        shap_exp = explainer(X_test_proc)
        shap_for_delay = shap_exp.values[:,1,:] if shap_exp.values.ndim==3 else shap_exp.values
        base_values = shap_exp.base_values[:,1] if shap_exp.values.ndim==3 else shap_exp.base_values

print("SHAP array shape (samples, features):", getattr(shap_for_delay, "shape", None))

# ---------------- Global Explainability ---------------- #
plt.figure()
shap.summary_plot(shap_for_delay, X_test_proc, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=200)
plt.close()

plt.figure()
shap.summary_plot(shap_for_delay, X_test_proc, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm.png"), dpi=200)
plt.close()
print("Saved global SHAP summary and beeswarm to", OUT_DIR)

# ---------------- Local Waterfall ---------------- #
pos_probs = probs[:,1]
sorted_idx = np.argsort(pos_probs)
example_idxs = [sorted_idx[-1], sorted_idx[len(sorted_idx)//2], sorted_idx[0]]
example_names = ["high", "med", "low"]

for name, idx in zip(example_names, example_idxs):
    plt.figure()
    try:
        if 'shap_exp' in locals():
            sample_exp = shap_exp[idx,1] if shap_exp.values.ndim==3 else shap_exp[idx]
            shap.plots.waterfall(sample_exp, show=False)
        else:
            single_vals = shap_for_delay[idx, :, 1]
            single_base = base_values[idx] if isinstance(base_values, np.ndarray) else base_values
            sample_exp = shap.Explanation(values=single_vals,
                                          base_values=single_base,
                                          data=X_test_proc.iloc[idx].values,
                                          feature_names=feature_names)
            shap.plots.waterfall(sample_exp, show=False)
    except:
        abs_contrib = np.abs(single_vals)
        top_i = np.argsort(abs_contrib)[-10:][::-1]
        feat = [feature_names[i] for i in top_i]
        vals = single_vals[top_i]
        plt.barh(feat[::-1], vals[::-1])
        plt.title(f"Top contributions (fallback) — {name} (idx {idx})")
        plt.xlabel("SHAP value for delay class")
    plt.tight_layout()
    outp = os.path.join(OUT_DIR, f"shap_waterfall_{name}_{X_test_proc.index[idx]}.png")
    plt.savefig(outp, dpi=200)
    plt.close()
    print("Saved local waterfall for", name, "->", outp)

# ---------------- Supplier-Level Slice ---------------- #
if supplier_col and supplier_col in test.columns:
    suppliers_series = test[supplier_col].fillna("missing")
    supplier_summary = []
    for sup in suppliers_series.unique():
        idxs = suppliers_series[suppliers_series==sup].index.tolist()
        if len(idxs)<5: continue
        pos = [list(X_test_proc.index).index(i) for i in idxs if i in X_test_proc.index]
        if len(pos)==0: continue
        mean_abs = np.mean(np.abs(shap_for_delay[pos,:]), axis=0)
        top_idx = np.argsort(mean_abs)[-8:][::-1]
        top_feats = [(feature_names[i], float(mean_abs[i])) for i in top_idx]
        supplier_summary.append({"supplier":sup, "n_shipments":len(pos), "top_features":top_feats})
        plt.figure()
        feats = [f for f,_ in top_feats][::-1]
        vals = [v for _,v in top_feats][::-1]
        plt.barh(feats, vals)
        plt.title(f"Top features increasing delay risk — supplier {sup}")
        plt.xlabel("mean |SHAP| (delay class)")
        plt.tight_layout()
        safe_name = str(sup).replace("/","_").replace(" ","_")
        plt.savefig(os.path.join(OUT_DIR, f"supplier_{safe_name}_top_features.png"), dpi=200)
        plt.close()
    sup_df = pd.DataFrame([{"supplier":r["supplier"], "n_shipments":r["n_shipments"],
                            "top_features":"; ".join([f"{f}:{v:.4f}" for f,v in r["top_features"]])} 
                           for r in supplier_summary])
    sup_df.to_csv(os.path.join(OUT_DIR,"supplier_shap_summary.csv"), index=False)
    print("Saved supplier SHAP summaries to", OUT_DIR)
else:
    print("No supplier column present in test set — skipped supplier slice.")

print("All SHAP deliverables saved under", OUT_DIR)
