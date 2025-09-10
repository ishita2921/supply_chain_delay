import os
import glob
import json
import joblib
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Try to import shap (optional)
try:
    import shap
except ImportError:
    shap = None

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Supply Chain Monitoring + Prediction",
    page_icon="📦",
    layout="wide"
)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")

def _candidate_files() -> list:
    patterns = [
        "data/interim/*.parquet", "data/interim/*.csv",
        "data/processed/*.parquet", "data/processed/*.csv",
        "data/raw/*.parquet", "data/raw/*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    df["event_date"] = _to_datetime(df["event_date"])
    df["month"] = df["event_date"].dt.month

    if "logistics_delay" in df.columns:
        df["delay_flag"] = df["logistics_delay"].astype(bool)
    else:
        df["delay_flag"] = False

    return df

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
    if "__" in name and "_" in name:
        base = name.split("__", 1)[-1]
        parts = base.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]} = {parts[1]}"
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

@st.cache_data(show_spinner=False)
def load_data():
    files = _candidate_files()
    if not files:
        st.error("No data found under data/interim or data/processed.")
        st.stop()
    src = files[0]
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
    return _standardize_columns(df), src

@st.cache_resource(show_spinner=False)
def load_model_and_pipeline():
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
    Xt = pipeline.transform(X_raw)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xt)[:, 1][0])
    return float(model.predict(Xt)[0])

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df, source = load_data()

# Sidebar filters
st.sidebar.header("Filters")
if "month" in df.columns:
    min_month = int(df["month"].min())
    max_month = int(df["month"].max())
    if min_month == max_month:
        month_range = (min_month, max_month)
    else:
        month_range = st.sidebar.slider(
            "Month range (1=Jan, 12=Dec)",
            min_value=min_month,
            max_value=max_month,
            value=(min_month, max_month),
        )
    mask = (df["month"] >= month_range[0]) & (df["month"] <= month_range[1])
    df_f = df.loc[mask].copy()
else:
    df_f = df.copy()

if "supplier" in df_f.columns:
    suppliers = sorted(df_f["supplier"].unique())
    sel_suppliers = st.sidebar.multiselect("Supplier", suppliers, default=suppliers[:10])
    if sel_suppliers:
        df_f = df_f[df_f["supplier"].isin(sel_suppliers)]

if "mode" in df_f.columns:
    modes = sorted(df_f["mode"].unique())
    sel_modes = st.sidebar.multiselect("Mode", modes, default=modes)
    if sel_modes:
        df_f = df_f[df_f["mode"].isin(sel_modes)]

if "region" in df_f.columns:
    regions = sorted(df_f["region"].unique())
    sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
    if sel_regions:
        df_f = df_f[df_f["region"].isin(sel_regions)]

st.sidebar.caption(f"Source: {source}")

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
TAB_MON, TAB_DATA, TAB_DEFS, TAB_PRED, TAB_SHAP = st.tabs(
    ["Monitoring", "Data", "Definitions", "Predict", "Global SHAP"]
)

# ---------------- Monitoring ----------------
with TAB_MON:
    st.title("📦 Supply Chain Monitoring")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Delay Rate", f"{df_f['delay_flag'].mean():.1%}")
    with c2:
        st.metric("Shipments", f"{len(df_f):,}")
    with c3:
        st.metric("Delayed", f"{df_f['delay_flag'].sum():,}")

    daily = df_f.groupby(df_f["event_date"].dt.date)["delay_flag"].mean().reset_index()
    fig = px.line(daily, x="event_date", y="delay_flag", title="Delay Rate Over Time")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delay rate by supplier")
    if "supplier" in df_f.columns:
        supplier_perf = (
            df_f.groupby("supplier")["delay_flag"].mean().reset_index().sort_values("delay_flag", ascending=False)
        )
        fig1 = px.bar(supplier_perf, x="supplier", y="delay_flag")
        fig1.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)

# ---------------- Data ----------------
with TAB_DATA:
    st.subheader("Sample data")
    st.dataframe(df_f.head(100), use_container_width=True)

# ---------------- Definitions ----------------
with TAB_DEFS:
    st.subheader("Definitions")
    st.markdown(
        """
        - **Delay rate** = Delayed shipments / Total shipments  
        - Delay flag inferred from `logistics_delay`  
        - Threshold for prediction loaded from `artifacts/threshold.json`  
        """
    )

# ---------------- Predict ----------------
with TAB_PRED:
    st.subheader("🔮 Predict shipment delay")
    try:
        model, pipeline, threshold, raw_features = load_model_and_pipeline()
        st.info(f"Model loaded. Decision threshold = {threshold:.2f}")

        with st.form("predict_form"):
            inputs = {}
            cols = st.columns(2)
            for i, feat in enumerate(raw_features):
                col = cols[i % 2]
                if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                    default = float(df[feat].median())
                    inputs[feat] = col.number_input(feat, value=default)
                else:
                    opts = sorted(df[feat].dropna().unique().astype(str))[:50] if feat in df.columns else []
                    inputs[feat] = col.selectbox(feat, options=opts) if opts else col.text_input(feat)
            submitted = st.form_submit_button("Predict")

        if submitted:
            X_input = pd.DataFrame([inputs])
            prob = predict_probability(model, pipeline, X_input)
            decision = "Delayed" if prob >= threshold else "On-time"
            st.metric("Delay probability", f"{prob:.1%}")
            st.success(f"Decision: {decision}")

            # ---------------- Local SHAP ----------------
            if shap is not None:
                try:
                    bg = joblib.load("artifacts/shap_background.pkl")
                    if not isinstance(bg, pd.DataFrame):
                        bg = pd.DataFrame(bg, columns=raw_features)

                    bg_trans = pipeline.transform(bg)
                    x_trans = pipeline.transform(X_input)

                    predict_fn = lambda X: model.predict_proba(X)[:, 1]
                    explainer = shap.Explainer(predict_fn, bg_trans)
                    shap_values = explainer(x_trans)

                    vals = np.array(shap_values.values)
                    if vals.ndim == 3:
                        contribs = vals[0, -1, :]
                    elif vals.ndim == 2:
                        contribs = vals[0, :]
                    else:
                        contribs = vals.ravel()

                    base_val = None
                    if hasattr(shap_values, "base_values") and shap_values.base_values is not None:
                        base_val = float(np.atleast_1d(shap_values.base_values)[0])
                    elif hasattr(explainer, "expected_value"):
                        try:
                            base_val = float(np.atleast_1d(explainer.expected_value)[0])
                        except Exception:
                            base_val = None
                    if base_val is None:
                        base_val = float(model.predict_proba(bg_trans)[:, 1].mean())

                    trans_names = get_transformed_feature_names(pipeline, raw_features)
                    friendly_names = [clean_feature_name(n) for n in trans_names]

                    k = min(5, len(contribs))
                    top_idx = np.argsort(np.abs(contribs))[::-1][:k]
                    top_features = [friendly_names[int(i)] for i in top_idx]
                    top_vals = [float(contribs[int(i)]) for i in top_idx]
                    top_inputs = [input_value_for_transformed_feature(f, X_input) for f in top_features]

                    top_df = pd.DataFrame({
                        "feature": top_features,
                        "input_value": top_inputs,
                        "shap_contribution": top_vals
                    })

                    st.subheader("Top SHAP Drivers (local)")
                    st.table(top_df)

                    # ✅ Fix: create matplotlib Figure for SHAP waterfall
                    expl = shap.Explanation(values=contribs, base_values=base_val,
                                             data=x_trans[0], feature_names=friendly_names)
                    plt.figure(figsize=(8, 6))
                    shap.plots.waterfall(expl, show=False)
                    st.pyplot(plt.gcf())

                except Exception:
                    st.error("SHAP explanation failed.")
                    st.text(traceback.format_exc())
            else:
                st.info("Install `shap` to see feature drivers.")

    except Exception:
        st.error("Model/pipeline not available or prediction failed.")
        st.text(traceback.format_exc())

# ---------------- Global SHAP ----------------
with TAB_SHAP:
    st.subheader("🌍 Global SHAP — Overall Drivers")
    if shap is not None:
        try:
            model, pipeline, _, raw_features = load_model_and_pipeline()
            bg = joblib.load("artifacts/shap_background.pkl")
            if not isinstance(bg, pd.DataFrame):
                bg = pd.DataFrame(bg, columns=raw_features)

            bg_trans = pipeline.transform(bg)
            predict_fn = lambda X: model.predict_proba(X)[:, 1]
            explainer = shap.Explainer(predict_fn, bg_trans)

            sample_trans = bg_trans[:200]
            shap_values_global = explainer(sample_trans)

            st.info("Beeswarm plot: each dot = one shipment, position = SHAP impact.")
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values_global, show=False)
            st.pyplot(plt.gcf())

        except Exception:
            st.error("Global SHAP computation failed.")
            st.text(traceback.format_exc())
    else:
        st.info("Install `shap` to enable global feature importance.")
