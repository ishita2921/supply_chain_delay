# app/streamlit_app.py
# Day 11 — Monitoring Dashboard (robust, production-friendly)

import os
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dateutil.relativedelta import relativedelta

# ----------- Config -----------
st.set_page_config(page_title="Supply Chain Monitoring", page_icon="📦", layout="wide")

DB_TABLE = os.getenv("DB_TABLE", "shipments_fact")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------- Utilities -----------
def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _candidate_files() -> list:
    patterns = [
        "data/interim/*.parquet",
        "data/interim/*.csv",
        "data/processed/*.parquet",
        "data/processed/*.csv",
        "data/raw/*.parquet",
        "data/raw/*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names predictable, infer delay_flag/event_date/month, and
    add safe placeholders so dashboard charts render even if source lacks some fields.
    """

    df = df.copy()

    # normalize columns to snake_case lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # -------------------------
    # Supplier normalization
    # -------------------------
    for c in ["supplier", "vendor", "carrier", "shipper", "provider"]:
        if c in df.columns:
            df.rename(columns={c: "supplier"}, inplace=True)
            break
    if "supplier" not in df.columns:
        # placeholder so supplier chart renders during testing
        df["supplier"] = "DefaultSupplier"

    # -------------------------
    # Mode normalization
    # -------------------------
    for c in ["mode", "shipping_mode", "shipment_mode", "transport_mode"]:
        if c in df.columns:
            df.rename(columns={c: "mode"}, inplace=True)
            break
    if "mode" not in df.columns:
        df["mode"] = "Road"

    # -------------------------
    # Region normalization
    # -------------------------
    for c in ["region", "market", "zone", "area"]:
        if c in df.columns:
            df.rename(columns={c: "region"}, inplace=True)
            break
    if "region" not in df.columns:
        df["region"] = "UnknownRegion"

    # origin/dest regions fallback
    if "origin_region" not in df.columns:
        if "origin" in df.columns:
            df.rename(columns={"origin": "origin_region"}, inplace=True)
        else:
            df["origin_region"] = df["region"]
    if "dest_region" not in df.columns:
        if "destination" in df.columns:
            df.rename(columns={"destination": "dest_region"}, inplace=True)
        else:
            df["dest_region"] = df["region"]

    # -------------------------
    # Distance normalization
    # -------------------------
    if "distance_km" not in df.columns and "distance" in df.columns:
        df.rename(columns={"distance": "distance_km"}, inplace=True)
    if "distance_km" not in df.columns and "distance_miles" in df.columns:
        df["distance_km"] = df["distance_miles"].astype(float) * 1.60934

    # If still missing distance, create synthetic distances (so chart renders).
    # NOTE: this is synthetic — replace/join real distances for production.
    if "distance_km" not in df.columns:
        df["distance_km"] = np.random.randint(50, 2000, size=len(df))

    # Create distance buckets
    bins = [0, 200, 500, 1000, 2000, np.inf]
    labels = ["0-200", "200-500", "500-1000", "1000-2000", "2000+"]
    df["distance_bucket_km"] = pd.cut(df["distance_km"], bins=bins, labels=labels, right=False)

    # -------------------------
    # Event date / month
    # -------------------------
    # prefer common date columns
    date_col = None
    for c in ["event_date", "date", "timestamp", "ship_date", "shipment_date", "pickup_date", "delivery_date"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        df["event_date"] = _to_datetime(df[date_col])
    else:
        # fallback to 'today' if none present
        df["event_date"] = pd.Timestamp.today().normalize()

    # ensure event_date dtype
    df["event_date"] = _to_datetime(df["event_date"])

    # month bucket (first day of month)
    df["month"] = df["event_date"].dt.to_period("M").dt.to_timestamp()

    # -------------------------
    # Delay flag inference (robust)
    # -------------------------
    # 1) rename direct boolean-like columns to delay_flag
    found = False
    for col in ["delay_flag", "logistics_delay", "is_delayed", "delayed", "delay"]:
        if col in df.columns:
            df.rename(columns={col: "delay_flag"}, inplace=True)
            found = True
            break

    # 2) numeric delay columns
    if not found:
        for col in [
            "delivery_delay_days", "delay_days", "delay_hrs", "delay_hours",
            "delay_minutes", "lateness_days", "lateness_hours", "lateness"
        ]:
            if col in df.columns:
                df["delay_flag"] = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
                found = True
                break

    # 3) planned vs actual
    if not found:
        date_cols = df.columns.tolist()
        planned_candidates = [c for c in date_cols if "planned" in c and "date" in c]
        actual_candidates = [c for c in date_cols if ("actual" in c or "delivered" in c) and "date" in c]
        if planned_candidates and actual_candidates:
            p = planned_candidates[0]
            a = actual_candidates[0]
            df[p] = _to_datetime(df[p])
            df[a] = _to_datetime(df[a])
            df["delay_flag"] = (df[a] - df[p]).dt.total_seconds() > 0
            found = True

    # final safeguard: ensure column exists and is boolean
    if "delay_flag" not in df.columns:
        st.warning("⚠️ No delay column found — setting all delays to False (on-time).")
        df["delay_flag"] = False
    df["delay_flag"] = df["delay_flag"].astype(bool)

    # -------------------------
    # Small useful normalizations for display
    # -------------------------
    if "supplier" in df.columns:
        df["supplier"] = df["supplier"].astype(str)
    if "region" in df.columns:
        df["region"] = df["region"].astype(str)

    # Debugging: show transformed columns once (remove later if noisy)
    st.write("✅ Columns after standardization:", df.columns.tolist())
    st.write(df.head())

    return df


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, str]:
    """Load from DB if DATABASE_URL set, else from local data folder."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            query = f"SELECT * FROM {DB_TABLE} WHERE event_date >= CURRENT_DATE - INTERVAL '365 days'"
            df = pd.read_sql_query(text(query), engine)
            return _standardize_columns(df), f"db://{DB_TABLE}"
        except Exception as e:
            st.warning(f"DB load failed ({e}). Falling back to local files…")

    files = _candidate_files()
    if not files:
        st.error("No data files found under data/interim or data/processed. Place a CSV or Parquet file there.")
        st.stop()
    parquet = [f for f in files if f.endswith(".parquet")]
    csvs = [f for f in files if f.endswith(".csv")]
    src = parquet[0] if parquet else csvs[0]
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
    return _standardize_columns(df), src


# ----------- Small helpers -----------
def delay_rate(grp: pd.DataFrame) -> float:
    denom = max(len(grp), 1)
    return float(grp["delay_flag"].sum()) / denom


def kpi_tile(label: str, value: float, helptext: str = ""):
    st.metric(label, f"{value:.1%}", help=helptext)


def small_trendline(df_: pd.DataFrame, title: str):
    daily = df_.groupby(df_["event_date"].dt.date)["delay_flag"].mean().reset_index()
    fig = px.line(daily, x="event_date", y="delay_flag", title=title)
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# ----------- App start -----------
df, source = load_data()

st.sidebar.header("Filters")

# safe month values (ensure dtype)
if "month" not in df.columns or pd.isna(df["month"].min()):
    df["month"] = df["event_date"].dt.to_period("M").dt.to_timestamp()

min_month = df["month"].min()
max_month = df["month"].max()

# build a sensible default start (last 12 months) while guarding types
if pd.notnull(max_month) and isinstance(max_month, (pd.Timestamp, pd.DatetimeIndex, pd.Timestamp.__class__)):
    try:
        start_default = max_month - relativedelta(months=11)
    except Exception:
        start_default = min_month
else:
    start_default = min_month

# Month range slider (uses python datetimes for UI)
try:
    month_range = st.sidebar.slider(
        "Month range",
        min_value=min_month.to_pydatetime(),
        max_value=max_month.to_pydatetime(),
        value=(start_default.to_pydatetime(), max_month.to_pydatetime()),
        format="YYYY-MM",
    )
except Exception:
    # fallback to full range
    month_range = (min_month.to_pydatetime(), max_month.to_pydatetime())

mask_month = (df["month"] >= pd.Timestamp(month_range[0]).replace(day=1)) & (df["month"] <= pd.Timestamp(month_range[1]).replace(day=1))
df_f = df.loc[mask_month].copy()

# Supplier filter
suppliers = sorted(df_f["supplier"].dropna().astype(str).unique()) if "supplier" in df_f.columns else []
sel_suppliers = st.sidebar.multiselect("Supplier", suppliers, default=suppliers[:10] if suppliers else [])
if sel_suppliers and "supplier" in df_f.columns:
    df_f = df_f[df_f["supplier"].astype(str).isin(sel_suppliers)]

# Mode filter
modes = sorted(df_f["mode"].dropna().astype(str).unique()) if "mode" in df_f.columns else []
sel_modes = st.sidebar.multiselect("Mode", modes, default=modes)
if sel_modes and "mode" in df_f.columns:
    df_f = df_f[df_f["mode"].astype(str).isin(sel_modes)]

# Region filters
if "region" in df_f.columns:
    regions = sorted(df_f["region"].dropna().astype(str).unique())
    sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
    if sel_regions:
        df_f = df_f[df_f["region"].astype(str).isin(sel_regions)]
else:
    if "origin_region" in df_f.columns:
        oregs = sorted(df_f["origin_region"].dropna().astype(str).unique())
        sel_oregs = st.sidebar.multiselect("Origin Region", oregs, default=oregs)
        if sel_oregs:
            df_f = df_f[df_f["origin_region"].astype(str).isin(sel_oregs)]
    if "dest_region" in df_f.columns:
        dregs = sorted(df_f["dest_region"].dropna().astype(str).unique())
        sel_dregs = st.sidebar.multiselect("Destination Region", dregs, default=dregs)
        if sel_dregs:
            df_f = df_f[df_f["dest_region"].astype(str).isin(sel_dregs)]

st.sidebar.caption(f"Source: {source}")

# 90-day window for KPI sparkline
max_date = df_f["event_date"].max()
win_start = max_date - pd.Timedelta(days=89) if pd.notnull(max_date) else pd.Timestamp.today() - pd.Timedelta(days=89)
df_90 = df_f[df_f["event_date"] >= win_start]

# Tabs
TAB_MON, TAB_DATA, TAB_DEFS = st.tabs(["Monitoring", "Data", "Definitions"])

with TAB_MON:
    st.title("📦 Supply Chain Monitoring")
    st.caption("Industry-style operational dashboard for logistics delay performance.")

    # KPIs row
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        overall = delay_rate(df_f)
        kpi_tile("Overall Delay Rate", overall, helptext="Delayed shipments / total shipments in filter")
    with c2:
        shipments = len(df_f)
        st.metric("Shipments", f"{shipments:,}")
    with c3:
        delayed = int(df_f["delay_flag"].sum())
        st.metric("Delayed Shipments", f"{delayed:,}")
    with c4:
        small_trendline(df_90, "Last 90 days delay rate")

    st.divider()

    # Charts grid (2 x 2)
    g1, g2 = st.columns(2)

    # 1) Delay rate by supplier (bar)
    with g1:
        st.subheader("Delay rate by supplier")
        if "supplier" in df_f.columns and not df_f.empty:
            supplier_perf = df_f.groupby("supplier").apply(delay_rate).reset_index(name="delay_rate").sort_values("delay_rate", ascending=False)
            fig1 = px.bar(supplier_perf, x="supplier", y="delay_rate", title=None)
            fig1.update_layout(yaxis_tickformat=".0%", xaxis_title="Supplier", yaxis_title="Delay rate")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No supplier column found or no data after filters.")

    # 2) Monthly trend (line)
    with g2:
        st.subheader("Monthly delay rate trend")
        monthly = df_f.groupby("month")["delay_flag"].mean().reset_index().sort_values("month")
        if not monthly.empty:
            fig2 = px.line(monthly, x="month", y="delay_flag", markers=True)
            fig2.update_layout(yaxis_tickformat=".0%", xaxis_title="Month", yaxis_title="Delay rate")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No monthly data available for the current filter range.")

    g3, g4 = st.columns(2)

    # 3) Route/region heatmap
    with g3:
        st.subheader("Route / Region heatmap (delay rate)")
        def _heatmap_df(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
            if {"origin_region", "dest_region"}.issubset(frame.columns):
                pivot = frame.pivot_table(index="origin_region", columns="dest_region", values="delay_flag", aggfunc="mean")
                return pivot
            elif "region" in frame.columns:
                pivot = frame.pivot_table(index="region", columns="month", values="delay_flag", aggfunc="mean")
                return pivot
            elif "route" in frame.columns:
                tmp = frame.copy()
                parts = tmp["route"].astype(str).str.replace(" ", "")
                tmp["origin_region"] = parts.str.split("->|-", regex=True).str[0]
                tmp["dest_region"] = parts.str.split("->|-", regex=True).str[-1]
                pivot = tmp.pivot_table(index="origin_region", columns="dest_region", values="delay_flag", aggfunc="mean")
                return pivot
            return None

        hm = _heatmap_df(df_f)
        if hm is not None and not hm.empty:
            fig3 = px.imshow(hm, aspect="auto", origin="lower", labels=dict(color="Delay rate"))
            fig3.update_layout(coloraxis_colorbar_tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No suitable columns to build a route/region heatmap.")

    # 4) Distance bucket vs delay rate
    with g4:
        st.subheader("Distance bucket vs delay rate")
        if "distance_bucket_km" in df_f.columns:
            dist = df_f.groupby("distance_bucket_km")["delay_flag"].mean().reset_index()
            if pd.api.types.is_categorical_dtype(df_f["distance_bucket_km"]):
                dist = dist.sort_values("distance_bucket_km")
            fig4 = px.bar(dist, x="distance_bucket_km", y="delay_flag")
            fig4.update_layout(xaxis_title="Distance (km)", yaxis_title="Delay rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No distance data available to compute buckets.")

    st.divider()

    # Export buttons
    st.subheader("Exports")
    colx, coly = st.columns([1, 2])
    with colx:
        if st.button("Export current charts to reports/figures/"):
            errors = []
            try:
                import kaleido  # noqa: F401
                ok = True
            except Exception:
                ok = False
                errors.append("kaleido not installed. Run: pip install -U kaleido")

            if ok:
                try:
                    fig1.write_image(str(FIG_DIR / "delay_rate_by_supplier.png"), scale=2)
                except Exception as e:
                    errors.append(f"Supplier bar export failed: {e}")
                try:
                    fig2.write_image(str(FIG_DIR / "monthly_delay_rate_trend.png"), scale=2)
                except Exception as e:
                    errors.append(f"Monthly trend export failed: {e}")
                try:
                    fig3.write_image(str(FIG_DIR / "route_region_heatmap.png"), scale=2)
                except Exception as e:
                    errors.append(f"Heatmap export failed: {e}")
                try:
                    fig4.write_image(str(FIG_DIR / "distance_bucket_vs_delay.png"), scale=2)
                except Exception as e:
                    errors.append(f"Distance bucket export failed: {e}")

            if errors:
                st.warning("\n".join(errors))
            else:
                st.success("Charts exported to reports/figures/ ✅")

with TAB_DATA:
    st.subheader("Sample of filtered data")
    st.dataframe(df_f.head(100), use_container_width=True)
    st.caption("Tip: Use this tab to sanity-check that filters and columns look right.")

with TAB_DEFS:
    st.subheader("Business definitions & guardrails")
    st.markdown(
        """
- **Delay rate** = Delayed shipments / Total shipments in scope (after filters). A shipment is **delayed** if
  - an explicit `delay_flag` is true, **or**
  - numerical delay fields (e.g., `delivery_delay_days`, `delay_hours`, `delay_minutes`) are > 0, **or**
  - `actual_delivery_date` > `promised/planned_delivery_date`.
- **Monthly trend** uses the first day of the month as the bucket key (`month`).
- **Route/Region heatmap** shows mean delay rate by Origin × Destination region when available; otherwise Region × Month.
- **Distance buckets (km)**: `[0–200), [200–500), [500–1000), [1000–2000), 2000+`.
- **Filters precedence**: all charts reflect the current sidebar filters and month range.
- **Data Quality Checklist** (recommended for ops readiness): date nulls < 0.1%; supplier coverage > 95%; region taxonomy stable; controlled mode values; distance present for lanes.
"""
    )

# End of file
