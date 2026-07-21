"""
 ML RETAIL FORECASTING SYSTEM
================================================================================
Dark "ops dashboard" themed retail sales forecasting tool.

Sidebar: Dashboard Overview / Historical Data nav, plus always-on
Model Configurations, Promotional Events and What-If Analysis controls.
Main panel: KPI cards, Sales Trend & ML Predictions chart, Demand by
Category, and ML Model Accuracy Comparison — matching the reference
dashboard theme.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

import data_utils
import feature_engineering
import models
import sql_utils

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Retail Forecasting System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME — dark navy background, cyan / purple / green accents
# ---------------------------------------------------------------------------
CYAN = "#22d3ee"
PURPLE = "#a78bfa"
GREEN = "#4ade80"
RED = "#f87171"
BG = "#0a0e1a"
CARD = "#121a2b"
BORDER = "#22304a"
TEXT = "#e2e8f0"
SUBTEXT = "#8b98ab"

st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    section[data-testid="stSidebar"] {{
        background: #0d1424;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}

    .proj-header {{
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: {TEXT};
        margin-bottom: 2px;
    }}
    .proj-header span {{ color: {CYAN}; }}
    .proj-sub {{
        color: {SUBTEXT};
        font-size: 0.85rem;
        margin-bottom: 18px;
    }}

    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 18px 10px 18px;
    }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {{
        color: {SUBTEXT} !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    /* High-contrast KPI values — always bright, regardless of base theme */
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {{
        color: #c7d2e0 !important;
        font-weight: 500 !important;
    }}

    /* Real bordered cards — all st.container(border=True, key="card_*") */
    div[data-testid="stVerticalBlock"][class*="st-key-card_"] {{
        background: {CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }}
    .section-title {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {SUBTEXT};
        margin-bottom: 10px;
    }}

    /* Sidebar nav buttons (secondary = inactive, primary = active page) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {{
        width: 100%;
        text-align: left !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: {SUBTEXT} !important;
        font-weight: 500 !important;
        justify-content: flex-start !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: #16213a !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] p {{
        text-align: left !important;
    }}

    div[data-testid="stExpander"] {{
        background: #0f1729;
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    div[data-testid="stExpander"] summary {{
        color: {TEXT} !important;
        font-size: 0.82rem;
        font-weight: 600;
    }}

    .stButton button[kind="primary"] {{
        background: linear-gradient(90deg, {CYAN}, {PURPLE});
        border: none;
        font-weight: 700;
        color: #0a0e1a;
    }}

    div[data-baseweb="select"] > div {{
        background-color: #0f1729 !important;
        border-color: {BORDER} !important;
    }}

    .stSlider [data-baseweb="slider"] {{
        color: {CYAN};
    }}

    hr {{ border-color: {BORDER}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #0f1729;
        border-radius: 8px 8px 0 0;
        padding: 6px 14px;
        color: {SUBTEXT};
    }}
    .stTabs [aria-selected="true"] {{
        color: {CYAN} !important;
        background-color: #16213a !important;
    }}
</style>
""", unsafe_allow_html=True)


def dark_layout(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, size=12),
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 20, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )
    return fig


# ---------------------------------------------------------------------------
# Forecast Period control — recompute the forecast for any horizon using the
# already-trained best model (no retraining, same forecasting functions).
# ---------------------------------------------------------------------------
HORIZON_LABELS = {7: "7 Day", 30: "30 Day", 90: "Quarterly"}
HORIZON_VALUES = {"7 Day": 7, "30 Day": 30, "Quarterly": 90}


def get_forecast_for_horizon(r, horizon):
    cache = st.session_state.setdefault("forecast_cache", {})
    cache_key = (r["best_model_name"], horizon)
    if cache_key in cache:
        return cache[cache_key]

    if r["best_is_ml"]:
        best_res = r["ml_results"][r["best_model_name"]]
        fc = models.recursive_ml_forecast(
            best_res["model"], r["featured_df"], r["date_col"], r["target_col"],
            best_res["feature_cols"], steps=horizon,
        )
        fc = fc.rename(columns={r["date_col"]: "Date"})
    else:
        ts = r["ts_results"][r["best_model_name"]]
        full_series = pd.concat([ts["train_series"], ts["test_series"]])
        refit, _ = models.train_arima(full_series, 1, order=(5, 1, 0))
        future_forecast = refit.forecast(steps=horizon)
        future_dates = pd.date_range(full_series.index.max() + timedelta(days=1), periods=horizon)
        fc = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast.values})

    cache[cache_key] = fc
    return fc



# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "raw_df": None,
    "detected": None,
    "results": None,
    "page": "Dashboard Overview",
    "show_upload_dialog": False,
    "learning_rate": 0.05,
    "batch_size": 20,
    "bonus_advanced": False,
    "promo_events": [],
    "price_adjustment": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------
def run_pipeline(raw_df, date_col, target_col, category_col, region_col, horizon,
                  status, learning_rate=0.05, n_estimators_scale=20):
    status.update(label="Validating dataset...")
    is_valid, message = data_utils.validate_dataset(raw_df)
    if not is_valid:
        raise ValueError(message)

    status.update(label="Cleaning data...")
    cleaned_df = data_utils.auto_clean(raw_df, date_col, target_col) if hasattr(data_utils, "auto_clean") \
        else data_utils.handle_missing_values(data_utils.remove_duplicates(raw_df), "median")

    status.update(label="Engineering features...")
    featured_df = feature_engineering.build_feature_set(cleaned_df, date_col, target_col)
    if len(featured_df) < 20:
        raise ValueError(
            "Not enough daily history after feature engineering to train a reliable model "
            "(need at least ~20 days). Please upload a dataset spanning a longer period."
        )

    status.update(label="Training machine learning models...")
    feature_cols = [c for c in featured_df.columns if c not in [date_col, target_col]]
    X, y = featured_df[feature_cols], featured_df[target_col]
    split_idx = int(len(featured_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = featured_df[date_col].iloc[split_idx:]

    ml_results = {}
    ml_model_names = ["Linear Regression", "Random Forest"] + (["XGBoost"] if models.XGBOOST_AVAILABLE else [])
    for name in ml_model_names:
        try:
            if name == "XGBoost" and models.XGBOOST_AVAILABLE:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=max(50, int(n_estimators_scale) * 10), max_depth=6,
                    learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.9, random_state=42,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = models.evaluate_predictions(y_test, preds)
            elif name == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=max(50, int(n_estimators_scale) * 10), max_depth=12,
                    random_state=42, n_jobs=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = models.evaluate_predictions(y_test, preds)
            else:
                model, preds, metrics = models.train_ml_model(name, X_train, y_train, X_test, y_test)
            ml_results[name] = {
                "model": model, "metrics": metrics, "preds": preds,
                "y_test": y_test.values, "dates": test_dates.values, "feature_cols": feature_cols,
            }
        except Exception:
            continue

    status.update(label="Training time series models...")
    ts_results = {}
    try:
        ts_series = cleaned_df.groupby(date_col, as_index=True)[target_col].sum().asfreq("D").interpolate()
        ts_split = int(len(ts_series) * 0.8)
        train_series, test_series = ts_series.iloc[:ts_split], ts_series.iloc[ts_split:]
        if len(test_series) >= 3:
            fitted, forecast = models.train_arima(train_series, len(test_series), order=(5, 1, 0))
            ts_results["ARIMA"] = {
                "fitted": fitted, "metrics": models.evaluate_predictions(test_series.values, forecast.values),
                "train_series": train_series, "test_series": test_series, "forecast": forecast,
            }
    except Exception:
        pass

    status.update(label="Evaluating models and selecting the best performer...")
    all_scores = {name: r["metrics"]["RMSE"] for name, r in ml_results.items()}
    all_scores.update({name: r["metrics"]["RMSE"] for name, r in ts_results.items()})
    if not all_scores:
        raise ValueError("No model could be trained successfully on this dataset.")
    best_model_name = min(all_scores, key=all_scores.get)
    best_is_ml = best_model_name in ml_results

    status.update(label=f"Forecasting the next {horizon} days...")
    if best_is_ml:
        best_res = ml_results[best_model_name]
        forecast_df = models.recursive_ml_forecast(
            best_res["model"], featured_df, date_col, target_col, best_res["feature_cols"], steps=horizon
        )
        forecast_df = forecast_df.rename(columns={date_col: "Date"})
    else:
        full_series = pd.concat([ts_results[best_model_name]["train_series"], ts_results[best_model_name]["test_series"]])
        refit, _ = models.train_arima(full_series, 1, order=(5, 1, 0))
        future_forecast = refit.forecast(steps=horizon)
        future_dates = pd.date_range(full_series.index.max() + timedelta(days=1), periods=horizon)
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast.values})

    status.update(label="Finalizing results...")
    return {
        "cleaned_df": cleaned_df, "featured_df": featured_df, "date_col": date_col, "target_col": target_col,
        "category_col": category_col, "region_col": region_col, "ml_results": ml_results, "ts_results": ts_results,
        "best_model_name": best_model_name, "best_is_ml": best_is_ml, "forecast_df": forecast_df, "horizon": horizon,
    }


# fallback if data_utils has no auto_clean (older util version safety net)
if not hasattr(data_utils, "auto_clean"):
    def _auto_clean(df, date_col, target_col):
        df = data_utils.remove_duplicates(df)
        df = data_utils.convert_to_datetime(df, date_col)
        df = df.dropna(subset=[date_col])
        df = data_utils.handle_missing_values(df, "median")
        return df.reset_index(drop=True)
    data_utils.auto_clean = _auto_clean


# ---------------------------------------------------------------------------
# UPLOAD DIALOG — mirrors "Upload New Dataset for Forecasting" reference
# ---------------------------------------------------------------------------
@st.dialog("Upload New Dataset for Forecasting")
def upload_dialog():
    st.caption("Select historical transactional data (SQL dump, CSV, Excel)")
    uploaded_file = st.file_uploader("Drag & drop CSV or Excel", type=["csv", "xlsx"], label_visibility="collapsed")

    map_mode = st.radio("Column mapping", ["Auto-map columns", "Manual configuration"], horizontal=True,
                         label_visibility="collapsed")

    date_col = target_col = category_col = region_col = None
    df_preview = None

    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") \
                else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            df_preview = None

    if df_preview is not None:
        detected = data_utils.auto_detect_columns(df_preview)
        st.success(f"Loaded {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns")

        if map_mode == "Manual configuration":
            all_cols = df_preview.columns.tolist()
            numeric_cols = df_preview.select_dtypes(include=np.number).columns.tolist()
            object_cols = [c for c in all_cols if c not in numeric_cols]
            date_col = st.selectbox("Date column", all_cols,
                                     index=all_cols.index(detected["date_col"]) if detected["date_col"] in all_cols else 0)
            target_col = st.selectbox("Sales / revenue column", numeric_cols if numeric_cols else all_cols,
                                       index=numeric_cols.index(detected["target_col"]) if detected["target_col"] in numeric_cols else 0)
            category_col = st.selectbox("Category column (optional)", ["None"] + object_cols)
            region_col = st.selectbox("Region column (optional)", ["None"] + object_cols)
            category_col = None if category_col == "None" else category_col
            region_col = None if region_col == "None" else region_col
        else:
            date_col, target_col = detected["date_col"], detected["target_col"]
            category_col, region_col = detected["category_col"], detected["region_col"]
            st.caption(f"Auto-mapped → Date: **{date_col}** · Sales: **{target_col}** · "
                       f"Category: **{category_col or '—'}** · Region: **{region_col or '—'}**")

        horizon_label = st.selectbox("Forecast period", ["Next 7 Days", "Next 30 Days", "Next 90 Days (Quarterly)"], index=1)
        horizon = {"Next 7 Days": 7, "Next 30 Days": 30, "Next 90 Days (Quarterly)": 90}[horizon_label]

        if st.button("Start Prediction Model", type="primary", use_container_width=True):
            st.session_state.raw_df = df_preview
            st.session_state.detected = {"date_col": date_col, "target_col": target_col,
                                          "category_col": category_col, "region_col": region_col}
            try:
                with st.status("Processing your data...", expanded=False) as status:
                    results = run_pipeline(
                        df_preview, date_col, target_col, category_col, region_col, horizon, status,
                        learning_rate=st.session_state.learning_rate, n_estimators_scale=st.session_state.batch_size,
                    )
                    status.update(label="Done!", state="complete")
                st.session_state.results = results
                st.session_state.forecast_cache = {}
                st.session_state.forecast_period_radio = HORIZON_LABELS.get(horizon, "30 Day")
                st.session_state.show_upload_dialog = False
                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong while processing your data: {e}")
    else:
        st.info("Upload a CSV or Excel file with at least a date column and a sales column to begin.")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div style="font-weight:800; font-size:1.05rem; letter-spacing:1px; '
                f'padding:6px 4px 18px 4px; color:{TEXT};">◆ ML FORECAST</div>', unsafe_allow_html=True)

    nav_items = ["Dashboard Overview", "Historical Data"]
    if st.session_state.results is not None:
        nav_items += ["Model Performance", "Forecast Detail", "Downloads"]

    for item in nav_items:
        is_active = st.session_state.page == item
        label = f"{'▸ ' if is_active else '   '}{item}"
        if st.button(label, key=f"nav_{item}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = item
            st.rerun()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    with st.expander("⚙️  Model Configurations", expanded=False):
        st.session_state.learning_rate = st.slider("Learning Rate", 0.01, 0.50, st.session_state.learning_rate, 0.01)
        st.session_state.batch_size = st.slider("Batch Size", 8, 128, st.session_state.batch_size, 4)
        st.session_state.bonus_advanced = st.toggle("Bonus as Advanced", value=st.session_state.bonus_advanced)
        st.caption("Applied to Random Forest / XGBoost on the next dataset run.")

    with st.expander("📣  Promotional Events", expanded=False):
        st.session_state.promo_events = st.multiselect(
            "Active promotions", ["Black Friday", "Christmas Sale", "Summer Sale", "New Year Clearance", "Flash Sale"],
            default=st.session_state.promo_events,
        )
        st.caption("Each active event adds ~8% demand uplift to the forecast preview.")

    with st.expander("💲  What-If Analysis", expanded=False):
        st.session_state.price_adjustment = st.slider("Price adjustment (%)", -100, 100, st.session_state.price_adjustment, 1)
        st.caption("Simple elasticity: −0.5 × price change applied to forecasted sales.")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("⬆️  Upload New Dataset", use_container_width=True, type="primary"):
        upload_dialog()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(f'<p class="proj-header">PROJECT: <span>RETAIL SALES FORECASTING SYSTEM</span></p>', unsafe_allow_html=True)
st.markdown('<p class="proj-sub">Upload transactional sales data to train, evaluate and forecast with '
            'ARIMA, Random Forest, XGBoost and Linear Regression models.</p>', unsafe_allow_html=True)


def promo_uplift():
    return 1 + 0.08 * len(st.session_state.promo_events)


def price_multiplier():
    return 1 - 0.5 * (st.session_state.price_adjustment / 100.0)


# ---------------------------------------------------------------------------
# NO DATA YET — empty state
# ---------------------------------------------------------------------------
if st.session_state.results is None:
    with st.container(border=True, key="card_empty_state"):
        st.markdown(f"""
        <div style="text-align:center; padding:44px 20px;">
            <div style="font-size:1.1rem; font-weight:600; color:{TEXT};">No dataset loaded yet</div>
            <div style="color:{SUBTEXT}; margin-top:6px;">Use <b>Upload New Dataset</b> in the sidebar to run the forecasting pipeline.</div>
        </div>
        """, unsafe_allow_html=True)
    if st.button("⬆️  Upload New Dataset for Forecasting", type="primary"):
        upload_dialog()

# ---------------------------------------------------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------------------------------------------------
elif st.session_state.page == "Dashboard Overview":
    r = st.session_state.results
    cleaned_df, date_col, target_col = r["cleaned_df"], r["date_col"], r["target_col"]

    # Selected forecast period drives everything below — default to the
    # horizon the model was trained/forecast at, but respond immediately
    # to a different Forecast Period selection.
    default_label = HORIZON_LABELS.get(r["horizon"], "30 Day")
    selected_label = st.session_state.get("forecast_period_radio", default_label)
    selected_horizon = HORIZON_VALUES[selected_label]

    f = get_forecast_for_horizon(r, selected_horizon).copy()
    uplift = promo_uplift() * price_multiplier()
    f["Forecast"] = f["Forecast"] * uplift

    best_metrics = (r["ml_results"].get(r["best_model_name"]) or r["ts_results"].get(r["best_model_name"]))["metrics"]
    mape = best_metrics.get("MAPE", np.nan)
    accuracy = max(0, 100 - mape) if not np.isnan(mape) else np.nan
    mad = np.nanmean(np.abs(np.array(f["Forecast"]) - np.nanmean(f["Forecast"])))

    last_actual = cleaned_df.sort_values(date_col)[target_col].tail(7).mean()
    first_forecast = f["Forecast"].head(7).mean()
    variance_pct = ((first_forecast - last_actual) / last_actual * 100) if last_actual else 0

    inventory_alerts = 0
    if r["category_col"] and r["category_col"] in cleaned_df.columns:
        cat_hist = cleaned_df.groupby(r["category_col"])[target_col].mean()
        cat_share = (cat_hist / cat_hist.sum()).fillna(0)
        predicted_demand = cat_share * f["Forecast"].sum()
        inventory_alerts = int((predicted_demand > cat_hist * 1.5).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Forecasted Sales", f"${f['Forecast'].sum():,.0f}", f"Next {selected_horizon} days")
    k2.metric("Forecast Accuracy (MAPE/MAD)", f"{accuracy:,.1f}%" if not np.isnan(accuracy) else "—", f"MAD {mad:,.1f}")
    k3.metric("Current vs Predicted Variance", f"{variance_pct:+.2f}%")
    k4.metric("Inventory Alert Count", f"{inventory_alerts}", "Alerts" if inventory_alerts else "Clear")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        with st.container(border=True, key="card_sales_trend"):
            st.markdown('<div class="section-title">Sales Trend &amp; ML Predictions</div>', unsafe_allow_html=True)

            hist = cleaned_df.groupby(date_col, as_index=False)[target_col].sum().sort_values(date_col)
            region_filter = st.session_state.get("region_select")
            if region_filter and region_filter != "All Regions" and r["region_col"] in cleaned_df.columns:
                hist = cleaned_df[cleaned_df[r["region_col"]] == region_filter].groupby(date_col, as_index=False)[target_col].sum().sort_values(date_col)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist[date_col], y=hist[target_col], mode="lines",
                                      name="Actual Sales", line=dict(color=CYAN, width=2)))
            band_upper = f["Forecast"] * 1.08
            band_lower = f["Forecast"] * 0.92
            fig.add_trace(go.Scatter(x=pd.concat([f["Date"], f["Date"][::-1]]),
                                      y=pd.concat([band_upper, band_lower[::-1]]),
                                      fill="toself", fillcolor="rgba(148,163,184,0.15)",
                                      line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=f["Date"], y=f["Forecast"], mode="lines+markers",
                                      name="ML Forecast Horizon", line=dict(color=PURPLE, width=2, dash="dot")))
            fig.add_vline(x=hist[date_col].max(), line_dash="dash", line_color=BORDER)
            st.plotly_chart(dark_layout(fig, height=380), use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            with st.container(border=True, key="card_demand_category"):
                st.markdown('<div class="section-title">Demand by Category</div>', unsafe_allow_html=True)
                if r["category_col"] and r["category_col"] in cleaned_df.columns:
                    cat_hist = cleaned_df.groupby(r["category_col"])[target_col].sum().sort_values(ascending=False).head(8)
                    cat_share = cat_hist / cat_hist.sum()
                    predicted = cat_share * f["Forecast"].sum()
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(y=cat_hist.index, x=predicted.values, name="Predicted Demand",
                                           orientation="h", marker_color=CYAN))
                    fig2.add_trace(go.Bar(y=cat_hist.index, x=cat_hist.values, name="Current Stock",
                                           orientation="h", marker_color=PURPLE))
                    fig2.update_layout(barmode="group")
                    st.plotly_chart(dark_layout(fig2, height=320), use_container_width=True)
                else:
                    st.info("No category column detected in this dataset.")

        with cB:
            with st.container(border=True, key="card_model_accuracy"):
                st.markdown('<div class="section-title">ML Model Accuracy Comparison</div>', unsafe_allow_html=True)
                names, accs, colors = [], [], []
                palette = [CYAN, PURPLE, GREEN, "#fbbf24"]
                for i, (name, res) in enumerate({**r["ml_results"], **r["ts_results"]}.items()):
                    m = res["metrics"].get("MAPE", np.nan)
                    names.append(name)
                    accs.append(max(0, 100 - m) if not np.isnan(m) else 0)
                    colors.append(palette[i % len(palette)])
                fig3 = go.Figure(go.Bar(x=names, y=accs, marker_color=colors, text=[f"{a:.1f}%" for a in accs], textposition="outside"))
                fig3.update_layout(yaxis_range=[0, 105])
                st.plotly_chart(dark_layout(fig3, height=320), use_container_width=True)

    with col_side:
        with st.container(border=True, key="card_forecast_period"):
            st.markdown('<div class="section-title">Forecast Period</div>', unsafe_allow_html=True)
            st.radio("Forecast Period", ["7 Day", "30 Day", "Quarterly"],
                     index=["7 Day", "30 Day", "Quarterly"].index(selected_label),
                     label_visibility="collapsed", key="forecast_period_radio")

        with st.container(border=True, key="card_region"):
            st.markdown('<div class="section-title">Region</div>', unsafe_allow_html=True)
            if r["region_col"] and r["region_col"] in cleaned_df.columns:
                options = ["All Regions"] + sorted(cleaned_df[r["region_col"]].dropna().unique().tolist())
                st.selectbox("Region", options, key="region_select", label_visibility="collapsed")
            else:
                st.caption("No region column detected.")

        with st.container(border=True, key="card_model"):
            st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
            model_keys = list({**r["ml_results"], **r["ts_results"]}.keys())
            st.selectbox("Model", model_keys, index=model_keys.index(r["best_model_name"]),
                         label_visibility="collapsed", key="model_select", disabled=True)
            st.caption("Best model auto-selected on RMSE.")

        with st.container(border=True, key="card_adjustments"):
            st.markdown('<div class="section-title">Active Adjustments</div>', unsafe_allow_html=True)
            st.caption(f"Promotions: {', '.join(st.session_state.promo_events) if st.session_state.promo_events else 'None'}")
            st.caption(f"Price adjustment: {st.session_state.price_adjustment:+d}%")


# ---------------------------------------------------------------------------
# HISTORICAL DATA
# ---------------------------------------------------------------------------
elif st.session_state.page == "Historical Data":
    r = st.session_state.results
    cleaned_df, date_col, target_col = r["cleaned_df"], r["date_col"], r["target_col"]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Historical Dataset</div>', unsafe_allow_html=True)
    st.dataframe(cleaned_df, use_container_width=True, height=460)
    st.caption(f"{cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns after automatic cleaning")
    st.download_button("Download Cleaned CSV", data=data_utils.to_csv_bytes(cleaned_df),
                        file_name="cleaned_dataset.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif st.session_state.page == "Model Performance":
    r = st.session_state.results
    rows = []
    for name, res in r["ml_results"].items():
        m = res["metrics"]
        rows.append({"Model": name, "Type": "Machine Learning", "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE (%)": m["MAPE"], "R2": m["R2"]})
    for name, res in r["ts_results"].items():
        m = res["metrics"]
        rows.append({"Model": name, "Type": "Time Series", "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE (%)": m["MAPE"], "R2": np.nan})
    perf_df = pd.DataFrame(rows).sort_values("RMSE")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.dataframe(perf_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.2f}", "R2": "{:.3f}"}),
                 use_container_width=True)
    st.success(f"Best performing model: **{r['best_model_name']}** (lowest RMSE on held-out data)")
    fig = go.Figure(go.Bar(x=perf_df["Model"], y=perf_df["RMSE"], marker_color=CYAN))
    st.plotly_chart(dark_layout(fig, height=380, title="RMSE by Model"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if r["best_is_ml"]:
        best_res = r["ml_results"][r["best_model_name"]]
        imp = models.feature_importance(best_res["model"], best_res["feature_cols"])
        if imp is not None:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">What Drives the Forecast — Feature Importance</div>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Bar(x=imp.head(10)["Importance"], y=imp.head(10)["Feature"], orientation="h", marker_color=PURPLE))
            fig2.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(dark_layout(fig2, height=380), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FORECAST DETAIL
# ---------------------------------------------------------------------------
elif st.session_state.page == "Forecast Detail":
    r = st.session_state.results
    f = r["forecast_df"].copy()
    f["Forecast"] = f["Forecast"] * promo_uplift() * price_multiplier()
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Forecast for the Next {r["horizon"]} Days (via {r["best_model_name"]})</div>', unsafe_allow_html=True)
    st.dataframe(f, use_container_width=True)
    fig = go.Figure(go.Scatter(x=f["Date"], y=f["Forecast"], mode="lines+markers", line=dict(color=PURPLE)))
    st.plotly_chart(dark_layout(fig, height=380), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast Total", f"{f['Forecast'].sum():,.0f}")
    c2.metric("Forecast Average", f"{f['Forecast'].mean():,.2f}")
    c3.metric("Forecast Peak Day", f"{f['Forecast'].max():,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DOWNLOADS
# ---------------------------------------------------------------------------
elif st.session_state.page == "Downloads":
    r = st.session_state.results
    cleaned_df = r["cleaned_df"]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export Center</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Cleaned Dataset**")
        st.download_button("Download Cleaned CSV", data=data_utils.to_csv_bytes(cleaned_df),
                            file_name="cleaned_dataset.csv", mime="text/csv")
    with c2:
        st.markdown("**Forecast Results**")
        st.download_button("Download Forecast CSV", data=data_utils.to_csv_bytes(r["forecast_df"]),
                            file_name="forecast_results.csv", mime="text/csv")
    with c3:
        st.markdown("**Model Predictions (Test Set)**")
        if r["best_is_ml"]:
            best_res = r["ml_results"][r["best_model_name"]]
            pred_df = pd.DataFrame({"Date": best_res["dates"], "Actual": best_res["y_test"], "Predicted": best_res["preds"]})
        else:
            ts_res = r["ts_results"][r["best_model_name"]]
            pred_df = pd.DataFrame({"Date": ts_res["test_series"].index, "Actual": ts_res["test_series"].values,
                                     "Predicted": ts_res["forecast"].values})
        st.download_button("Download Predictions CSV", data=data_utils.to_csv_bytes(pred_df),
                            file_name="predictions.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p style="color:{SUBTEXT}; font-size:0.75rem; text-align:center; margin-top:30px;">'
            f'Built with Streamlit · Scikit-learn · Statsmodels · Plotly</p>', unsafe_allow_html=True)
