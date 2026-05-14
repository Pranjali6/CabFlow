"""CabFlow -- NYC Yellow Taxi Demand Forecasting Dashboard."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Expose Streamlit secrets to libraries that read from os.environ
# (e.g. the anthropic SDK). Set in Streamlit Cloud's Secrets UI.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

from src.evaluation.metrics import compute_all_metrics
from src.models.base import BaseForecaster
from src.utils.helpers import get_feature_columns, load_config

st.set_page_config(page_title="CabFlow", page_icon="🚕", layout="wide")

# ---------------------------------------------------------------------------
# Dark mode toggle (Streamlit 1.30+ removed the built-in theme menu)
# ---------------------------------------------------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False, key="cabflow_dark_mode")
PLOTLY_TEMPLATE = "plotly_dark" if dark_mode else "plotly_white"

if dark_mode:
    st.markdown(
        """
        <style>
            .stApp, [data-testid="stAppViewContainer"] {
                background-color: #0e1117 !important;
                color: #fafafa !important;
            }
            [data-testid="stSidebar"] {
                background-color: #161a24 !important;
            }
            [data-testid="stHeader"] { background-color: #0e1117 !important; }
            h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown,
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
                color: #fafafa !important;
            }
            [data-testid="stMetric"] {
                background-color: #1a1f2e !important;
                border-radius: 8px;
                padding: 12px;
            }
            div[data-testid="stExpander"] details {
                background-color: #161a24 !important;
            }
            .stDataFrame, .stTable {
                background-color: #1a1f2e !important;
            }
            .stRadio > label, .stSelectbox > label, .stSlider > label {
                color: #fafafa !important;
            }
            /* Inline insight panel */
            div[style*="background:#f0f2f6"] {
                background: #1a1f2e !important;
                color: #fafafa !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🚕 CabFlow — NYC Yellow Taxi Demand Forecast")
st.caption("Hourly pickup forecasting across 263 NYC taxi zones | NYC TLC public data")


# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    config = load_config("config/config.yaml")
    for p in [
        Path("data/processed/trips_featured.parquet"),
        Path("data/processed/trips_hourly.parquet"),
    ]:
        if p.exists():
            df = pd.read_parquet(p)
            if "hour" in df.columns:
                df["hour"] = pd.to_datetime(df["hour"])
            return df, config
    st.error("No processed data found. Run preprocessing first: `python -m src.data.preprocessor`")
    st.stop()


@st.cache_data
def load_geojson():
    p = Path("data/raw/taxi_zones.geojson")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    return BaseForecaster.load(path) if path.exists() else None


df, config = load_data()
geo = load_geojson()

target_col = config["data"]["target_col"]
date_col = config["data"]["date_col"]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
available_models = {}
for _name in ["xgboost", "lightgbm"]:
    _m = load_model(f"models/{_name}.pkl")
    if _m:
        available_models[_name] = _m

if not available_models:
    st.sidebar.warning("No trained models found. Run `python -m src.pipeline.train_pipeline`.")

model_names = list(available_models.keys()) if available_models else ["none"]
selected_model = st.sidebar.selectbox("Model", model_names)

if "zone_name" in df.columns:
    zone_options = (
        df[["PULocationID", "zone_name", "Borough"]]
        .drop_duplicates()
        .sort_values(["Borough", "zone_name"])
    )
    zone_options["label"] = (
        zone_options["zone_name"] + " — " + zone_options["Borough"]
    )
    zone_label_to_id = dict(zip(zone_options["label"], zone_options["PULocationID"]))
    default_idx = 0
    for i, lbl in enumerate(zone_options["label"]):
        if "Times Sq" in lbl or "Midtown" in lbl:
            default_idx = i
            break
    selected_label = st.sidebar.selectbox(
        "Pickup Zone", zone_options["label"].tolist(), index=default_idx
    )
    selected_zone = int(zone_label_to_id[selected_label])
else:
    selected_zone = int(df["PULocationID"].iloc[0])
    selected_label = str(selected_zone)

horizon = st.sidebar.slider("Forecast Horizon (hours)", 6, 168, 24)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_zone_series(zid):
    return df[df["PULocationID"] == zid].sort_values(date_col).copy()


def _predict(model, data):
    try:
        feat_cols = get_feature_columns(data)
        preds = model.predict(data[feat_cols] if hasattr(model, "_feature_cols") else data)
        return np.maximum(np.round(preds).astype(int), 0)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def _insight(text):
    st.markdown(
        f"<div style='background:#f0f2f6;padding:12px 16px;border-radius:8px;"
        f"border-left:4px solid #f1c40f;margin:8px 0 20px 0;font-size:14px;color:#333'>"
        f"<b>💡 Insight:</b> {text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(
    [
        "City Overview",
        "Zone Forecast",
        "Zone Map",
        "Model Comparison",
        "Data Explorer",
        "Statistical Analysis",
        "Trends & Patterns",
        "Agent Insights",
    ]
)


# ========================= TAB 1: CITY OVERVIEW ============================
with tabs[0]:
    st.header("NYC Taxi Demand Overview")

    total_zones = df["PULocationID"].nunique()
    total_pickups = int(df[target_col].sum())
    avg_hourly = df[target_col].mean()
    peak_hour = df.groupby(df[date_col].dt.hour)[target_col].mean().idxmax()
    busiest_zone = df.groupby("zone_name")[target_col].sum().idxmax() if "zone_name" in df.columns else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active Zones", f"{total_zones:,}")
    k2.metric("Total Pickups", f"{total_pickups:,}")
    k3.metric("Avg Pickups / Hr / Zone", f"{avg_hourly:.1f}")
    k4.metric("Peak Hour (City-wide)", f"{int(peak_hour):02d}:00")
    k5.metric("Busiest Zone", busiest_zone[:18])

    # City-wide hourly trend
    city_hourly = df.groupby(date_col)[target_col].sum().reset_index()
    city_hourly["rolling_24h"] = city_hourly[target_col].rolling(24, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=city_hourly[date_col],
            y=city_hourly[target_col],
            mode="lines",
            name="Hourly Pickups (City)",
            line=dict(color="#f39c12", width=1),
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=city_hourly[date_col],
            y=city_hourly["rolling_24h"],
            mode="lines",
            name="24-hour rolling mean",
            line=dict(color="#c0392b", width=2.5),
        )
    )
    fig.update_layout(
        title="City-wide Hourly Pickup Volume",
        height=420,
        template=PLOTLY_TEMPLATE,
        xaxis_title="Time",
        yaxis_title="Pickups",
    )
    st.plotly_chart(fig, use_container_width=True)
    _insight(
        "The 24-hour rolling mean reveals the underlying trend (e.g. weekday vs weekend). Sharp dips at 3-5 AM and peaks around 6-8 PM are expected."
    )

    # Heatmap: hour-of-day vs day-of-week
    df_hod = df.copy()
    df_hod["hour_of_day"] = df_hod[date_col].dt.hour
    df_hod["dow"] = df_hod[date_col].dt.dayofweek
    heat = df_hod.groupby(["dow", "hour_of_day"])[target_col].mean().reset_index()
    heat_pivot = heat.pivot(index="dow", columns="hour_of_day", values=target_col)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = px.imshow(
        heat_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Pickups/Zone"),
        x=[f"{h:02d}" for h in heat_pivot.columns],
        y=[day_labels[i] for i in heat_pivot.index],
        color_continuous_scale="YlOrRd",
        title="Demand Heatmap: Day-of-Week × Hour-of-Day",
        aspect="auto",
    )
    fig.update_layout(height=380, template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)
    best_dow, best_hr = heat_pivot.stack().idxmax()
    _insight(
        f"Highest average per-zone demand: {day_labels[int(best_dow)]} at {int(best_hr):02d}:00. "
        f"Friday and Saturday late nights are the strongest demand windows for taxis."
    )

    # Borough breakdown
    if "Borough" in df.columns:
        bo = df.groupby("Borough")[target_col].sum().reset_index().sort_values(target_col, ascending=False)
        fig = px.bar(
            bo,
            x="Borough",
            y=target_col,
            color=target_col,
            color_continuous_scale="Oranges",
            title="Total Pickups by Borough",
        )
        fig.update_layout(height=350, template=PLOTLY_TEMPLATE, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        _insight(
            f"{bo.iloc[0]['Borough']} dominates yellow-taxi demand — yellow cabs are still allowed to street-hail there. "
            "Outer boroughs see less yellow activity (most rides are for-hire / app-based)."
        )

    # Top zones table
    if "zone_name" in df.columns:
        top_zones = (
            df.groupby(["zone_name", "Borough"])[target_col]
            .agg(["sum", "mean"])
            .reset_index()
            .sort_values("sum", ascending=False)
            .head(15)
        )
        top_zones.columns = ["Zone", "Borough", "Total Pickups", "Avg per Hour"]
        st.subheader("Top 15 Pickup Zones")
        st.dataframe(top_zones, use_container_width=True, hide_index=True)


# ========================= TAB 2: ZONE FORECAST ============================
with tabs[1]:
    st.header(f"Zone Forecast — {selected_label}")
    zdata = _get_zone_series(selected_zone)
    history_days = st.radio("History to display", [3, 7, 14, 30], index=1, horizontal=True)

    if available_models and selected_model in available_models and len(zdata) > horizon:
        model = available_models[selected_model]
        test_data = zdata.tail(horizon).copy()
        train_data = zdata.iloc[:-horizon]
        preds = _predict(model, test_data)

        if preds is not None:
            actuals = test_data[target_col].values.astype(int)
            residuals = actuals - preds

            train_preds = _predict(model, train_data.tail(168))
            if train_preds is not None:
                train_res_std = np.std(
                    train_data.tail(168)[target_col].values.astype(int) - train_preds
                )
            else:
                train_res_std = np.std(residuals)
            upper = preds + 1.96 * train_res_std
            lower = np.maximum(preds - 1.96 * train_res_std, 0)

            fig = go.Figure()
            hist = train_data.tail(history_days * 24)
            fig.add_trace(
                go.Scatter(
                    x=hist[date_col],
                    y=hist[target_col],
                    mode="lines",
                    name="Historical",
                    line=dict(color="#3498db"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_data[date_col],
                    y=actuals,
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#27ae60", width=2),
                    marker=dict(size=4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_data[date_col],
                    y=preds,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#e74c3c", dash="dash", width=2),
                    marker=dict(size=4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([test_data[date_col], test_data[date_col][::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(231,76,60,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% Interval",
                )
            )
            fig.update_layout(
                title=f"Pickup Forecast — Zone {selected_zone}",
                height=470,
                template=PLOTLY_TEMPLATE,
                xaxis_title="Time",
                yaxis_title="Pickups per Hour",
            )
            st.plotly_chart(fig, use_container_width=True)

            metrics = compute_all_metrics(actuals.astype(float), preds.astype(float))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE", f"{metrics['rmse']:.2f} pickups")
            m2.metric("MAE", f"{metrics['mae']:.2f} pickups")
            m3.metric("SMAPE", f"{metrics['smape']:.1f}%")
            exact = int(np.sum(preds == actuals))
            m4.metric("Exact Matches", f"{exact}/{len(actuals)}")

            # Residual bar chart
            colors = ["#27ae60" if r >= 0 else "#e74c3c" for r in residuals]
            fig = go.Figure(go.Bar(x=test_data[date_col], y=residuals, marker_color=colors))
            fig.update_layout(
                title="Hourly Forecast Errors (Actual − Predicted)",
                height=260,
                template=PLOTLY_TEMPLATE,
                xaxis_title="Hour",
                yaxis_title="Error",
            )
            st.plotly_chart(fig, use_container_width=True)

            _insight(
                f"Model RMSE is {metrics['rmse']:.1f} pickups/hour for this zone. "
                f"Use the 95% interval to set fleet rebalancing buffers — at peak hours a few extra cabs cover most uncertainty."
            )

            with st.expander("📋 Forecast detail table"):
                detail = pd.DataFrame(
                    {
                        "Hour": test_data[date_col].dt.strftime("%Y-%m-%d %H:00"),
                        "Actual": actuals,
                        "Predicted": preds,
                        "Error": residuals,
                        "|Error|": np.abs(residuals),
                    }
                )
                st.dataframe(detail, use_container_width=True, hide_index=True)
    else:
        if date_col in zdata.columns:
            fig = px.line(
                zdata.tail(7 * 24),
                x=date_col,
                y=target_col,
                title=f"Pickups: zone {selected_zone}",
            )
            fig.update_layout(template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        st.info("Train a model to see forecasts: `python -m src.pipeline.train_pipeline`")


# ========================= TAB 3: ZONE MAP =================================
with tabs[2]:
    st.header("NYC Zone Demand Map")
    if geo is None:
        st.warning("Zone GeoJSON not found. Run `python -m src.data.downloader` to fetch it.")
    else:
        slice_by = st.radio(
            "Color zones by",
            ["Total pickups", "Avg pickups/hour", "Peak hour demand"],
            horizontal=True,
        )
        if slice_by == "Total pickups":
            agg = df.groupby(["PULocationID", "zone_name", "Borough"])[target_col].sum().reset_index()
            value_label = "Total Pickups"
        elif slice_by == "Avg pickups/hour":
            agg = df.groupby(["PULocationID", "zone_name", "Borough"])[target_col].mean().reset_index()
            value_label = "Avg Pickups / Hr"
        else:
            agg = df.groupby(["PULocationID", "zone_name", "Borough"])[target_col].max().reset_index()
            value_label = "Peak Hour Pickups"

        agg = agg.rename(columns={target_col: value_label})
        # Detect the feature-id key in the geojson (LocationID vs location_id)
        sample = geo["features"][0]["properties"]
        id_key = "location_id" if "location_id" in sample else "LocationID"
        agg["PULocationID"] = agg["PULocationID"].astype(str)

        fig = px.choropleth_mapbox(
            agg,
            geojson=geo,
            locations="PULocationID",
            featureidkey=f"properties.{id_key}",
            color=value_label,
            color_continuous_scale="YlOrRd",
            mapbox_style="carto-positron",
            zoom=9.5,
            center={"lat": 40.74, "lon": -73.95},
            opacity=0.75,
            hover_data=["zone_name", "Borough"],
        )
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        _insight(
            "Hot zones cluster in Midtown, Times Square, Penn Station, and the JFK/LGA airport hexes. "
            "Outer boroughs are cold for yellow taxis — that's where Uber/Lyft fills the gap."
        )


# ========================= TAB 4: MODEL COMPARISON =========================
with tabs[3]:
    st.header("Model Comparison")
    if len(available_models) < 2:
        st.info("Train at least 2 models to compare.")
    else:
        zdata = _get_zone_series(selected_zone)
        if len(zdata) <= horizon:
            st.warning("Not enough history for this zone.")
        else:
            comp_data = zdata.tail(horizon)
            actuals = comp_data[target_col].values.astype(int)
            comp_results, all_preds = {}, {}
            for name, m in available_models.items():
                preds = _predict(m, comp_data)
                if preds is not None:
                    mets = compute_all_metrics(actuals.astype(float), preds.astype(float))
                    comp_results[name] = mets
                    all_preds[name] = preds

            if comp_results:
                comp_df = pd.DataFrame(comp_results).T.round(3)
                st.dataframe(
                    comp_df.style.highlight_min(axis=0, color="#d4edda"),
                    use_container_width=True,
                )
                best = comp_df["rmse"].idxmin()
                _insight(
                    f"**{best}** wins on RMSE ({comp_df.loc[best, 'rmse']:.3f}). "
                    "Lower RMSE = better hour-by-hour forecast accuracy."
                )

            if all_preds:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=comp_data[date_col],
                        y=actuals,
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="black", width=2),
                    )
                )
                cab_palette = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
                for i, (name, preds) in enumerate(all_preds.items()):
                    fig.add_trace(
                        go.Scatter(
                            x=comp_data[date_col],
                            y=preds,
                            mode="lines",
                            name=name,
                            line=dict(
                                color=cab_palette[i % len(cab_palette)],
                                dash="dash",
                                width=2,
                            ),
                        )
                    )
                fig.update_layout(
                    title="All Models vs Actual",
                    height=420,
                    template=PLOTLY_TEMPLATE,
                    yaxis_title="Pickups",
                )
                st.plotly_chart(fig, use_container_width=True)


# ========================= TAB 5: DATA EXPLORER ============================
with tabs[4]:
    st.header("Data Explorer")
    zdata = _get_zone_series(selected_zone)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            zdata,
            x=target_col,
            nbins=40,
            title="Pickup Count Distribution",
            color_discrete_sequence=["#3498db"],
        )
        fig.update_layout(height=320, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        zdata["hour_of_day"] = zdata[date_col].dt.hour
        hod = zdata.groupby("hour_of_day")[target_col].mean().reset_index()
        fig = px.line(
            hod,
            x="hour_of_day",
            y=target_col,
            title="Avg Pickups by Hour of Day",
            markers=True,
        )
        fig.update_layout(height=320, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        peak = int(hod.loc[hod[target_col].idxmax(), "hour_of_day"])
        _insight(
            f"Demand peaks at {peak:02d}:00 for this zone. "
            "Plan driver shifts to overlap with these peak hours."
        )

    # Weekend vs weekday
    zdata["is_weekend"] = (zdata[date_col].dt.dayofweek >= 5).astype(int)
    we_wd = zdata.assign(
        period=np.where(zdata["is_weekend"] == 1, "Weekend", "Weekday")
    )
    fig = px.box(
        we_wd,
        x="period",
        y=target_col,
        color="period",
        title="Pickups: Weekend vs Weekday",
        color_discrete_map={"Weekend": "#e74c3c", "Weekday": "#3498db"},
    )
    fig.update_layout(height=350, template=PLOTLY_TEMPLATE, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 6: STATISTICAL ANALYSIS =========================
with tabs[5]:
    st.header("Statistical Analysis")
    zdata = _get_zone_series(selected_zone)

    if target_col not in zdata.columns or zdata.empty:
        st.warning("No data.")
    else:
        ts = zdata.set_index(date_col)[target_col].dropna()
        # Stationarity
        st.subheader("Stationarity Test (Augmented Dickey-Fuller)")
        try:
            from statsmodels.tsa.stattools import adfuller

            adf = adfuller(ts.values, autolag="AIC")
            c1, c2, c3 = st.columns(3)
            c1.metric("ADF Statistic", f"{adf[0]:.4f}")
            c2.metric("p-value", f"{adf[1]:.4f}")
            c3.metric("Stationary?", "✅ Yes" if adf[1] < 0.05 else "❌ No")
        except Exception as e:
            st.error(f"ADF test failed: {e}")

        # ACF/PACF
        st.subheader("Autocorrelation (ACF / PACF)")
        n_obs = len(ts)
        max_lags = min(72, n_obs // 2 - 1)
        sig = 1.96 / np.sqrt(n_obs)
        try:
            from statsmodels.tsa.stattools import acf as acf_fn
            from statsmodels.tsa.stattools import pacf as pacf_fn

            c1, c2 = st.columns(2)
            if max_lags > 1:
                with c1:
                    acf_vals = acf_fn(ts.values, nlags=max_lags)
                    fig = go.Figure(
                        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color="#3498db")
                    )
                    fig.add_hline(y=sig, line_dash="dash", line_color="red")
                    fig.add_hline(y=-sig, line_dash="dash", line_color="red")
                    fig.update_layout(title="ACF", height=300, template=PLOTLY_TEMPLATE, xaxis_title="Lag (hours)")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    pacf_vals = pacf_fn(ts.values, nlags=max_lags, method="ywm")
                    fig = go.Figure(
                        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color="#e67e22")
                    )
                    fig.add_hline(y=sig, line_dash="dash", line_color="red")
                    fig.add_hline(y=-sig, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title="PACF", height=300, template=PLOTLY_TEMPLATE, xaxis_title="Lag (hours)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            _insight(
                "Strong autocorrelation at lag 24 = daily seasonality, at lag 168 = weekly seasonality. "
                "Our lag features explicitly capture both."
            )
        except Exception as e:
            st.error(f"ACF/PACF failed: {e}")

        # STL
        st.subheader("Seasonal Decomposition (STL, period=24)")
        try:
            from statsmodels.tsa.seasonal import STL

            if len(ts) >= 48:
                stl = STL(ts, period=24, robust=True).fit()
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=["Observed", "Trend", "Daily Seasonality", "Residual"],
                    vertical_spacing=0.06,
                )
                for i, (data, color) in enumerate(
                    [
                        (stl.observed, "#3498db"),
                        (stl.trend, "#c0392b"),
                        (stl.seasonal, "#27ae60"),
                        (stl.resid, "gray"),
                    ],
                    1,
                ):
                    fig.add_trace(
                        go.Scatter(x=ts.index, y=data, mode="lines", line=dict(color=color)),
                        row=i,
                        col=1,
                    )
                fig.update_layout(height=620, template=PLOTLY_TEMPLATE, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"STL failed: {e}")


# ===================== TAB 7: TRENDS & PATTERNS ============================
with tabs[6]:
    st.header("Trends & Patterns")
    zdata = _get_zone_series(selected_zone)

    if target_col not in zdata.columns:
        st.warning("No data.")
    else:
        # Hour-of-day x Day-of-week heatmap (zone)
        zdata["hour_of_day"] = zdata[date_col].dt.hour
        zdata["dow"] = zdata[date_col].dt.dayofweek
        heat = zdata.groupby(["dow", "hour_of_day"])[target_col].mean().reset_index()
        heat_pivot = heat.pivot(index="dow", columns="hour_of_day", values=target_col)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = px.imshow(
            heat_pivot,
            x=[f"{h:02d}" for h in heat_pivot.columns],
            y=[day_labels[i] for i in heat_pivot.index],
            color_continuous_scale="YlOrRd",
            title=f"Zone {selected_zone}: DoW × Hour Heatmap",
            aspect="auto",
        )
        fig.update_layout(height=380, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        # Borough comparison
        if "Borough" in df.columns:
            bo_hourly = (
                df.assign(hour_of_day=df[date_col].dt.hour)
                .groupby(["Borough", "hour_of_day"])[target_col]
                .mean()
                .reset_index()
            )
            fig = px.line(
                bo_hourly,
                x="hour_of_day",
                y=target_col,
                color="Borough",
                title="Average Hourly Pickups by Borough",
                markers=True,
            )
            fig.update_layout(height=400, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
            _insight(
                "Manhattan has a flatter, higher demand curve all day. "
                "Outer boroughs have sharper morning + evening commute spikes."
            )

        # MA crossover
        ts = zdata.set_index(date_col)[target_col].sort_index()
        ma_short = ts.rolling(24, min_periods=1).mean()
        ma_long = ts.rolling(168, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ts.index, y=ts.values, mode="lines", name="Pickups", line=dict(color="lightgray", width=1)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ma_short.index,
                y=ma_short.values,
                mode="lines",
                name="24h MA",
                line=dict(color="#3498db", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ma_long.index,
                y=ma_long.values,
                mode="lines",
                name="168h MA",
                line=dict(color="#c0392b", width=2),
            )
        )
        fig.update_layout(
            title="24-Hour vs 168-Hour (Weekly) Moving Averages",
            height=380,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 8: AGENT INSIGHTS ===============================
with tabs[7]:
    st.header("AI Agent Insights")
    st.markdown("Run the agent orchestrator to generate AI-powered insights using Claude.")

    if st.button("🤖 Run AI Analysis", type="primary"):
        with st.spinner("Running AI agents..."):
            try:
                from src.agents.orchestrator import AgentOrchestrator

                orch = AgentOrchestrator()
                zdata = _get_zone_series(selected_zone)
                results_data = {}
                if available_models:
                    test_data = zdata.tail(horizon)
                    for name, m in available_models.items():
                        preds = _predict(m, test_data)
                        if preds is not None:
                            mets = compute_all_metrics(
                                test_data[target_col].values.astype(float), preds.astype(float)
                            )
                            results_data[name] = mets
                results_df = pd.DataFrame(results_data).T if results_data else pd.DataFrame()
                output = orch.run_full_analysis(df=zdata, results_df=results_df)

                st.subheader("Data Quality Report")
                st.json(output.get("data_quality", {}).get("stats", {}))
                st.markdown(output.get("data_quality", {}).get("analysis", ""))

                st.subheader("Forecast Insights")
                st.markdown(output.get("insights", ""))

                st.subheader("Executive Report")
                st.markdown(output.get("report", ""))
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.info("Set ANTHROPIC_API_KEY to enable AI insights.")
