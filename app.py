"""
Albania CPI Forecasting — Premium Apple-inspired UI
"""

import warnings, logging, itertools, os
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from xgboost import XGBRegressor
from typing import Dict, Optional
import io

np.random.seed(42)

st.set_page_config(
    page_title="Albania CPI Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 0 1rem !important;  /* ADDED horizontal padding instead of 0 */
    max-width: 100% !important; 
    overflow-x: hidden !important; 
}
/* ══════════════════════════════════════════════
   HERO  — full-width cinematic header
══════════════════════════════════════════════ */
.hero-wrap {
    background: #000;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 40%, rgba(0,180,255,.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 60%, rgba(120,40,255,.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 100%, rgba(0,200,180,.08) 0%, transparent 70%);
    padding: 2rem 1.5rem 2rem;  /* REDUCED from 4.5rem 2rem 3.5rem */
    position: relative; 
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,.07);
    box-sizing: border-box;
    width: 100%;
    max-width: 100%;  /* ADDED */
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-eyebrow {
    font-size: .72rem; font-weight: 600; letter-spacing: .18em;
    text-transform: uppercase; color: #00c8ff; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: .6rem;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 24px; height: 1px; background: #00c8ff;
}
.hero-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.8rem;      /* REDUCED from 4.5rem */
    font-weight: 400; 
    line-height: 1.1;       /* Tightened from 1.05 */
    color: #fff; 
    margin: 0 0 0.75rem;    /* REDUCED from 1rem */
    letter-spacing: -.02em;
}
.hero-title em { font-style: italic; color: #00c8ff; }
.hero-sub {
    font-size: 1.05rem;      /* ← REMOVE this line */
    font-weight: 300; 
    color: rgba(255,255,255,.5);
    max-width: 560px;        /* ← REMOVE this line */
    line-height: 1.6;        /* ← REMOVE this line */
    margin-bottom: 2rem;     /* ← REMOVE this line */
}
.hero-stats {
    display: flex; 
    gap: 3rem;               /* ← REMOVE this line */
    margin-top: 2rem;        /* ← REMOVE this line */
    border-top: 1px solid rgba(255,255,255,.08); 
    padding-top: 2rem;       /* ← REMOVE this line */
}
.hero-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 600; color: #fff;
}
.hero-stat-label {
    font-size: .72rem; color: rgba(255,255,255,.35);
    text-transform: uppercase; letter-spacing: .1em; margin-top: .2rem;
}
.hero-badges { 
    display: flex; 
    flex-wrap: wrap; 
    gap: .5rem;              /* ← REMOVE this line */
    margin-top: 1.8rem;      /* ← REMOVE this line */
}
.hbadge {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
    color: rgba(255,255,255,.7); font-size: .7rem; font-weight: 500;
    padding: .3rem .85rem; border-radius: 999px;
    backdrop-filter: blur(10px);
    transition: all .2s;
}

/* ══════════════════════════════════════════════
   CONTENT AREA
══════════════════════════════════════════════ */
.content-wrap { 
    padding: 1rem 1.5rem;  /* REDUCED from 2rem */
    max-width: 1400px;     /* ADDED constraint instead of 100% */
    margin: 0 auto;        /* Center instead of full width */
    box-sizing: border-box;
    width: 100%;
}
/* ══════════════════════════════════════════════
   CONTROL PANEL
══════════════════════════════════════════════ */
.ctrl-wrap {
    background: #fafafa;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 1rem 1.25rem;  /* REDUCED from 1.4rem 1.8rem 1rem */
    margin-bottom: 1.5rem;  /* REDUCED from 2rem */
    width: 100%;
    max-width: 100%;        /* ADDED */
    box-sizing: border-box;
    overflow: hidden;       /* ADDED to contain children */
}
.ctrl-label {
    font-size: .65rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #8896a5; margin-bottom: 1rem;
}

/* ══════════════════════════════════════════════
   SECTION HEADINGS
══════════════════════════════════════════════ */
.sec-head {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem; font-weight: 400; color: #0a0f1e;
    margin: 2rem 0 .8rem; letter-spacing: -.01em;
}
.sec-head em { font-style: italic; color: #00c8ff; }
.sec-divider {
    height: 1px; background: #e8ecf0; margin-bottom: 1.2rem;
}

/* ══════════════════════════════════════════════
   METRIC STRIP
══════════════════════════════════════════════ */
.metric-strip {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 1px; background: #e8ecf0;
    border: 1px solid #e8ecf0; border-radius: 12px;
    overflow: hidden; margin-bottom: 1.5rem;
}
.metric-cell {
    background: #fff; 
    padding: 0.8rem 1rem;   /* REDUCED from 1.1rem 1.4rem */
}

.metric-cell .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;     /* REDUCED from 1.55rem */
    font-weight: 600; 
    color: #0a0f1e;
}
.metric-cell .lbl {
    font-size: .68rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #8896a5; margin-bottom: .2rem;
}
.metric-cell .delta { font-size: .8rem; color: #16a34a; margin-top: .15rem; }

/* ══════════════════════════════════════════════
   INFO / WARN CARDS
══════════════════════════════════════════════ */
.info-card {
    background: #f0f9ff; border-left: 3px solid #00c8ff;
    padding: .8rem 1.1rem; border-radius: 0 8px 8px 0;
    font-size: .84rem; margin: .8rem 0; color: #1a2340; line-height: 1.5;
}
.warn-card {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    padding: .8rem 1.1rem; border-radius: 0 8px 8px 0;
    font-size: .84rem; margin: .8rem 0; color: #78350f; line-height: 1.5;
}

/* ══════════════════════════════════════════════
   WINNER TAG
══════════════════════════════════════════════ */
.winner-tag {
    background: #0a0f1e; color: #00c8ff;
    font-family: 'JetBrains Mono', monospace; font-size: .65rem; font-weight: 600;
    padding: .15rem .65rem; border-radius: 999px; margin-left: .5rem;
    vertical-align: middle;
}

/* ══════════════════════════════════════════════
   TRAIN BUTTON
══════════════════════════════════════════════ */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #1a2f50 100%) !important;
    color: #fff !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    font-size: .9rem !important; letter-spacing: .02em !important;
    padding: .65rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,.2) !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.3) !important;
}

/* ══════════════════════════════════════════════
   TABS
══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #e8ecf0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: .82rem; font-weight: 500; color: #8896a5;
    padding: .6rem 1.2rem; border-radius: 0;
    background: transparent; border: none;
}
.stTabs [aria-selected="true"] {
    color: #0a0f1e !important; font-weight: 600 !important;
    border-bottom: 2px solid #00c8ff !important;
}
</style>
""", unsafe_allow_html=True)

# ── colour palette ─────────────────────────────────────────────────────────
COLORS = {
    "sarima": "#ef4444", "boosted_sarima": "#f97316",
    "prophet": "#a855f7", "prophet_boost": "#00c8ff",
}
BASE = dict(
    font_family="Inter, -apple-system, sans-serif",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(255,255,255,.95)", bordercolor="#e8ecf0",
                borderwidth=1, font=dict(size=11)),
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_wide_to_long(df_wide):
    dates = df_wide.iloc[3, 1:].dropna().values
    date_list = []
    for d in dates:
        try: date_list.append(pd.to_datetime(d, format="%Y-%m"))
        except: date_list.append(pd.NaT)
    rows = []
    for idx in range(4, len(df_wide)):
        cell = df_wide.iloc[idx, 0]
        if pd.isna(cell) or str(cell) == "nan": continue
        cat = str(cell)
        code = cat.split(" ")[0] if " " in cat else cat
        name = cat[len(code):].strip() if " " in cat else cat
        for dt, v in zip(date_list, df_wide.iloc[idx, 1:len(dates)+1].values):
            if pd.notna(dt) and pd.notna(v):
                try: rows.append({"Date": dt, "Category": name, "Category_Code": code, "CPI": float(v)})
                except: pass
    return pd.DataFrame(rows).sort_values(["Category_Code","Date"]).reset_index(drop=True)

def get_series(df_long, code="000000"):
    return df_long[df_long["Category_Code"]==code][["Date","CPI"]].sort_values("Date").reset_index(drop=True)

def chronological_split(df, ratio=0.8):
    k = int(len(df)*ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def generate_future_dates(last_date, periods=36):
    return pd.DataFrame({"Date": pd.date_range(
        start=last_date+pd.DateOffset(months=1), periods=periods, freq="MS")})

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAForecaster:
    def __init__(self, order=(0,1,0), seasonal_order=(1,1,2,12)):
        self.order = order; self.seasonal_order = seasonal_order
        self.fitted_model = self.fitted_values = self.residuals = self._y = None

    def fit(self, df):
        y = df["CPI"].values
        self.fitted_model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        self.fitted_values = self.fitted_model.fittedvalues
        self.residuals = self.fitted_model.resid
        self._y = y
        return self

    def predict(self, steps):
        m2 = SARIMAX(self._y, order=self.order, seasonal_order=self.seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return np.array(m2.forecast(steps=steps))

    @property
    def aic(self): return self.fitted_model.aic


class BoostedSARIMA:
    def __init__(self, sarima_order=(1,1,1), sarima_seasonal=(0,1,1,12), xgb_params=None):
        self.sarima = SARIMAForecaster(order=sarima_order, seasonal_order=sarima_seasonal)
        self.xgb = XGBRegressor(**(xgb_params or {
            "n_estimators":100,"max_depth":5,"learning_rate":.1,
            "subsample":.8,"colsample_bytree":.8,"random_state":42,"objective":"reg:squarederror"}))
        self.feature_cols = None

    def _feats(self, df):
        d = df.copy()
        d["year"]=d["Date"].dt.year; d["month"]=d["Date"].dt.month
        d["quarter"]=d["Date"].dt.quarter; d["doy"]=d["Date"].dt.dayofyear
        d["m_sin"]=np.sin(2*np.pi*d["month"]/12); d["m_cos"]=np.cos(2*np.pi*d["month"]/12)
        d["t"]=np.arange(len(d))
        if "CPI" in d.columns:
            d["lag1"]=d["CPI"].shift(1); d["lag2"]=d["CPI"].shift(2)
            d["rm3"]=d["CPI"].shift(1).rolling(3).mean()
        return d

    def fit(self, df):
        self.sarima.fit(df)
        resid = df["CPI"].values - self.sarima.fitted_values
        f = self._feats(df); f["resid"] = resid
        fc = ["year","month","quarter","doy","m_sin","m_cos","t","lag1","lag2","rm3"]
        cl = f.dropna(); self.xgb.fit(cl[fc], cl["resid"]); self.feature_cols = fc
        return self

    def predict(self, df_future):
        sp = self.sarima.predict(len(df_future))
        f = self._feats(df_future)
        for col in ["lag1","lag2","rm3"]:
            if col in f.columns:
                f[col] = f[col].fillna(float(np.mean(sp)))
        return sp + self.xgb.predict(f[self.feature_cols])

    def feature_importances(self):
        return dict(zip(self.feature_cols, self.xgb.feature_importances_))


class ProphetForecaster:
    def __init__(self, **kw):
        self.model = Prophet(**kw, interval_width=.95); self.fitted = None

    def fit(self, df):
        self.fitted = self.model.fit(df.rename(columns={"Date":"ds","CPI":"y"}))
        return self

    def predict(self, df_future):
        fc = self.fitted.predict(df_future.rename(columns={"Date":"ds"}))
        return {"forecast":fc["yhat"].values,"lower":fc["yhat_lower"].values,"upper":fc["yhat_upper"].values}


class ProphetBoostForecaster:
    def __init__(self, prophet_params=None, xgb_params=None):
        pp = {"growth":"linear","yearly_seasonality":True,"weekly_seasonality":False,
              "daily_seasonality":False,"n_changepoints":25,"changepoint_range":.8,
              "changepoint_prior_scale":.05,"seasonality_prior_scale":10.}
        xp = {"n_estimators":150,"max_depth":6,"learning_rate":.05,"subsample":.8,
              "colsample_bytree":.8,"random_state":42,"objective":"reg:squarederror",
              "reg_alpha":.1,"reg_lambda":1.}
        self.prophet = ProphetForecaster(**{**pp, **(prophet_params or {})})
        self.xgb = XGBRegressor(**{**xp, **(xgb_params or {})})
        self.feat_cols = None

    def _feats(self, df):
        d = df.copy()
        d["year"]=d["Date"].dt.year; d["month"]=d["Date"].dt.month
        d["quarter"]=d["Date"].dt.quarter; d["doy"]=d["Date"].dt.dayofyear
        d["m_sin"]=np.sin(2*np.pi*d["month"]/12); d["m_cos"]=np.cos(2*np.pi*d["month"]/12)
        d["t"]=np.arange(len(d))
        return d

    def fit(self, df):
        self.prophet.fit(df)
        future = self.prophet.fitted.make_future_dataframe(periods=0, freq="MS")
        pf = self.prophet.fitted.predict(future)["yhat"].values
        resid = df["CPI"].values - pf
        f = self._feats(df); f["resid"] = resid
        fc = ["year","month","quarter","doy","m_sin","m_cos","t"]
        cl = f.dropna(); self.xgb.fit(cl[fc], cl["resid"]); self.feat_cols = fc
        return self

    def predict(self, df_future):
        pr = self.prophet.predict(df_future)
        f = self._feats(df_future).fillna(0)
        corr = self.xgb.predict(f[self.feat_cols])
        return {"forecast":pr["forecast"]+corr,"lower":pr["lower"]+corr,
                "upper":pr["upper"]+corr,"prophet":pr["forecast"],"xgb_corr":corr}

    def feature_importances(self):
        return dict(zip(self.feat_cols, self.xgb.feature_importances_))

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_metrics(y_true, y_pred, y_train=None):
    yt = np.array(y_true).flatten(); yp = np.array(y_pred).flatten()
    mask = ~(np.isnan(yt)|np.isnan(yp)); yt, yp = yt[mask], yp[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt-yp)/np.maximum(yt,1e-10)))*100
    mase = mae/np.mean(np.abs(np.diff(y_train))) if y_train is not None and len(y_train)>1 else np.nan
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, MASE=mase)

def auto_sarima(series, max_p=2, max_q=2, max_P=1, max_Q=2, d=1, D=1, m=12):
    best_aic, best_order, best_seasonal = np.inf, (0,1,0), (1,1,1,12)
    y = series.values
    for p,q,P,Q in itertools.product(range(max_p+1),range(max_q+1),range(max_P+1),range(max_Q+1)):
        try:
            res = SARIMAX(y,order=(p,d,q),seasonal_order=(P,D,Q,m),
                enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
            if res.aic < best_aic:
                best_aic=res.aic; best_order=(p,d,q); best_seasonal=(P,D,Q,m)
        except: pass
    return best_order, best_seasonal, best_aic

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS  — y-axis always starts near the data, not zero
# ══════════════════════════════════════════════════════════════════════════════

def _yrange(series, pad=0.08):
    """Return a padded y-axis range that zooms into the data."""
    lo, hi = float(series.min()), float(series.max())
    rng = hi - lo if hi > lo else 1
    return [lo - rng*pad, hi + rng*pad]

def chart_trend(df):
    yr = _yrange(df["CPI"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["CPI"], mode="lines", name="CPI",
        line=dict(color="#00c8ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,255,.06)"))
    fig.update_layout(**BASE, height=300, title="",
        yaxis=dict(range=yr, showgrid=True, gridcolor="#f1f5f9", linecolor="#e8ecf0", zeroline=False),
        xaxis=dict(showgrid=False, linecolor="#e8ecf0", zeroline=False))
    return fig

def chart_yoy(df):
    d = df.copy().sort_values("Date")
    d["YoY"] = d["CPI"].pct_change(12)*100
    d = d.dropna(subset=["YoY"])
    fig = go.Figure()
    fig.add_bar(x=d["Date"], y=d["YoY"],
        marker_color=["#ef4444" if v>0 else "#22c55e" for v in d["YoY"]], name="YoY %")
    fig.add_hline(y=0, line_color="#94a3b8", line_dash="dash", line_width=1)
    fig.update_layout(**BASE, height=240, title="", yaxis_title="% Change")
    return fig

def chart_decomp(df, period=12):
    dec = seasonal_decompose(df.set_index("Date")["CPI"], model="additive", period=period)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
        subplot_titles=["Observed","Trend","Seasonal","Residual"])
    pairs = [(dec.observed,"#00c8ff"),(dec.trend,"#a855f7"),
             (dec.seasonal,"#f59e0b"),(dec.resid,"#ef4444")]
    for i,(s,c) in enumerate(pairs, 1):
        yr = _yrange(s.dropna())
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
            line=dict(color=c, width=1.5), showlegend=False), row=i, col=1)
        fig.update_yaxes(range=yr, row=i, col=1)
    fig.update_layout(**BASE, height=660, title="Seasonal Decomposition — Additive Model")
    return fig

def chart_split(train_df, test_df):
    all_cpi = pd.concat([train_df["CPI"], test_df["CPI"]])
    yr = _yrange(all_cpi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["Date"], y=train_df["CPI"], mode="lines",
        name="Train", line=dict(color="#0a0f1e", width=1.8)))
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["CPI"], mode="lines+markers",
        name="Test", line=dict(color="#00c8ff", width=2.2), marker=dict(size=4)))
    fig.add_shape(type="line", x0=test_df["Date"].min(), x1=test_df["Date"].min(),
        y0=0, y1=1, xref="x", yref="paper", line=dict(color="#ef4444", width=1.5, dash="dash"))
    fig.update_layout(**BASE, height=280, title="",
        yaxis=dict(range=yr, showgrid=True, gridcolor="#f1f5f9", zeroline=False))
    return fig

def chart_model_comparison(train_df, test_df, preds):
    all_vals = list(test_df["CPI"].values) + [v for vals in preds.values() for v in vals]
    yr = _yrange(pd.Series(all_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["Date"], y=train_df["CPI"], mode="lines",
        name="Training", line=dict(color="#cbd5e1", width=1.2)))
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["CPI"], mode="lines+markers",
        name="Actual", line=dict(color="#0a0f1e", width=2.5), marker=dict(size=5)))
    for key, col in COLORS.items():
        if key in preds:
            fig.add_trace(go.Scatter(x=test_df["Date"], y=preds[key], mode="lines",
                name=key.replace("_"," ").title(), line=dict(color=col, width=1.8, dash="dot")))
    fig.update_layout(**BASE, height=360, title="",
        yaxis=dict(range=yr, showgrid=True, gridcolor="#f1f5f9", zeroline=False))
    return fig

def chart_forecast(hist_df, fc_df, model_name, show_ci=True):
    all_vals = list(hist_df["CPI"].values[-36:]) + list(fc_df["Forecast"].values)
    yr = _yrange(pd.Series(all_vals), pad=0.05)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["CPI"], mode="lines",
        name="Historical", line=dict(color="#94a3b8", width=1.5)))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=fc_df["Date"].tolist()+fc_df["Date"].tolist()[::-1],
            y=fc_df["Upper"].tolist()+fc_df["Lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(0,200,255,.1)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines+markers",
        name=f"{model_name} Forecast", line=dict(color="#00c8ff", width=2.5), marker=dict(size=5)))
    fig.add_shape(type="line", x0=hist_df["Date"].max(), x1=hist_df["Date"].max(),
        y0=0, y1=1, xref="x", yref="paper", line=dict(color="#ef4444", width=1.5, dash="dash"))
    fig.add_annotation(x=hist_df["Date"].max(), y=0.97, xref="x", yref="paper",
        text="Forecast start", showarrow=False, yanchor="top",
        font=dict(color="#ef4444", size=10), bgcolor="rgba(255,255,255,.8)", borderpad=3)
    fig.update_layout(**BASE, height=420, title="",
        yaxis=dict(range=yr, showgrid=True, gridcolor="#f1f5f9", zeroline=False))
    return fig

def chart_whatif(hist_df, fc_df, shock_pct):
    shocked = fc_df.copy()
    shocked["Forecast"] = shocked["Forecast"]*(1+shock_pct/100)
    all_vals = list(hist_df["CPI"].values[-24:]) + list(fc_df["Forecast"]) + list(shocked["Forecast"])
    yr = _yrange(pd.Series(all_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["CPI"], mode="lines",
        name="Historical", line=dict(color="#94a3b8", width=1.5)))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines",
        name="Base Forecast", line=dict(color="#00c8ff", width=2.2)))
    fig.add_trace(go.Scatter(x=shocked["Date"], y=shocked["Forecast"], mode="lines",
        name=f"Shocked ({shock_pct:+.1f}%)", line=dict(color="#f97316", width=2.2, dash="dot")))
    fig.add_shape(type="line", x0=hist_df["Date"].max(), x1=hist_df["Date"].max(),
        y0=0, y1=1, xref="x", yref="paper", line=dict(color="#ef4444", width=1.5, dash="dash"))
    fig.update_layout(**BASE, height=380, title="",
        yaxis=dict(range=yr, showgrid=True, gridcolor="#f1f5f9", zeroline=False))
    return fig

def chart_feature_importance(importances, model_name):
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [i[0] for i in items]; vals = [i[1] for i in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#e0f2fe"],[1,"#00c8ff"]], showscale=False),
        marker_line_width=0))
    la = {**BASE, "height":280, "title":f"{model_name}",
          "xaxis_title":"Importance", "yaxis":dict(autorange="reversed")}
    fig.update_layout(**la)
    return fig

def chart_residuals(y_true, y_pred, model_name, color):
    resid = np.array(y_true)-np.array(y_pred)
    max_lags = min(20, len(resid)//2 - 1)
    acf_v  = acf(resid, nlags=max_lags, fft=True)
    pacf_v = pacf(resid, nlags=max_lags)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Residuals","ACF","PACF"],
                        horizontal_spacing=.08)
    fig.add_trace(go.Scatter(x=list(range(len(resid))), y=resid, mode="lines",
        line=dict(color=color, width=1.2), showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(acf_v))), y=acf_v,
        marker_color=color, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=list(range(len(pacf_v))), y=pacf_v,
        marker_color=color, showlegend=False), row=1, col=3)
    ci = 1.96/np.sqrt(len(resid))
    for cc in [2, 3]:
        fig.add_hline(y=ci, line_dash="dot", line_color="#ef4444", row=1, col=cc)
        fig.add_hline(y=-ci, line_dash="dot", line_color="#ef4444", row=1, col=cc)
    fig.update_layout(**BASE, height=260, title=f"Residuals — {model_name}", showlegend=False)
    return fig

def chart_rolling(df_cpi, window=48, step=6):
    records = []; n = len(df_cpi)
    for start in range(window, n-12, step):
        train = df_cpi.iloc[:start]; test = df_cpi.iloc[start:start+12]
        if len(test) < 6: break
        y_true = test["CPI"].values
        for label, factory in [("SARIMA", lambda: SARIMAForecaster((0,1,0),(1,1,2,12))),
                                ("Prophet Boost", lambda: ProphetBoostForecaster())]:
            try:
                m = factory(); m.fit(train)
                p = m.predict(len(test)) if label=="SARIMA" else m.predict(test)["forecast"]
                records.append({"Date":test["Date"].iloc[0],"Model":label,
                    "RMSE":np.sqrt(mean_squared_error(y_true,p))})
            except: pass
    if not records: return go.Figure()
    df_r = pd.DataFrame(records); fig = go.Figure()
    for model, col in [("SARIMA","#ef4444"),("Prophet Boost","#00c8ff")]:
        sub = df_r[df_r["Model"]==model]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["RMSE"], mode="lines+markers",
                name=model, line=dict(color=col, width=2), marker=dict(size=5)))
    fig.update_layout(**BASE, height=340, title="",
        yaxis=dict(title="RMSE", showgrid=True, gridcolor="#f1f5f9", zeroline=False))
    return fig

def chart_subcategory(df_long, codes, periods=18):
    fig = go.Figure()
    palette = ["#00c8ff","#a855f7","#f97316","#22c55e","#ef4444"]
    for code, color in zip(codes, palette):
        sub = get_series(df_long, code)
        if len(sub) < 30: continue
        try:
            m = ProphetBoostForecaster().fit(sub)
            future = generate_future_dates(sub["Date"].max(), periods)
            fc = m.predict(future)
            cat_name = df_long[df_long["Category_Code"]==code]["Category"].iloc[0][:30]
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["CPI"], mode="lines",
                name=f"{cat_name} (hist)", line=dict(color=color, width=1, dash="dot"),
                opacity=.35, showlegend=False))
            fig.add_trace(go.Scatter(x=future["Date"], y=fc["forecast"], mode="lines",
                name=cat_name, line=dict(color=color, width=2.2)))
        except: pass
    fig.update_layout(**BASE, height=400, title="")
    return fig

def chart_metrics_bars(results):
    models = list(results.keys())
    pal = [COLORS.get(k,"#64748b") for k in models]
    fig = make_subplots(rows=2, cols=2, subplot_titles=["RMSE","MAE","MAPE (%)","MASE"],
                        vertical_spacing=.2, horizontal_spacing=.1)
    for idx, metric in enumerate(["RMSE","MAE","MAPE","MASE"]):
        r, c = idx//2+1, idx%2+1
        vals = [results[m][metric] for m in models]
        fig.add_trace(go.Bar(x=[m.replace("_"," ").title() for m in models], y=vals,
            marker_color=pal, text=[f"{v:.3f}" for v in vals],
            textposition="auto", showlegend=False), row=r, col=c)
    fig.update_layout(**BASE, height=520, title="", showlegend=False)
    return fig

def build_excel(results, forecast_df, hist_df, test_preds_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(results).T.round(4).to_excel(w, sheet_name="Model_Metrics")
        if len(forecast_df): forecast_df.to_excel(w, sheet_name="Forecast", index=False)
        hist_df.to_excel(w, sheet_name="Historical_CPI", index=False)
        if len(test_preds_df): test_preds_df.to_excel(w, sheet_name="Test_Predictions", index=False)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_bytes(raw):
    return clean_wide_to_long(pd.read_excel(io.BytesIO(raw), header=None))

@st.cache_data(show_spinner=False)
def load_default():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "instat1_20260319-113807.xlsx")
    if os.path.exists(path):
        with open(path,"rb") as f:
            return clean_wide_to_long(pd.read_excel(f, header=None))
    return None

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (early, needed for hero stats)
# ══════════════════════════════════════════════════════════════════════════════

# We need the upload widget before hero, but render hero first visually.
# Solution: render hero placeholder, then controls, then fill hero with real data.

uploaded_raw = None

with st.spinner(""):
    _default = load_default()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

_obs  = len(_default) // len(_default["Category_Code"].unique()) if _default is not None else 228
_cats = _default["Category_Code"].nunique() if _default is not None else 0
_yrs  = f"{_default['Date'].min().year}–{_default['Date'].max().year}" if _default is not None else "2007–2025"

st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eyebrow">Albania · INSTAT · Consumer Price Index</div>
  <div class="hero-title">Forecast <em>inflation</em><br>with precision.</div>
  <div class="hero-sub">
    Four competing models — SARIMA, Boosted SARIMA, Prophet, and Prophet Boost —
    evaluated head-to-head on Albanian CPI data from {_yrs}.
    The best model wins and forecasts forward.
  </div>
  <div class="hero-badges">
    <span class="hbadge">Auto SARIMA Selection</span>
    <span class="hbadge">XGBoost Residual Boosting</span>
    <span class="hbadge">What-If Shock Scenarios</span>
    <span class="hbadge">Feature Importance</span>
    <span class="hbadge">Walk-Forward Validation</span>
    <span class="hbadge">Sub-Category Analysis</span>
    <span class="hbadge">Excel Export</span>
  </div>
  <div class="hero-stats">
    <div>
      <div class="hero-stat-value">{_obs}</div>
      <div class="hero-stat-label">Monthly observations</div>
    </div>
    <div>
      <div class="hero-stat-value">{_yrs}</div>
      <div class="hero-stat-label">Date range</div>
    </div>
    <div>
      <div class="hero-stat-value">{_cats}</div>
      <div class="hero-stat-label">CPI categories</div>
    </div>
    <div>
      <div class="hero-stat-value">4</div>
      <div class="hero-stat-label">Models compared</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTROL PANEL  — compact 2-row layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='padding:0 2rem'>", unsafe_allow_html=True)
st.markdown('<div class="ctrl-wrap">', unsafe_allow_html=True)
st.markdown('<div class="ctrl-label">⚙ Pipeline Configuration</div>', unsafe_allow_html=True)


# Row 1: upload | train ratio | forecast horizon | train button
r1a, r1b, r1c, r1d = st.columns([2.5, 1, 1, 1])  # Adjusted ratios

with r1a:
    uploaded = st.file_uploader("Upload CPI Excel (optional)",  # Shortened label
                                 type=["xlsx","xls"], label_visibility="visible")
with r1b:
    train_ratio = st.slider("Train ratio", .5, .9, .8, .05, label_visibility="collapsed")
    st.caption("Train ratio")  # Separate caption to save space
with r1c:
    forecast_periods = st.slider("Forecast months", 12, 60, 36, 6, label_visibility="collapsed")  # Shortened
    st.caption("Forecast months")
with r1d:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)  # Reduced spacer
    run_btn = st.button("▶ Train All", type="primary", use_container_width=True)  # Shortened text

# Row 2: spacer | confidence intervals toggle | auto SARIMA toggle
_, r2b, r2c, _ = st.columns([3, 1.2, 1.2, 1.2])
with r2b:
    show_ci = st.toggle("Show confidence intervals", value=True)
with r2c:
    auto_order = st.toggle("Auto-select SARIMA order", value=False)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data…"):
    if uploaded is not None:
        df_long = load_bytes(uploaded.read())
        st.success("✅ Custom file loaded.")
    else:
        df_long = load_default()
        if df_long is None:
            st.error("❌ Default data not found. Please upload your CPI Excel file above.")
            st.stop()

all_codes  = sorted(df_long["Category_Code"].unique().tolist())
cat_labels = {c: f"{c} — {df_long[df_long['Category_Code']==c]['Category'].iloc[0][:55]}"
              for c in all_codes}

# Category selector + KPI strip
sel_col, _, _ = st.columns([2, 2, 1])
with sel_col:
    selected_code = st.selectbox("CPI Category", options=all_codes,
        format_func=lambda c: cat_labels[c],
        index=all_codes.index("000000") if "000000" in all_codes else 0)

df_cpi = get_series(df_long, selected_code)
total_growth = (df_cpi["CPI"].iloc[-1]/df_cpi["CPI"].iloc[0]-1)*100

st.markdown(f"""
<div class="metric-strip">
  <div class="metric-cell">
    <div class="lbl">Observations</div>
    <div class="val">{len(df_cpi)}</div>
  </div>
  <div class="metric-cell">
    <div class="lbl">Date range</div>
    <div class="val">{df_cpi['Date'].min().year}–{df_cpi['Date'].max().year}</div>
  </div>
  <div class="metric-cell">
    <div class="lbl">CPI Start</div>
    <div class="val">{df_cpi['CPI'].iloc[0]:.2f}</div>
  </div>
  <div class="metric-cell">
    <div class="lbl">CPI Latest</div>
    <div class="val">{df_cpi['CPI'].iloc[-1]:.2f}</div>
    <div class="delta">↑ {total_growth:.1f}% total growth</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Historical","🔬 Decomposition","⚙️ Training",
                "📊 Results","🔮 Forecast","🎯 What-If",
                "🔍 Diagnostics","🔄 Rolling","📦 Sub-Categories"])

# ── 0 Historical ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec-head">Historical <em>CPI</em> Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_trend(df_cpi), use_container_width=True)

    st.markdown('<div class="sec-head">Year-over-Year <em>Inflation</em> Rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_yoy(df_cpi), use_container_width=True)

    st.markdown('<div class="sec-head">ADF Stationarity Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    adf_res = adfuller(df_cpi["CPI"].dropna())
    is_stat = adf_res[1] <= .05
    a1, a2, a3 = st.columns(3)
    a1.metric("ADF Statistic", f"{adf_res[0]:.4f}")
    a2.metric("p-value", f"{adf_res[1]:.4f}")
    a3.metric("Result", "✅ Stationary" if is_stat else "⚠️ Non-Stationary")
    st.markdown(f"""<div class="{'info-card' if is_stat else 'warn-card'}">
    {"Series is stationary — models can be applied directly." if is_stat else
     "Non-stationary series (expected for CPI data). SARIMA handles this via d=1 differencing; Prophet fits an explicit trend function. No action required."}
    </div>""", unsafe_allow_html=True)
    with st.expander("View raw data table"):
        st.dataframe(df_cpi.style.format({"CPI":"{:.2f}"}), use_container_width=True, height=300)
        st.download_button("⬇ Download CSV", df_cpi.to_csv(index=False), "cpi_data.csv","text/csv")

# ── 1 Decomposition ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="sec-head">Seasonal <em>Decomposition</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_decomp(df_cpi), use_container_width=True)
    st.markdown("""<div class="info-card">
    <strong>Additive model:</strong> Y(t) = Trend(t) + Seasonal(t) + Residual(t) &nbsp;·&nbsp;
    Period = 12 months &nbsp;·&nbsp; Small, noise-like residuals confirm the model captures the main structure well.
    </div>""", unsafe_allow_html=True)

# ── 2 Training ────────────────────────────────────────────────────────────────
with tabs[2]:
    train_df, test_df = chronological_split(df_cpi, train_ratio)
    st.markdown('<div class="sec-head">Train / Test <em>Split</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_split(train_df, test_df), use_container_width=True)
    t1, t2, t3 = st.columns(3)
    t1.metric("Train", len(train_df), f"until {train_df['Date'].max().strftime('%b %Y')}")
    t2.metric("Test",  len(test_df),  f"from  {test_df['Date'].min().strftime('%b %Y')}")
    t3.metric("Split date", test_df["Date"].min().strftime("%Y-%m"))

    if not run_btn:
        st.markdown("""<div class="info-card">
        Press <strong>▶ Train All Models</strong> in the configuration panel above to fit all four models on the training data.
        </div>""", unsafe_allow_html=True)
    else:
        pbar = st.progress(0, "Starting…")
        if auto_order:
            with st.spinner("Grid-searching best SARIMA order by AIC…"):
                best_order, best_seasonal, best_aic = auto_sarima(train_df["CPI"])
            st.success(f"Best SARIMA: {best_order}{best_seasonal} — AIC {best_aic:.2f}")
        else:
            best_order, best_seasonal = (0,1,0),(1,1,2,12)

        preds = {}; fitted_models = {}

        pbar.progress(10, "Fitting SARIMA…")
        m1 = SARIMAForecaster(best_order, best_seasonal).fit(train_df)
        preds["sarima"] = m1.predict(len(test_df)); fitted_models["sarima"] = m1

        pbar.progress(30, "Fitting Boosted SARIMA…")
        m2 = BoostedSARIMA().fit(train_df)
        preds["boosted_sarima"] = m2.predict(test_df); fitted_models["boosted_sarima"] = m2

        pbar.progress(55, "Fitting Prophet…")
        m3 = ProphetForecaster(growth="linear",yearly_seasonality=True,weekly_seasonality=False,
            daily_seasonality=False,n_changepoints=25,changepoint_range=.8,
            changepoint_prior_scale=.05,seasonality_prior_scale=10.)
        m3.fit(train_df)
        preds["prophet"] = m3.predict(test_df)["forecast"]; fitted_models["prophet"] = m3

        pbar.progress(80, "Fitting Prophet Boost…")
        m4 = ProphetBoostForecaster().fit(train_df)
        preds["prophet_boost"] = m4.predict(test_df)["forecast"]; fitted_models["prophet_boost"] = m4

        pbar.progress(100, "Done ✓"); pbar.empty()

        y_true = test_df["CPI"].values; y_train = train_df["CPI"].values
        results = {k: calc_metrics(y_true, v, y_train) for k,v in preds.items()}
        best_key = min(results, key=lambda k: results[k]["RMSE"])

        st.session_state.update({
            "train_df":train_df,"test_df":test_df,"preds":preds,
            "results":results,"best_key":best_key,"fitted_models":fitted_models,
            "sarima_order":best_order,"sarima_seasonal":best_seasonal,"models_trained":True
        })
        st.success(f"✅ All models trained. Winner by RMSE: **{best_key.replace('_',' ').title()}**")

    if st.session_state.get("models_trained"):
        st.markdown('<div class="sec-head">Model <em>Comparison</em> on Test Set</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.plotly_chart(chart_model_comparison(
            st.session_state["train_df"], st.session_state["test_df"],
            st.session_state["preds"]), use_container_width=True)

# ── 3 Results ─────────────────────────────────────────────────────────────────
with tabs[3]:
    if not st.session_state.get("models_trained"):
        st.warning("Train models first — press **▶ Train All Models** above.")
    else:
        results  = st.session_state["results"]
        best_key = st.session_state["best_key"]
        cmp = pd.DataFrame(results).T.round(4)
        cmp.index = [i.replace("_"," ").title() for i in cmp.index]

        st.markdown(
            f'<div class="sec-head">Evaluation <em>Metrics</em> — '
            f'Winner: {best_key.replace("_"," ").title()} <span class="winner-tag">★ BEST</span></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

        st.dataframe(cmp.style.format("{:.4f}").highlight_min(axis=0, color="#dcfce7"),
                     use_container_width=True, height=200)
        st.plotly_chart(chart_metrics_bars(results), use_container_width=True)

        st.markdown('<div class="sec-head">XGBoost <em>Feature Importance</em></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        fi1, fi2 = st.columns(2)
        for widget, key in zip([fi1,fi2], ["boosted_sarima","prophet_boost"]):
            m = st.session_state["fitted_models"].get(key)
            if m and hasattr(m, "feature_importances"):
                widget.plotly_chart(chart_feature_importance(
                    m.feature_importances(), key.replace("_"," ").title()), use_container_width=True)

        tp = pd.DataFrame({"Date":st.session_state["test_df"]["Date"],
            "Actual":st.session_state["test_df"]["CPI"].values,
            **{k.replace("_"," ").title():v for k,v in st.session_state["preds"].items()}})
        st.download_button("⬇ Download Results Excel",
            build_excel(results, pd.DataFrame(), df_cpi, tp),
            "cpi_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

# ── 4 Forecast ────────────────────────────────────────────────────────────────
with tabs[4]:
    if not st.session_state.get("models_trained"):
        st.warning("Train models first.")
    else:
        if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
            best_key  = st.session_state["best_key"]
            future_df = generate_future_dates(df_cpi["Date"].max(), forecast_periods)
            with st.spinner(f"Fitting {best_key.replace('_',' ').title()} on full dataset…"):
                if best_key == "sarima":
                    fm = SARIMAForecaster(st.session_state["sarima_order"],
                                         st.session_state["sarima_seasonal"]).fit(df_cpi)
                    raw = fm.predict(forecast_periods)
                    std = float(np.std(df_cpi["CPI"].diff().dropna()))
                    fc_result = {"forecast":raw,"lower":raw-1.96*std,"upper":raw+1.96*std}
                elif best_key == "boosted_sarima":
                    fm = BoostedSARIMA().fit(df_cpi)
                    raw = fm.predict(future_df)
                    std = float(np.std(df_cpi["CPI"].diff().dropna()))
                    fc_result = {"forecast":raw,"lower":raw-1.96*std,"upper":raw+1.96*std}
                elif best_key == "prophet":
                    fm = ProphetForecaster(growth="linear",yearly_seasonality=True,
                        weekly_seasonality=False,daily_seasonality=False,
                        n_changepoints=25,changepoint_range=.8,
                        changepoint_prior_scale=.05,seasonality_prior_scale=10.).fit(df_cpi)
                    pr = fm.predict(future_df)
                    fc_result = {"forecast":pr["forecast"],"lower":pr["lower"],"upper":pr["upper"]}
                else:
                    fm = ProphetBoostForecaster().fit(df_cpi)
                    pb = fm.predict(future_df)
                    fc_result = {"forecast":pb["forecast"],"lower":pb["lower"],"upper":pb["upper"]}

                fc_df = pd.DataFrame({"Date":future_df["Date"],"Forecast":fc_result["forecast"],
                    "Lower":fc_result["lower"],"Upper":fc_result["upper"]})
                st.session_state.update({"fc_df":fc_df,"fc_model":best_key,"forecast_done":True})

        if st.session_state.get("forecast_done"):
            fc_df    = st.session_state["fc_df"]
            fc_model = st.session_state["fc_model"].replace("_"," ").title()
            st.markdown(f'<div class="sec-head">{forecast_periods}-Month <em>Forecast</em></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
            st.plotly_chart(chart_forecast(df_cpi, fc_df, fc_model, show_ci), use_container_width=True)
            last = df_cpi["CPI"].iloc[-1]; end = fc_df["Forecast"].iloc[-1]
            f1,f2,f3,f4 = st.columns(4)
            f1.metric("Current CPI",  f"{last:.2f}")
            f2.metric("Forecast end", f"{end:.2f}", delta=f"{end-last:+.2f}")
            f3.metric("Total growth", f"{(end/last-1)*100:+.2f}%")
            f4.metric("Model used",   fc_model)
            disp = fc_df.copy(); disp["Date"] = disp["Date"].dt.strftime("%Y-%m")
            st.dataframe(disp.round(2), use_container_width=True, height=300)
            tp = pd.DataFrame({"Date":st.session_state["test_df"]["Date"],
                "Actual":st.session_state["test_df"]["CPI"].values,
                **{k.replace("_"," ").title():v for k,v in st.session_state["preds"].items()}})
            st.download_button("⬇ Download Full Results Excel",
                build_excel(st.session_state["results"], fc_df, df_cpi, tp),
                "albania_cpi_full_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

# ── 5 What-If ─────────────────────────────────────────────────────────────────
with tabs[5]:
    if not st.session_state.get("forecast_done"):
        st.warning("Generate a forecast first (Forecast tab).")
    else:
        st.markdown('<div class="sec-head">Inflation <em>Shock</em> Simulator</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        Apply a hypothetical one-time percentage shock to the entire forecast path.
        Useful for stress-testing: energy crisis (+15%), policy intervention (−5%), etc.
        </div>""", unsafe_allow_html=True)
        shock = st.slider("Shock magnitude (%)", -20.0, 30.0, 0.0, 0.5)
        st.plotly_chart(chart_whatif(df_cpi, st.session_state["fc_df"], shock), use_container_width=True)
        base_end = st.session_state["fc_df"]["Forecast"].iloc[-1]
        shocked_end = base_end*(1+shock/100)
        s1,s2,s3 = st.columns(3)
        s1.metric("Base end CPI",    f"{base_end:.2f}")
        s2.metric("Shocked end CPI", f"{shocked_end:.2f}", delta=f"{shocked_end-base_end:+.2f}")
        s3.metric("Shock applied",   f"{shock:+.1f}%")

# ── 6 Diagnostics ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="sec-head">CPI by <em>Category</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    
    # Get main categories (exclude subcategories for cleaner view)
    main_codes = [c for c in all_codes if c.endswith("0000") or c == "000000"]
    
    # Chart 1: All main categories over time
    fig1 = go.Figure()
    colors = ["#00c8ff", "#ef4444", "#f97316", "#22c55e", "#a855f7", "#f59e0b", "#8b5cf6"]
    
    for i, code in enumerate(main_codes[:7]):  # Limit to 7 for readability
        df_cat = get_series(df_long, code)
        if len(df_cat) > 0:
            name = cat_labels[code].split("—")[1][:30] if "—" in cat_labels[code] else cat_labels[code][:30]
            fig1.add_trace(go.Scatter(
                x=df_cat["Date"], 
                y=df_cat["CPI"],
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig1.update_layout(
        **BASE,
        height=400,
        title="CPI Trends by Main Category",
        yaxis_title="CPI Index",
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: YoY inflation comparison (latest month)
    st.markdown('<div class="sec-head">Latest <em>YoY Inflation</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    
    latest = df_long["Date"].max()
    year_ago = latest - pd.DateOffset(months=12)
    
    inflation_data = []
    for code in main_codes:
        df_cat = get_series(df_long, code)
        if len(df_cat) >= 13:
            latest_val = df_cat[df_cat["Date"] == latest]["CPI"].values
            past_val = df_cat[df_cat["Date"] == year_ago]["CPI"].values
            
            if len(latest_val) > 0 and len(past_val) > 0:
                yoy = ((latest_val[0] / past_val[0]) - 1) * 100
                name = cat_labels[code].split("—")[1][:35] if "—" in cat_labels[code] else cat_labels[code][:35]
                inflation_data.append({"Category": name, "Inflation": yoy})
    
    if inflation_data:
        inf_df = pd.DataFrame(inflation_data).sort_values("Inflation", ascending=True)
        
        fig2 = go.Figure()
        colors2 = ["#22c55e" if x < 0 else "#ef4444" for x in inf_df["Inflation"]]
        fig2.add_trace(go.Bar(
            y=inf_df["Category"],
            x=inf_df["Inflation"],
            orientation="h",
            marker_color=colors2,
            text=[f"{x:+.1f}%" for x in inf_df["Inflation"]],
            textposition="outside"
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig2.update_layout(
            **BASE,
            height=350,
            title=f"YoY Inflation by Category ({latest.strftime('%b %Y')})",
            xaxis_title="% Change",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
# ── 7 Rolling ─────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="sec-head">Walk-Forward <em>Rolling Accuracy</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Retrains SARIMA and Prophet Boost every 6 months across the full history,
    evaluating 12-month-ahead RMSE each time. Reveals how each model would have
    performed in real deployment — not just on a single test split. Takes ~2 minutes.
    </div>""", unsafe_allow_html=True)
    if st.button("▶ Run Rolling Evaluation", use_container_width=True):
        with st.spinner("Running walk-forward evaluation…"):
            st.plotly_chart(chart_rolling(df_cpi, window=48, step=6), use_container_width=True)

# ── 8 Sub-Categories ──────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown('<div class="sec-head">Sub-Category <em>Comparison</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Forecast multiple CPI sub-categories side-by-side using Prophet Boost.
    Pinpoints which sectors are driving or moderating overall inflation.
    </div>""", unsafe_allow_html=True)
    available = [c for c in all_codes if len(get_series(df_long,c))>=30]
    selected_cats = st.multiselect("Choose categories (max 5)", options=available,
        format_func=lambda c: cat_labels[c],
        default=available[:3] if len(available)>=3 else available, max_selections=5)
    sub_periods = st.slider("Forecast horizon (months)", 6, 36, 18, 6)
    if st.button("▶ Forecast Sub-Categories", use_container_width=True):
        if not selected_cats:
            st.warning("Select at least one category.")
        else:
            with st.spinner("Forecasting sub-categories with Prophet Boost…"):
                st.plotly_chart(chart_subcategory(df_long, selected_cats, sub_periods),
                                use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close content-wrap 

st.markdown("""
<div style='text-align:center;padding:3rem 0 1.5rem;font-size:.75rem;color:#94a3b8;
     border-top:1px solid #e8ecf0;margin-top:3rem'>
  Albania CPI Forecasting Pipeline &nbsp;·&nbsp;
  SARIMA · Prophet · XGBoost Hybrid Models
</div>""", unsafe_allow_html=True)