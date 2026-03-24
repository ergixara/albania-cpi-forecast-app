"""
Albania CPI Forecasting — Full Featured Streamlit App
Enhancements: category selector, YoY chart, CI toggle, Excel export,
feature importance, what-if scenarios, residual diagnostics, rolling accuracy,
sub-category comparison, automated SARIMA order selection.
"""

import warnings, logging, itertools
import os
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
from typing import Tuple, Dict, Optional
import io

np.random.seed(42)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Albania CPI Forecaster", page_icon="📈", 
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
section[data-testid="stSidebar"] { background: #0a0f1e; border-right: 1px solid #1a2340; }
section[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 60%, #0a0f1e 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
    border: 1px solid #1a3050; position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,.06) 0%, transparent 70%);
}
.hero h1 { font-size: 2.6rem; font-weight: 700; color: #fff; margin: 0 0 .4rem; }
.hero h1 span { color: #00c8ff; }
.hero p { color: #7a90b0; font-size: .95rem; margin: 0; }
.badge {
    display: inline-block; background: rgba(0,200,255,.1);
    border: 1px solid rgba(0,200,255,.25); color: #00c8ff;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    padding: .2rem .6rem; border-radius: 999px; margin: .6rem .3rem 0 0;
}
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #0a0f1e;
    padding: .5rem 0; border-bottom: 2px solid #e8edf5; margin: 1.5rem 0 1rem;
}
.winner-tag {
    background: #0a0f1e; color: #00c8ff;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    padding: .2rem .7rem; border-radius: 999px; margin-left: .5rem;
}
.info-card {
    background: #f0f9ff; border-left: 3px solid #00c8ff;
    padding: .9rem 1.1rem; border-radius: 0 8px 8px 0;
    font-size: .85rem; margin: .8rem 0; color: #1a2340;
}
.warn-card {
    background: #fff8f0; border-left: 3px solid #ff9500;
    padding: .9rem 1.1rem; border-radius: 0 8px 8px 0;
    font-size: .85rem; margin: .8rem 0; color: #4a2800;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
            
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: white !important;
    background: #1a2340 !important;
    border-radius: 0 8px 8px 0 !important;
}
button[kind="header"] {
    color: white !important;
    visibility: visible !important;
}

.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# ── Colours ────────────────────────────────────────────────────────────────────
COLORS = {
    "sarima": "#ef4444", "boosted_sarima": "#f97316",
    "prophet": "#a855f7", "prophet_boost": "#00c8ff",
}
BASE = dict(
    font_family="Sora", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False),
    margin=dict(l=10, r=10, t=45, b=10),
    legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#e2e8f0", borderwidth=1),
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
# MODEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAForecaster:
    def __init__(self, order=(0,1,0), seasonal_order=(1,1,2,12)):
        self.order = order; self.seasonal_order = seasonal_order
        self.fitted_model = self.fitted_values = self.residuals = None

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
                f[col] = f[col].fillna(pd.Series(sp).mean())
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
        
        # Fix for Python 3.8: use dict unpacking instead of |
        merged_prophet_params = {**pp, **(prophet_params or {})}
        merged_xgb_params = {**xp, **(xgb_params or {})}
        
        self.prophet = ProphetForecaster(**merged_prophet_params)
        self.xgb = XGBRegressor(**merged_xgb_params)
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
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"],y=df["CPI"],mode="lines",name="CPI",
        line=dict(color="#00c8ff",width=2.5),fill="tozeroy",fillcolor="rgba(0,200,255,.07)"))
    fig.update_layout(**BASE,height=380,title="Historical CPI Series")
    return fig

def chart_yoy(df):
    d = df.copy().sort_values("Date")
    d["YoY"] = d["CPI"].pct_change(12)*100
    d = d.dropna(subset=["YoY"])
    colors = ["#ef4444" if v>0 else "#22c55e" for v in d["YoY"]]
    fig = go.Figure()
    fig.add_bar(x=d["Date"],y=d["YoY"],marker_color=colors,name="YoY %")
    fig.add_hline(y=0,line_color="#64748b",line_dash="dash")
    fig.update_layout(**BASE,height=360,title="Year-over-Year Inflation Rate (%)",yaxis_title="% Change")
    return fig

def chart_decomp(df, period=12):
    dec = seasonal_decompose(df.set_index("Date")["CPI"],model="additive",period=period)
    fig = make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.05,
        subplot_titles=["Observed","Trend","Seasonal","Residual"])
    for i,(s,c) in enumerate([(dec.observed,"#00c8ff"),(dec.trend,"#a855f7"),
                               (dec.seasonal,"#f59e0b"),(dec.resid,"#ef4444")],1):
        fig.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines",
            line=dict(color=c,width=1.5),showlegend=False),row=i,col=1)
    fig.update_layout(**BASE,height=700,title="Seasonal Decomposition (Additive)")
    return fig

def chart_split(train_df, test_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["Date"],y=train_df["CPI"],mode="lines",
        name="Train",line=dict(color="#0a0f1e",width=2)))
    fig.add_trace(go.Scatter(x=test_df["Date"],y=test_df["CPI"],mode="lines+markers",
        name="Test",line=dict(color="#00c8ff",width=2.5),marker=dict(size=5)))
    fig.add_shape(type="line",x0=test_df["Date"].min(),x1=test_df["Date"].min(),
        y0=0,y1=1,xref="x",yref="paper",line=dict(color="#ef4444",width=2,dash="dash"))
    fig.update_layout(**BASE,height=340,title="Train / Test Split")
    return fig

def chart_model_comparison(train_df, test_df, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["Date"],y=train_df["CPI"],mode="lines",
        name="Training",line=dict(color="#94a3b8",width=1.5)))
    fig.add_trace(go.Scatter(x=test_df["Date"],y=test_df["CPI"],mode="lines+markers",
        name="Actual",line=dict(color="#0a0f1e",width=2.5),marker=dict(size=6)))
    for key,col in COLORS.items():
        if key in preds:
            fig.add_trace(go.Scatter(x=test_df["Date"],y=preds[key],mode="lines+markers",
                name=key.replace("_"," ").title(),line=dict(color=col,width=2,dash="dot"),
                marker=dict(size=4)))
    fig.update_layout(**BASE,height=420,title="Model Predictions vs Actual (Test Set)")
    return fig

def chart_forecast(hist_df, fc_df, model_name, show_ci=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Date"],y=hist_df["CPI"],mode="lines",
        name="Historical",line=dict(color="#64748b",width=1.5)))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=fc_df["Date"].tolist()+fc_df["Date"].tolist()[::-1],
            y=fc_df["Upper"].tolist()+fc_df["Lower"].tolist()[::-1],
            fill="toself",fillcolor="rgba(0,200,255,.12)",
            line=dict(color="rgba(255,255,255,0)"),name="95% CI"))
    fig.add_trace(go.Scatter(x=fc_df["Date"],y=fc_df["Forecast"],mode="lines+markers",
        name=f"{model_name} Forecast",line=dict(color="#00c8ff",width=3),marker=dict(size=5)))
    fig.add_shape(type="line",x0=hist_df["Date"].max(),x1=hist_df["Date"].max(),
        y0=0,y1=1,xref="x",yref="paper",line=dict(color="#ef4444",width=2,dash="dash"))
    fig.add_annotation(x=hist_df["Date"].max(),y=1,xref="x",yref="paper",
        text="Forecast start",showarrow=False,yanchor="bottom",font=dict(color="#ef4444",size=11))
    fig.update_layout(**BASE,height=500,title=f"CPI Forecast — {model_name}")
    return fig

def chart_whatif(hist_df, fc_df, shock_pct):
    shocked = fc_df.copy()
    shocked["Forecast"] = shocked["Forecast"]*(1+shock_pct/100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Date"],y=hist_df["CPI"],mode="lines",
        name="Historical",line=dict(color="#64748b",width=1.5)))
    fig.add_trace(go.Scatter(x=fc_df["Date"],y=fc_df["Forecast"],mode="lines",
        name="Base Forecast",line=dict(color="#00c8ff",width=2.5)))
    fig.add_trace(go.Scatter(x=shocked["Date"],y=shocked["Forecast"],mode="lines",
        name=f"Shocked ({shock_pct:+.1f}%)",line=dict(color="#f97316",width=2.5,dash="dot")))
    fig.add_shape(type="line",x0=hist_df["Date"].max(),x1=hist_df["Date"].max(),
        y0=0,y1=1,xref="x",yref="paper",line=dict(color="#ef4444",width=1.5,dash="dash"))
    fig.update_layout(**BASE,height=420,title="What-If Scenario: Inflation Shock")
    return fig

def chart_feature_importance(importances, model_name):
    items = sorted(importances.items(),key=lambda x:x[1],reverse=True)
    names=[i[0] for i in items]; vals=[i[1] for i in items]
    fig = go.Figure(go.Bar(x=vals,y=names,orientation="h",
        marker_color="#00c8ff",marker_line_width=0))
    
    # Create a copy of BASE and update yaxis instead of passing it twice
    layout_args = BASE.copy()
    layout_args.update({
        "height": 320,
        "title": f"Feature Importance — {model_name}",
        "xaxis_title": "Importance",
        "yaxis": dict(autorange="reversed")
    })
    
    fig.update_layout(**layout_args)
    return fig

def chart_residuals(test_df, y_true, y_pred, model_name, color):
    resid = np.array(y_true)-np.array(y_pred)
    max_lags = min(20, len(resid)//2 - 1)
    acf_v = acf(resid, nlags=max_lags, fft=True)
    pacf_v = pacf(resid, nlags=max_lags)
    fig = make_subplots(rows=1,cols=3,subplot_titles=["Residuals","ACF","PACF"],
                        horizontal_spacing=.1)
    fig.add_trace(go.Scatter(x=list(range(len(resid))),y=resid,mode="lines+markers",
        line=dict(color=color,width=1.5),marker=dict(size=4),showlegend=False),row=1,col=1)
    fig.add_hline(y=0,line_dash="dash",line_color="#94a3b8",row=1,col=1)
    fig.add_trace(go.Bar(x=list(range(len(acf_v))),y=acf_v,
        marker_color=color,showlegend=False),row=1,col=2)
    fig.add_trace(go.Bar(x=list(range(len(pacf_v))),y=pacf_v,
        marker_color=color,showlegend=False),row=1,col=3)
    ci = 1.96/np.sqrt(len(resid))
    for ci_col in [2,3]:
        fig.add_hline(y=ci,line_dash="dot",line_color="#ef4444",row=1,col=ci_col)
        fig.add_hline(y=-ci,line_dash="dot",line_color="#ef4444",row=1,col=ci_col)
    fig.update_layout(**BASE,height=300,title=f"Residual Diagnostics — {model_name}",showlegend=False)
    return fig

def chart_rolling(df_cpi, window=48, step=6):
    records = []; n = len(df_cpi)
    for start in range(window, n-12, step):
        train = df_cpi.iloc[:start]; test = df_cpi.iloc[start:start+12]
        if len(test)<6: break
        y_true = test["CPI"].values
        for label, factory in [
            ("SARIMA", lambda: SARIMAForecaster((0,1,0),(1,1,2,12))),
            ("Prophet Boost", lambda: ProphetBoostForecaster())
        ]:
            try:
                m = factory(); m.fit(train)
                if label=="SARIMA": p=m.predict(len(test))
                else: p=m.predict(test)["forecast"]
                records.append({"Date":test["Date"].iloc[0],"Model":label,
                    "RMSE":np.sqrt(mean_squared_error(y_true,p))})
            except: pass
    if not records: return go.Figure()
    df_r = pd.DataFrame(records)
    fig = go.Figure()
    for model,col in [("SARIMA","#ef4444"),("Prophet Boost","#00c8ff")]:
        sub = df_r[df_r["Model"]==model]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["Date"],y=sub["RMSE"],mode="lines+markers",
                name=model,line=dict(color=col,width=2),marker=dict(size=5)))
    fig.update_layout(**BASE,height=380,
        title="Rolling 12-Month Walk-Forward RMSE (every 6 months)",yaxis_title="RMSE")
    return fig

def chart_subcategory(df_long, codes, periods=18):
    fig = go.Figure()
    palette = ["#00c8ff","#a855f7","#f97316","#22c55e","#ef4444"]
    for code, color in zip(codes, palette):
        sub = get_series(df_long, code)
        if len(sub)<30: continue
        try:
            m = ProphetBoostForecaster().fit(sub)
            future = generate_future_dates(sub["Date"].max(), periods)
            fc = m.predict(future)
            cat_name = df_long[df_long["Category_Code"]==code]["Category"].iloc[0][:30]
            fig.add_trace(go.Scatter(x=sub["Date"],y=sub["CPI"],mode="lines",
                name=f"{cat_name} (hist)",line=dict(color=color,width=1,dash="dot"),opacity=.4,showlegend=False))
            fig.add_trace(go.Scatter(x=future["Date"],y=fc["forecast"],mode="lines",
                name=cat_name,line=dict(color=color,width=2.5)))
        except: pass
    fig.update_layout(**BASE,height=460,title="Sub-Category CPI Forecast Comparison")
    return fig

def chart_metrics_bars(results):
    models = list(results.keys())
    pal = [COLORS.get(k,"#64748b") for k in models]
    fig = make_subplots(rows=2,cols=2,subplot_titles=["RMSE","MAE","MAPE (%)","MASE"],
                        vertical_spacing=.18,horizontal_spacing=.1)
    for idx,metric in enumerate(["RMSE","MAE","MAPE","MASE"]):
        r,c = idx//2+1, idx%2+1
        vals=[results[m][metric] for m in models]
        fig.add_trace(go.Bar(x=[m.replace("_"," ").title() for m in models],y=vals,
            marker_color=pal,text=[f"{v:.3f}" for v in vals],
            textposition="auto",showlegend=False),row=r,col=c)
    fig.update_layout(**BASE,height=600,title="Model Performance — Lower is Better",showlegend=False)
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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:.3rem 0 1.2rem'>
        <div style='font-family:Sora,sans-serif;font-size:1.25rem;font-weight:700;color:#fff'>
            Albania CPI<br><span style='color:#00c8ff'>Forecaster</span>
        </div>
        <div style='font-size:.68rem;color:#3a5070;letter-spacing:.08em;text-transform:uppercase;margin-top:.3rem'>
            SARIMA · Prophet · XGBoost
        </div>
    </div>""", unsafe_allow_html=True)

    # Add info about default data    
    uploaded = st.file_uploader("Upload your CPI Excel (.xlsx)", type=["xlsx","xls"])
    
    st.markdown("<hr style='border-color:#1a2340;margin:.8rem 0'>", unsafe_allow_html=True)
    train_ratio      = st.slider("Train ratio", .5, .9, .8, .05)
    forecast_periods = st.slider("Forecast horizon (months)", 12, 60, 36, 6)
    show_ci          = st.toggle("Show confidence intervals", value=True)
    auto_order       = st.toggle("Auto-select SARIMA order", value=False)
    st.markdown("<hr style='border-color:#1a2340;margin:.8rem 0'>", unsafe_allow_html=True)
    
    # THE BUTTON MUST BE HERE
    run_btn = st.button("▶  Train All Models", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>Albania <span>CPI</span> Forecasting Pipeline</h1>
    <p>Comparative analysis · SARIMA · Boosted SARIMA · Prophet · Prophet Boost</p>
    <span class="badge">Auto SARIMA</span><span class="badge">What-If Scenarios</span>
    <span class="badge">Feature Importance</span><span class="badge">Rolling Accuracy</span>
    <span class="badge">Sub-Category Compare</span><span class="badge">Excel Export</span>
</div>""", unsafe_allow_html=True)

pass

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CATEGORY SELECT
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_uploaded(raw):
    return clean_wide_to_long(pd.read_excel(io.BytesIO(raw), header=None))

@st.cache_data(show_spinner=False)
def load_default_data():
    """
    Load default CPI data from the data folder
    """
    filename = "instat1_20260319-113807.xlsx"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "data", filename)
    
    if os.path.exists(default_path):
        try:
            with open(default_path, "rb") as f:
                return clean_wide_to_long(pd.read_excel(f, header=None))
        except Exception as e:
            st.error(f"Error loading default data: {e}")
            return None
    else:
        st.warning(f"Default file not found at: {default_path}")
        return None

with st.spinner("Loading data…"):
    if uploaded is not None:
        df_long = load_uploaded(uploaded.read())
        st.success("✅ Custom dataset loaded successfully!")
    else:
        df_long = load_default_data()
        if df_long is not None:
            st.info("📊 Using default INSTAT Albania CPI dataset. Upload your own file to override.")
        else:
            st.error("❌ No data available. Please upload a file.")
            st.stop()

all_codes = sorted(df_long["Category_Code"].unique().tolist())
cat_labels = {}
for c in all_codes:
    name = df_long[df_long["Category_Code"]==c]["Category"].iloc[0]
    cat_labels[c] = f"{c} — {name[:45]}"

selected_code = st.selectbox("**Select CPI category to analyse**",
    options=all_codes,
    format_func=lambda c: cat_labels[c],
    index=all_codes.index("000000") if "000000" in all_codes else 0)

df_cpi = get_series(df_long, selected_code)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Observations", len(df_cpi))
c2.metric("Date range", f"{df_cpi['Date'].min().year}–{df_cpi['Date'].max().year}")
c3.metric("CPI start",  f"{df_cpi['CPI'].iloc[0]:.2f}")
c4.metric("CPI latest", f"{df_cpi['CPI'].iloc[-1]:.2f}",
          delta=f"{(df_cpi['CPI'].iloc[-1]/df_cpi['CPI'].iloc[0]-1)*100:+.1f}% total")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Historical","🔬 Decomposition","⚙️ Training",
                "📊 Results","🔮 Forecast","🎯 What-If",
                "🔍 Diagnostics","🔄 Rolling","📦 Sub-Categories"])

# ── 0 · Historical ────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">CPI Trend & Year-over-Year Inflation</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_trend(df_cpi), use_container_width=True)
    st.plotly_chart(chart_yoy(df_cpi),   use_container_width=True)

    st.markdown('<div class="section-title">ADF Stationarity Test</div>', unsafe_allow_html=True)
    adf_res = adfuller(df_cpi["CPI"].dropna())
    is_stat = adf_res[1] <= .05
    a1,a2,a3 = st.columns(3)
    a1.metric("ADF Statistic", f"{adf_res[0]:.4f}")
    a2.metric("p-value",       f"{adf_res[1]:.4f}")
    a3.metric("Result", "✅ Stationary" if is_stat else "⚠️ Non-Stationary")
    st.markdown(f"""<div class="{'info-card' if is_stat else 'warn-card'}">
    {"Stationary — models apply directly." if is_stat else
     "Non-stationary (expected for CPI). SARIMA uses d=1 differencing; Prophet fits a trend component. Both handle this automatically."}
    </div>""", unsafe_allow_html=True)
    with st.expander("View raw data"):
        st.dataframe(df_cpi.style.format({"CPI":"{:.2f}"}), use_container_width=True)
        st.download_button("⬇ Download CSV", df_cpi.to_csv(index=False), "cpi_data.csv", "text/csv")

# ── 1 · Decomposition ─────────────────────────────────────────────────────────
with tabs[1]:
    st.plotly_chart(chart_decomp(df_cpi), use_container_width=True)
    st.markdown("""<div class="info-card">
    <strong>Additive decomposition:</strong> Y(t) = Trend(t) + Seasonal(t) + Residual(t).
    Seasonal pattern repeats every 12 months. Small residuals = good model fit.
    </div>""", unsafe_allow_html=True)

# ── 2 · Training ──────────────────────────────────────────────────────────────
with tabs[2]:
    train_df, test_df = chronological_split(df_cpi, train_ratio)
    st.plotly_chart(chart_split(train_df, test_df), use_container_width=True)
    t1,t2,t3 = st.columns(3)
    t1.metric("Train obs", len(train_df), f"until {train_df['Date'].max().strftime('%b %Y')}")
    t2.metric("Test obs",  len(test_df),  f"from  {test_df['Date'].min().strftime('%b %Y')}")
    t3.metric("Split date", test_df["Date"].min().strftime("%Y-%m"))

    if not run_btn:
        st.info("👈 Press **Train All Models** in the sidebar to start.")
    else:
        pbar = st.progress(0, "Starting…")

        if auto_order:
            with st.spinner("Auto-selecting SARIMA order via AIC grid search…"):
                best_order, best_seasonal, best_aic = auto_sarima(train_df["CPI"])
            st.success(f"Best SARIMA order: {best_order}{best_seasonal} — AIC {best_aic:.2f}")
        else:
            best_order, best_seasonal = (0,1,0),(1,1,2,12)

        preds = {}; fitted_models = {}

        pbar.progress(10, "Fitting SARIMA…")
        m1 = SARIMAForecaster(best_order, best_seasonal).fit(train_df)
        preds["sarima"] = m1.predict(len(test_df))
        fitted_models["sarima"] = m1

        pbar.progress(30, "Fitting Boosted SARIMA…")
        m2 = BoostedSARIMA().fit(train_df)
        preds["boosted_sarima"] = m2.predict(test_df)
        fitted_models["boosted_sarima"] = m2

        pbar.progress(55, "Fitting Prophet…")
        m3 = ProphetForecaster(growth="linear",yearly_seasonality=True,weekly_seasonality=False,
            daily_seasonality=False,n_changepoints=25,changepoint_range=.8,
            changepoint_prior_scale=.05,seasonality_prior_scale=10.)
        m3.fit(train_df)
        preds["prophet"] = m3.predict(test_df)["forecast"]
        fitted_models["prophet"] = m3

        pbar.progress(80, "Fitting Prophet Boost…")
        m4 = ProphetBoostForecaster().fit(train_df)
        preds["prophet_boost"] = m4.predict(test_df)["forecast"]
        fitted_models["prophet_boost"] = m4

        pbar.progress(100, "Done ✓"); pbar.empty()

        y_true  = test_df["CPI"].values
        y_train = train_df["CPI"].values
        results = {k: calc_metrics(y_true, v, y_train) for k,v in preds.items()}
        best_key = min(results, key=lambda k: results[k]["RMSE"])

        st.session_state.update({
            "train_df":train_df,"test_df":test_df,"preds":preds,
            "results":results,"best_key":best_key,"fitted_models":fitted_models,
            "sarima_order":best_order,"sarima_seasonal":best_seasonal,
            "models_trained":True
        })
        st.success(f"✅ All models trained! Best by RMSE: **{best_key.replace('_',' ').title()}**")

    if st.session_state.get("models_trained"):
        st.plotly_chart(chart_model_comparison(
            st.session_state["train_df"],st.session_state["test_df"],
            st.session_state["preds"]), use_container_width=True)

# ── 3 · Results ───────────────────────────────────────────────────────────────
with tabs[3]:
    if not st.session_state.get("models_trained"):
        st.warning("Train models first (sidebar button).")
    else:
        results  = st.session_state["results"]
        best_key = st.session_state["best_key"]
        cmp = pd.DataFrame(results).T.round(4)
        cmp.index = [i.replace("_"," ").title() for i in cmp.index]

        st.markdown(f'**Winner by RMSE:** {best_key.replace("_"," ").title()} <span class="winner-tag">★ BEST</span>', unsafe_allow_html=True)
        st.dataframe(cmp.style.format("{:.4f}").highlight_min(axis=0,color="#d1fae5"),
                     use_container_width=True,height=210)
        st.plotly_chart(chart_metrics_bars(results), use_container_width=True)

        st.markdown('<div class="section-title">XGBoost Feature Importance</div>', unsafe_allow_html=True)
        fi1, fi2 = st.columns(2)
        for widget, key in zip([fi1,fi2],["boosted_sarima","prophet_boost"]):
            m = st.session_state["fitted_models"].get(key)
            if m and hasattr(m,"feature_importances"):
                widget.plotly_chart(chart_feature_importance(
                    m.feature_importances(),key.replace("_"," ").title()),use_container_width=True)

        test_preds_df = pd.DataFrame({
            "Date": st.session_state["test_df"]["Date"],
            "Actual": st.session_state["test_df"]["CPI"].values,
            **{k.replace("_"," ").title():v for k,v in st.session_state["preds"].items()}
        })
        st.download_button("⬇ Download Results Excel",
            build_excel(results, pd.DataFrame(), df_cpi, test_preds_df),
            "cpi_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

# ── 4 · Forecast ──────────────────────────────────────────────────────────────
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
                    fc_result={"forecast":raw,"lower":raw-1.96*std,"upper":raw+1.96*std}
                elif best_key == "boosted_sarima":
                    fm = BoostedSARIMA().fit(df_cpi)
                    raw = fm.predict(future_df)
                    std = float(np.std(df_cpi["CPI"].diff().dropna()))
                    fc_result={"forecast":raw,"lower":raw-1.96*std,"upper":raw+1.96*std}
                elif best_key == "prophet":
                    fm = ProphetForecaster(growth="linear",yearly_seasonality=True,
                        weekly_seasonality=False,daily_seasonality=False,
                        n_changepoints=25,changepoint_range=.8,
                        changepoint_prior_scale=.05,seasonality_prior_scale=10.).fit(df_cpi)
                    pr=fm.predict(future_df); fc_result={"forecast":pr["forecast"],"lower":pr["lower"],"upper":pr["upper"]}
                else:
                    fm = ProphetBoostForecaster().fit(df_cpi)
                    pb=fm.predict(future_df); fc_result={"forecast":pb["forecast"],"lower":pb["lower"],"upper":pb["upper"]}

                fc_df = pd.DataFrame({"Date":future_df["Date"],"Forecast":fc_result["forecast"],
                    "Lower":fc_result["lower"],"Upper":fc_result["upper"]})
                st.session_state.update({"fc_df":fc_df,"fc_model":best_key,"forecast_done":True})

        if st.session_state.get("forecast_done"):
            fc_df     = st.session_state["fc_df"]
            fc_model  = st.session_state["fc_model"].replace("_"," ").title()
            st.plotly_chart(chart_forecast(df_cpi,fc_df,fc_model,show_ci), use_container_width=True)
            last=df_cpi["CPI"].iloc[-1]; end=fc_df["Forecast"].iloc[-1]
            f1,f2,f3,f4 = st.columns(4)
            f1.metric("Current CPI",    f"{last:.2f}")
            f2.metric("End of forecast",f"{end:.2f}",  delta=f"{end-last:+.2f}")
            f3.metric("Total growth",   f"{(end/last-1)*100:+.2f}%")
            f4.metric("Model used",     fc_model)
            st.markdown("#### Forecast Table")
            disp = fc_df.copy(); disp["Date"]=disp["Date"].dt.strftime("%Y-%m")
            st.dataframe(disp.round(2), use_container_width=True, height=360)

            test_preds_df = pd.DataFrame({
                "Date":st.session_state["test_df"]["Date"],
                "Actual":st.session_state["test_df"]["CPI"].values,
                **{k.replace("_"," ").title():v for k,v in st.session_state["preds"].items()}
            })
            st.download_button("⬇ Download Full Results Excel",
                build_excel(st.session_state["results"],fc_df,df_cpi,test_preds_df),
                "albania_cpi_full_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

# ── 5 · What-If ───────────────────────────────────────────────────────────────
with tabs[5]:
    if not st.session_state.get("forecast_done"):
        st.warning("Generate a forecast first (Forecast tab).")
    else:
        st.markdown('<div class="section-title">Inflation Shock Simulator</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        Simulate an unexpected inflation shock by shifting the entire forecast path.
        Positive = inflation surge (e.g. energy crisis), Negative = deflation.
        </div>""", unsafe_allow_html=True)
        shock = st.slider("Shock magnitude (%)", -20.0, 30.0, 0.0, 0.5)
        st.plotly_chart(chart_whatif(df_cpi,st.session_state["fc_df"],shock), use_container_width=True)
        base_end = st.session_state["fc_df"]["Forecast"].iloc[-1]
        shocked_end = base_end*(1+shock/100)
        s1,s2,s3 = st.columns(3)
        s1.metric("Base end CPI",    f"{base_end:.2f}")
        s2.metric("Shocked end CPI", f"{shocked_end:.2f}", delta=f"{shocked_end-base_end:+.2f}")
        s3.metric("Shock applied",   f"{shock:+.1f}%")

# ── 6 · Diagnostics ───────────────────────────────────────────────────────────
with tabs[6]:
    if not st.session_state.get("models_trained"):
        st.warning("Train models first.")
    else:
        st.markdown('<div class="section-title">Residual Diagnostics — ACF & PACF</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        Red dashed lines = 95% significance bounds.
        Bars outside bounds suggest autocorrelation the model hasn't fully captured.
        </div>""", unsafe_allow_html=True)
        y_true = st.session_state["test_df"]["CPI"].values
        for key,col in COLORS.items():
            if key in st.session_state["preds"]:
                st.plotly_chart(chart_residuals(
                    st.session_state["test_df"],y_true,
                    st.session_state["preds"][key],
                    key.replace("_"," ").title(),col), use_container_width=True)

# ── 7 · Rolling Accuracy ─────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="section-title">Walk-Forward Rolling Accuracy</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Retrains SARIMA and Prophet Boost every 6 months, forecasts the next 12 months each time.
    Shows how each model would have performed in real deployment over the full history.
    Takes 2–3 minutes.
    </div>""", unsafe_allow_html=True)
    if st.button("▶ Run Rolling Evaluation", use_container_width=True):
        with st.spinner("Running walk-forward evaluation…"):
            fig_roll = chart_rolling(df_cpi, window=48, step=6)
        st.plotly_chart(fig_roll, use_container_width=True)

# ── 8 · Sub-Category Comparison ──────────────────────────────────────────────
with tabs[8]:
    st.markdown('<div class="section-title">Sub-Category CPI Forecast Comparison</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    Forecast multiple CPI sub-categories side-by-side using Prophet Boost.
    Identifies which sectors are driving overall inflation.
    </div>""", unsafe_allow_html=True)
    available = [c for c in all_codes if len(get_series(df_long,c))>=30]
    selected_cats = st.multiselect("Choose categories (max 5)",
        options=available, format_func=lambda c: cat_labels[c],
        default=available[:3] if len(available)>=3 else available,
        max_selections=5)
    sub_periods = st.slider("Sub-category forecast horizon (months)", 6, 36, 18, 6)
    if st.button("▶ Forecast Sub-Categories", use_container_width=True):
        if not selected_cats:
            st.warning("Select at least one category.")
        else:
            with st.spinner("Forecasting sub-categories with Prophet Boost…"):
                st.plotly_chart(chart_subcategory(df_long,selected_cats,sub_periods), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 .5rem;font-size:.75rem;color:#94a3b8'>
Albania CPI Forecasting · SARIMA · Prophet · XGBoost
</div>""", unsafe_allow_html=True)