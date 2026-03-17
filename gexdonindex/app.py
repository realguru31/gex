"""
GEXdon Index — Real-time GEX Dashboard for SPY, QQQ, SPX, NDX
"""
import os, sys, time, shutil, pickle, math
import json as json_mod
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dtime
import pytz

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config import (
    NYSE_TZ, MARKET_OPEN, MARKET_CLOSE,
    SNAPSHOT_INTERVAL_MIN, AUTO_REFRESH_SEC,
    TICKERS, TICKER_DISPLAY, STRIKE_RANGES, STRIKE_RANGE_LABELS,
    DTE_OPTIONS, DTE_LABELS, PERIOD_LABELS, CS, SNAPSHOT_DIR,
)
from gex.gex_utils import fetch_options_data, fetch_price_data, build_diagnostic_table, compute_net_vex
from gex.gex_chart_1 import generate_gex_chart
from gex.gex_chart_2 import generate_charm_chart
from gex.gex_chart_3 import generate_pressure_chart

st.set_page_config(page_title="GEXdon Index", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

def load_css():
    p = os.path.join(APP_DIR, "theme", "dark_blue.css")
    if os.path.exists(p):
        with open(p) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── Time Utilities ──
def now_et(): return datetime.now(NYSE_TZ)
def is_market_hours(dt_et=None):
    if dt_et is None: dt_et = now_et()
    t = dt_et.time()
    return dt_et.weekday() < 5 and MARKET_OPEN <= t <= MARKET_CLOSE
def current_bucket(dt_et=None):
    if dt_et is None: dt_et = now_et()
    minute = (dt_et.minute // SNAPSHOT_INTERVAL_MIN) * SNAPSHOT_INTERVAL_MIN
    return dt_et.replace(minute=minute, second=0, microsecond=0)
def hhmm_to_period_label(hhmm_str):
    letter = PERIOD_LABELS.get(hhmm_str, "?")
    return f"{letter} ({hhmm_str[:2]}:{hhmm_str[2:]})"

ALL_PERIODS = [
    ("0930","A"),("1000","B"),("1030","C"),("1100","D"),("1130","E"),
    ("1200","F"),("1230","G"),("1300","H"),("1330","I"),("1400","J"),
    ("1430","K"),("1500","L"),("1530","M"),("1600","N"),
]

# ── Snapshot Engine (JSON) ──
def snapshot_dir(ticker, date_str=None):
    if date_str is None: date_str = now_et().strftime("%Y-%m-%d")
    path = os.path.join(APP_DIR, SNAPSHOT_DIR, ticker.replace("^","_"), date_str)
    try: os.makedirs(path, exist_ok=True)
    except OSError: pass
    return path
def snapshot_filename(bucket_time):
    return (bucket_time.strftime("%H%M") if isinstance(bucket_time, datetime) else str(bucket_time)) + ".json"

def _snap_to_json(snap):
    out = {}
    for k, v in snap.items():
        if isinstance(v, pd.DataFrame):
            d = v.copy()
            if not isinstance(d.index, pd.RangeIndex): d = d.reset_index()
            out[k] = {"__df__": True, "data": d.to_dict(orient="list")}
        elif hasattr(v, "to_plotly_json"):
            out[k] = {"__plotly__": True, "data": v.to_plotly_json()}
        elif isinstance(v, (np.integer,)): out[k] = int(v)
        elif isinstance(v, (np.floating,)): out[k] = float(v)
        elif isinstance(v, np.ndarray): out[k] = v.tolist()
        else: out[k] = v
    return out

def _snap_from_json(raw):
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and v.get("__df__"):
            df = pd.DataFrame(v["data"])
            if k == "price_data" and "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            out[k] = df
        elif isinstance(v, dict) and v.get("__plotly__"):
            out[k] = go.Figure(v["data"])
        else: out[k] = v
    return out

def save_snapshot(ticker, bucket_dt, snap):
    sdir = snapshot_dir(ticker)
    fpath = os.path.join(sdir, snapshot_filename(bucket_dt))
    if os.path.exists(fpath): return False
    with open(fpath, "w") as f:
        json_mod.dump(_snap_to_json(snap), f, default=str)
    return True

def load_snapshot(ticker, bucket_dt, date_str=None):
    sdir = snapshot_dir(ticker, date_str)
    fpath = os.path.join(sdir, snapshot_filename(bucket_dt))
    if not os.path.exists(fpath):
        pkl = fpath.replace(".json", ".pkl")
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f: return pickle.load(f)
            except: return None
        return None
    with open(fpath, "r") as f:
        return _snap_from_json(json_mod.load(f))

def list_snapshots(ticker, date_str=None):
    sdir = snapshot_dir(ticker, date_str)
    if not os.path.exists(sdir): return []
    files = sorted([f for f in os.listdir(sdir) if f.endswith((".json",".pkl"))])
    return [f.replace(".json","").replace(".pkl","") for f in files]

def delete_all_snapshots(ticker):
    base = os.path.join(APP_DIR, SNAPSHOT_DIR, ticker.replace("^","_"))
    if os.path.exists(base): shutil.rmtree(base); return True
    return False

# ── Regime Assessment ──
REGIME_TABLE = {
    ("spike","positive"):  ("Dealer buying cushions the fall",          "dodgerblue"),
    ("spike","negative"):  ("Dealer selling amplifies the fall",        "magenta"),
    ("crush","positive"):  ("Mild selling, controlled drift down",      "magenta"),
    ("crush","negative"):  ("Sharp relief rally (dealers buy)",         "dodgerblue"),
    ("neutral","positive"):("Balanced — positive VEX stabilizing",      CS["gold"]),
    ("neutral","negative"):("Balanced — negative VEX, watch for moves", CS["gold"]),
    ("neutral","transition"):("Spot near VEX flip — watch for acceleration", CS["gold"]),
    ("spike","transition"):("IV spike near VEX flip — high volatility",  CS["gold"]),
    ("crush","transition"):("IV crush near VEX flip — sharp move likely", CS["gold"]),
}

def _vex_zone(snap):
    vex = snap.get("vex_info",{})
    pct = vex.get("vex_pct", 0)
    if abs(pct) < 30:
        return "transition"
    return "positive" if pct > 0 else "negative"

def _vex_label(snap):
    """Return display label like '+67% VEX' or '-12% VEX (transition)'"""
    vex = snap.get("vex_info",{})
    pct = vex.get("vex_pct", 0)
    zone = _vex_zone(snap)
    if zone == "transition":
        return f"{pct:+.0f}% VEX (transition)"
    return f"{pct:+.0f}% VEX"

def _vex_label_from_model(m):
    """Return VEX label from model result dict."""
    pct = m.get("vex_pct", 0)
    zone = m.get("vex_zone", "neutral")
    if zone == "transition":
        return f"{pct:+.0f}% VEX (transition)"
    return f"{pct:+.0f}% VEX"

def assess_regime_model1(snap, view_date=None):
    vex = snap.get("vex_info"); 
    if not vex or not vex.get("atm_iv"): return None
    atm_iv = vex["atm_iv"]
    date_str = view_date or now_et().strftime("%Y-%m-%d")
    open_snap = load_snapshot(snap.get("ticker","SPX"), datetime.combine(datetime.today(), dtime(9,30)), date_str=date_str)
    if not open_snap: return None
    open_iv = (open_snap.get("vex_info") or {}).get("atm_iv")
    if not open_iv or open_iv <= 0: return None
    et = now_et()
    close_t = datetime.combine(et.date(), dtime(16,0), tzinfo=NYSE_TZ)
    open_t = datetime.combine(et.date(), dtime(9,30), tzinfo=NYSE_TZ)
    try:
        snap_et = datetime.fromisoformat(snap.get("timestamp",""))
        if snap_et.tzinfo is None: snap_et = NYSE_TZ.localize(snap_et)
    except: snap_et = et
    t_open = max(0.01, (close_t - open_t).total_seconds()/3600)
    t_now = max(0.01, (close_t - snap_et).total_seconds()/3600)
    expected = open_iv * math.sqrt(t_now / t_open)
    ratio = atm_iv / expected if expected > 0 else 1
    iv_state = "spike" if ratio > 1.10 else "crush" if ratio < 0.90 else "neutral"
    zone = _vex_zone(snap)
    beh, col = REGIME_TABLE.get((iv_state, zone), ("Unknown", CS["gold"]))
    return {"model":"Model 1 (Theta Decay)","iv_state":iv_state,"vex_zone":zone,"behavior":beh,"color":col,
            "atm_iv":atm_iv,"open_iv":open_iv,"expected_iv":expected,"net_vex":vex.get("net_vex",0),"vex_pct":vex.get("vex_pct",0)}

def assess_regime_model2(snap, prev_snap=None):
    vex = snap.get("vex_info")
    if not vex or not vex.get("atm_iv"): return None
    atm_iv = vex["atm_iv"]
    prev_iv = (prev_snap.get("vex_info") or {}).get("atm_iv") if prev_snap else None
    if prev_iv and prev_iv > 0:
        ratio = atm_iv / prev_iv
        iv_state = "spike" if ratio > 1.10 else "crush" if ratio < 0.90 else "neutral"
    else: iv_state = "neutral"
    zone = _vex_zone(snap)
    beh, col = REGIME_TABLE.get((iv_state, zone), ("Unknown", CS["gold"]))
    return {"model":"Model 2 (Periodic)","iv_state":iv_state,"vex_zone":zone,"behavior":beh,"color":col,
            "atm_iv":atm_iv,"prev_iv":prev_iv,"net_vex":vex.get("net_vex",0),"vex_pct":vex.get("vex_pct",0)}

# ── Compute GEX Snapshot ──
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex_snapshot(ticker, percent_range, max_dte):
    data = fetch_options_data(ticker, max_dte=max_dte)
    if data is None: return None
    fig1, lv1 = generate_gex_chart(data, percent_range)
    fig2, lv2 = generate_charm_chart(data, percent_range)
    fig3, lv3 = generate_pressure_chart(data, percent_range)
    all_levels = {}
    all_levels.update({f"gamma_{k}":v for k,v in lv1.items()})
    all_levels.update({f"charm_{k}":v for k,v in lv2.items()})
    all_levels.update({f"pressure_{k}":v for k,v in lv3.items()})
    diag_table = build_diagnostic_table(data, percent_range)
    vex_info = compute_net_vex(data, percent_range)
    price_df = fetch_price_data(ticker, n_bars=600)
    return {
        "timestamp": now_et().isoformat(), "ticker": ticker,
        "percent_range": percent_range, "max_dte": max_dte,
        "spot": data["spot"], "expiry": data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage": data["iv_coverage"], "fallback_iv": data["fallback_iv"],
        "source": data.get("source","barchart"),
        "fetched_expiries": data.get("fetched_expiries",[]),
        "fig1": fig1, "fig2": fig2, "fig3": fig3, "levels": all_levels,
        "calls": data["calls"], "puts": data["puts"],
        "price_data": price_df if price_df is not None and not price_df.empty else None,
        "vex_info": vex_info,
        "diag_table": diag_table, "diag_info": data.get("diagnostics",{}),
    }

# ── Price Chart ──
def create_price_chart(ticker, gex_levels=None, spot=None, stored_price_df=None):
    price_df = stored_price_df.copy() if stored_price_df is not None and not stored_price_df.empty else fetch_price_data(ticker, n_bars=600)
    fig = go.Figure()
    if price_df.empty:
        fig.add_annotation(text="No price data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=CS["text"], size=14))
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], height=1550)
        return fig
    if not isinstance(price_df.index, pd.DatetimeIndex):
        for cn in ["datetime","date","time","index"]:
            if cn in price_df.columns:
                price_df[cn] = pd.to_datetime(price_df[cn]); price_df = price_df.set_index(cn); break
    cm = {c.lower():c for c in ["Open","High","Low","Close","Volume"]}
    price_df = price_df.rename(columns={k:v for k,v in cm.items() if k in price_df.columns and v not in price_df.columns})
    if isinstance(price_df.index, pd.DatetimeIndex):
        if price_df.index.tz is None:
            try: price_df.index = price_df.index.tz_localize("UTC")
            except: pass
        try: price_df.index = price_df.index.tz_convert(NYSE_TZ)
        except: pass
    else:
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], height=1550)
        return fig
    today_et = now_et().date()
    avail = sorted(set(price_df.index.date))
    target = today_et if today_et in avail else (avail[-1] if avail else today_et)
    day_df = price_df[price_df.index.date == target]
    rth = day_df[(day_df.index.time >= dtime(9,30)) & (day_df.index.time <= dtime(16,0))]
    if rth.empty: rth = day_df
    if rth.empty:
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], height=1550)
        return fig
    tl = [t.strftime("%H:%M") for t in rth.index]
    full = []; h,m = 9,30
    while h < 16 or (h==16 and m==0): full.append(f"{h:02d}:{m:02d}"); m+=5; h,m = (h+1,0) if m>=60 else (h,m)
    fig.add_trace(go.Candlestick(x=tl, open=rth["Open"], high=rth["High"], low=rth["Low"], close=rth["Close"],
        increasing=dict(line=dict(color="dodgerblue"),fillcolor="dodgerblue"),
        decreasing=dict(line=dict(color="magenta"),fillcolor="magenta"), name="Price"))
    if spot:
        fig.add_hline(y=spot, line=dict(color=CS["cyan"],width=1.5), annotation_text=f"{spot:.0f}",
                      annotation_font_color=CS["cyan"], annotation_position="right")
    if gex_levels:
        ls = {"gamma_k_star":("K*","magenta","dot"),"gamma_forward_price":("Fwd F","#ffffff","dashdot"),
              "gamma_max_gamma":("Max Γ",CS["gold"],"solid"),"gamma_zero_gamma":("Γ Flip","darkgoldenrod","dash"),
              "gamma_call_wall":("Call Wall",CS["green"],"dot"),"gamma_put_wall":("Put Wall",CS["red"],"dot"),
              "pressure_pressure_eq":("P.Eq",CS["purple"],"dash"),"charm_max_charm_strike":("Max Charm",CS["orange"],"dot"),
              "pressure_max_pressure_strike":("Max P","#e040fb","dashdot"),"pressure_max_accel_strike":("Accel","#76ff03","dashdot")}
        for key,(lbl,clr,dsh) in ls.items():
            if key in gex_levels:
                v = gex_levels[key]
                fig.add_hline(y=v, line=dict(color=clr,width=1.2,dash=dsh), annotation_text=f"{lbl} {v:.0f}",
                              annotation_font_color=clr, annotation_font_size=9, annotation_position="left")
    dt = TICKER_DISPLAY.get(ticker, ticker)
    fig.update_layout(template="plotly_dark", title=dict(text=f"{dt} RTH", font=dict(color=CS["text"],size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], font=dict(color=CS["text"],size=10),
        xaxis=dict(gridcolor=CS["grid"], rangeslider_visible=False, categoryorder="array", categoryarray=full,
                   range=[-0.5,len(full)-0.5], nticks=14),
        yaxis=dict(gridcolor=CS["grid"], title="Price", tickformat=".0f"),
        margin=dict(l=60,r=10,t=35,b=30), height=1550, showlegend=False)
    return fig

# ── Net OI / Volume Charts ──
def create_positions_chart(calls_df, puts_df, spot, pct=0.03, ticker="SPY"):
    fig = go.Figure()
    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        fig.update_layout(paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],height=500); return fig
    rng = spot*pct
    co = calls_df[(calls_df["strikePrice"]>=spot-rng)&(calls_df["strikePrice"]<=spot+rng)].groupby("strikePrice")["openInterest"].sum().reset_index(); co.columns=["strike","c"]
    po = puts_df[(puts_df["strikePrice"]>=spot-rng)&(puts_df["strikePrice"]<=spot+rng)].groupby("strikePrice")["openInterest"].sum().reset_index(); po.columns=["strike","p"]
    oi = pd.merge(co,po,on="strike",how="outer").fillna(0).sort_values("strike"); oi["net"]=oi["c"]-oi["p"]
    if len(oi)<1: fig.update_layout(paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],height=500); return fig
    dtick = 5 if ticker in ("SPX","NDX") else 1
    colors = ["dodgerblue" if v>=0 else CS["orange"] for v in oi["net"]]
    fig.add_trace(go.Bar(y=oi["strike"],x=oi["net"],orientation="h",marker_color=colors,opacity=0.85))
    fig.add_hline(y=spot,line=dict(color=CS["gold"],width=3),annotation_text=f"Spot {spot:.0f}",annotation_font_color=CS["gold"],annotation_font_size=10,annotation_position="top right")
    fig.update_layout(template="plotly_dark",title=dict(text="Net OI by Strike (C-P)",font=dict(color=CS["text"],size=12)),
        paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],font=dict(color=CS["text"],size=9),
        xaxis=dict(gridcolor=CS["grid"],title="Net OI",zeroline=True,zerolinecolor=CS["text"]),
        yaxis=dict(gridcolor=CS["grid"],tickformat=".0f",dtick=dtick),showlegend=False,margin=dict(l=55,r=10,t=35,b=25),height=500,bargap=0.15)
    return fig

def create_volume_chart(calls_df, puts_df, spot, pct=0.03, ticker="SPY"):
    fig = go.Figure()
    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        fig.update_layout(paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],height=500); return fig
    rng = spot*pct
    cv = calls_df[(calls_df["strikePrice"]>=spot-rng)&(calls_df["strikePrice"]<=spot+rng)].groupby("strikePrice")["volume"].sum().reset_index(); cv.columns=["strike","c"]
    pv = puts_df[(puts_df["strikePrice"]>=spot-rng)&(puts_df["strikePrice"]<=spot+rng)].groupby("strikePrice")["volume"].sum().reset_index(); pv.columns=["strike","p"]
    vol = pd.merge(cv,pv,on="strike",how="outer").fillna(0).sort_values("strike"); vol["net"]=vol["c"]-vol["p"]
    if len(vol)<1: fig.update_layout(paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],height=500); return fig
    dtick = 5 if ticker in ("SPX","NDX") else 1
    colors = ["dodgerblue" if v>=0 else CS["orange"] for v in vol["net"]]
    fig.add_trace(go.Bar(y=vol["strike"],x=vol["net"],orientation="h",marker_color=colors,opacity=0.85))
    fig.add_hline(y=spot,line=dict(color=CS["gold"],width=3),annotation_text=f"Spot {spot:.0f}",annotation_font_color=CS["gold"],annotation_font_size=10,annotation_position="top right")
    fig.update_layout(template="plotly_dark",title=dict(text="Net Volume by Strike (C-P)",font=dict(color=CS["text"],size=12)),
        paper_bgcolor=CS["bg"],plot_bgcolor=CS["plot_bg"],font=dict(color=CS["text"],size=9),
        xaxis=dict(gridcolor=CS["grid"],title="Net Vol",zeroline=True,zerolinecolor=CS["text"]),
        yaxis=dict(gridcolor=CS["grid"],tickformat=".0f",dtick=dtick),showlegend=False,margin=dict(l=55,r=10,t=35,b=25),height=500,bargap=0.15)
    return fig


def create_oi_volume_combined(calls_df, puts_df, spot, pct=0.03, ticker="SPY"):
    """
    Combined OI + Volume chart — vertical bars, strike on X-axis.
    Side-by-side bars per strike (grouped, not overlaid).
    Call OI (faded lime) + Call Vol (dodgerblue) go UP.
    Put OI (faded orange) + Put Vol (magenta) go DOWN.
    """
    fig = go.Figure()
    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], height=300)
        return fig

    rng = spot * pct
    c_oi = calls_df[(calls_df["strikePrice"] >= spot-rng) & (calls_df["strikePrice"] <= spot+rng)].groupby("strikePrice")["openInterest"].sum()
    c_vol = calls_df[(calls_df["strikePrice"] >= spot-rng) & (calls_df["strikePrice"] <= spot+rng)].groupby("strikePrice")["volume"].sum()
    p_oi = puts_df[(puts_df["strikePrice"] >= spot-rng) & (puts_df["strikePrice"] <= spot+rng)].groupby("strikePrice")["openInterest"].sum()
    p_vol = puts_df[(puts_df["strikePrice"] >= spot-rng) & (puts_df["strikePrice"] <= spot+rng)].groupby("strikePrice")["volume"].sum()

    all_strikes = sorted(set(c_oi.index) | set(c_vol.index) | set(p_oi.index) | set(p_vol.index))
    if len(all_strikes) < 1:
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"], height=300)
        return fig

    c_oi = c_oi.reindex(all_strikes, fill_value=0)
    c_vol = c_vol.reindex(all_strikes, fill_value=0)
    p_oi = p_oi.reindex(all_strikes, fill_value=0)
    p_vol = p_vol.reindex(all_strikes, fill_value=0)

    # Call OI (faded lime green, UP)
    fig.add_trace(go.Bar(
        x=all_strikes, y=c_oi.values, name="Call OI",
        marker_color="rgba(255,255,255,0.9)",
    ))
    # Call Volume (dodgerblue, UP)
    fig.add_trace(go.Bar(
        x=all_strikes, y=c_vol.values, name="Call Vol",
        marker_color="rgba(30,144,255,0.85)",
    ))
    # Put OI (faded orange, DOWN)
    fig.add_trace(go.Bar(
        x=all_strikes, y=-p_oi.values, name="Put OI",
        marker_color="rgba(255,140,0,0.9)",
    ))
    # Put Volume (magenta, DOWN)
    fig.add_trace(go.Bar(
        x=all_strikes, y=-p_vol.values, name="Put Vol",
        marker_color="rgba(255,0,255,0.85)",
    ))

    # Spot vertical line
    fig.add_vline(x=spot, line=dict(color=CS["cyan"], width=2, dash="dash"),
                  annotation_text=f"Spot {spot:.0f}",
                  annotation_font_color=CS["cyan"],
                  annotation_font_size=9,
                  annotation_position="top")

    # Zero line
    fig.add_hline(y=0, line=dict(color=CS["text"], width=0.5))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Positioning — OI (faded) vs Volume (bright) · Calls ▲ Puts ▼",
                   font=dict(color=CS["text"], size=12)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=9),
        xaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat=".0f"),
        yaxis=dict(gridcolor=CS["grid"], title="Contracts", zeroline=True,
                   zerolinecolor=CS["text"], zerolinewidth=1),
        barmode="group",
        legend=dict(
            bgcolor="rgba(13,31,60,0.9)", bordercolor=CS["grid"],
            font=dict(size=9, color="#ffffff"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
        margin=dict(l=50, r=10, t=45, b=30),
        height=300,
        bargap=0.1,
        bargroupgap=0.0,
    )
    return fig


# ── Session State ──
def init_state():
    for k,v in {"ticker":"SPX","strike_range":0.03,"max_dte":0,"selected_period":"LIVE",
                 "last_refresh":0,"last_auto_refresh":0,"live_snapshot":None,"view_date":None}.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚡ GEXdon Index"); st.markdown("---")
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda x: TICKER_DISPLAY.get(x,x),
        index=TICKERS.index(st.session_state.ticker), key="ticker_select")
    if ticker != st.session_state.ticker:
        st.session_state.ticker = ticker; st.session_state.live_snapshot = None; st.session_state.selected_period = "LIVE"
    sr = st.selectbox("Strike Range", STRIKE_RANGES, format_func=lambda x: STRIKE_RANGE_LABELS[x],
        index=STRIKE_RANGES.index(st.session_state.strike_range), key="range_select")
    if sr != st.session_state.strike_range: st.session_state.strike_range = sr; st.session_state.live_snapshot = None
    md = st.selectbox("DTE Range", DTE_OPTIONS, format_func=lambda x: DTE_LABELS[x],
        index=DTE_OPTIONS.index(st.session_state.max_dte), key="dte_select")
    if md != st.session_state.max_dte: st.session_state.max_dte = md; st.session_state.live_snapshot = None
    st.markdown("---")
    if st.button("🔄 Refresh Now", use_container_width=True):
        compute_gex_snapshot.clear(); st.session_state.live_snapshot = None; st.session_state.last_refresh = time.time()
    et = now_et()
    st.markdown(f"**{'🟢 MARKET OPEN' if is_market_hours(et) else '🔴 MARKET CLOSED'}**")
    st.markdown(f"**ET:** {et.strftime('%H:%M:%S')}")
    st.markdown("---"); st.markdown("### 📁 Snapshots")
    tsd = os.path.join(APP_DIR, SNAPSHOT_DIR, st.session_state.ticker)
    avd = []
    if os.path.exists(tsd):
        avd = sorted([d for d in os.listdir(tsd) if os.path.isdir(os.path.join(tsd,d))
            and any(f.endswith((".json",".pkl")) for f in os.listdir(os.path.join(tsd,d)))], reverse=True)
    today_str = now_et().strftime("%Y-%m-%d")
    if avd:
        do = avd if today_str in avd else [today_str]+avd
        sd = st.selectbox("📅 Date", do, index=0, key="snapshot_date",
            format_func=lambda d: f"{d} (today)" if d==today_str else d)
        st.session_state.view_date = sd
        sh = list_snapshots(st.session_state.ticker, sd)
        st.markdown(f"{len(sh)} snapshots" if sh else "No snapshots")
    else:
        st.session_state.view_date = today_str
        st.markdown("No snapshots yet")
    if st.button("🗑️ Delete All Snapshots", use_container_width=True):
        if delete_all_snapshots(st.session_state.ticker):
            st.session_state.selected_period = "LIVE"; st.rerun()
    if is_market_hours(et) and time.time()-st.session_state.last_auto_refresh > AUTO_REFRESH_SEC:
        st.session_state.last_auto_refresh = time.time(); compute_gex_snapshot.clear(); st.session_state.live_snapshot = None

# ── Compute Live ──
def compute_live():
    with st.spinner("Fetching options data..."):
        snap = compute_gex_snapshot(st.session_state.ticker, st.session_state.strike_range, st.session_state.max_dte)
    if snap is None: st.error("❌ Failed to fetch options data."); return None
    st.session_state.live_snapshot = snap
    if is_market_hours():
        bucket = current_bucket()
        if save_snapshot(st.session_state.ticker, bucket, snap):
            st.toast(f"💾 Snapshot saved: {hhmm_to_period_label(bucket.strftime('%H%M'))}")
    return snap

# ══════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════
st.markdown("# 📊 GEXdon Index")
snap = st.session_state.live_snapshot
if snap is None: snap = compute_live()

if snap is not None:
    # Header
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ticker", TICKER_DISPLAY.get(snap["ticker"],snap["ticker"]))
    c2.metric("Spot", f"${snap['spot']:.2f}")
    c3.metric("Expiry", snap["expiry_label"])
    c4.metric("IV Coverage", f"{snap['iv_coverage']:.0f}%")
    c5.metric("Range", STRIKE_RANGE_LABELS.get(snap["percent_range"],"3%"))
    fe = snap.get("fetched_expiries",[])
    if fe: st.caption(f"📅 Chains: {', '.join(fe)}")

    # ── Period Buttons ──
    view_date = st.session_state.view_date or today_str
    snaps_hhmm = list_snapshots(st.session_state.ticker, view_date)
    avail_set = set(snaps_hhmm)
    is_today = (view_date == now_et().strftime("%Y-%m-%d"))
    st.markdown("---")
    bcols = st.columns(len(ALL_PERIODS)+1)
    with bcols[0]:
        if is_today:
            if st.button("LIVE", key="btn_live", use_container_width=True,
                         type="primary" if st.session_state.selected_period=="LIVE" else "secondary"):
                st.session_state.selected_period = "LIVE"; st.rerun()
        else: st.button("LIVE", key="btn_live", disabled=True, use_container_width=True)
    for i,(hhmm,letter) in enumerate(ALL_PERIODS):
        with bcols[i+1]:
            has = hhmm in avail_set; sel = st.session_state.selected_period==hhmm
            if has:
                if st.button(letter, key=f"btn_{hhmm}", use_container_width=True,
                             type="primary" if sel else "secondary"):
                    st.session_state.selected_period = hhmm; st.rerun()
            else: st.button(letter, key=f"btn_{hhmm}", disabled=True, use_container_width=True)

    # Load snapshot
    if st.session_state.selected_period != "LIVE":
        hhmm = st.session_state.selected_period
        h,m = int(hhmm[:2]),int(hhmm[2:])
        hs = load_snapshot(st.session_state.ticker, datetime.combine(datetime.today(),dtime(h,m)), date_str=view_date)
        if hs:
            snap = hs; ltr = PERIOD_LABELS.get(hhmm,"?")
            dl = f" ({view_date})" if not is_today else ""
            st.info(f"📷 Viewing: {ltr} ({hhmm[:2]}:{hhmm[2:]}){dl}")

    # ── Regime Banner ──
    vex = snap.get("vex_info")
    if vex:
        prev_snap = None
        cur = st.session_state.selected_period if st.session_state.selected_period != "LIVE" else None
        if cur and cur in snaps_hhmm:
            idx = snaps_hhmm.index(cur)
            if idx > 0:
                ph = snaps_hhmm[idx-1]
                prev_snap = load_snapshot(st.session_state.ticker, datetime.combine(datetime.today(),dtime(int(ph[:2]),int(ph[2:]))), date_str=view_date)
        m1 = assess_regime_model1(snap, view_date)
        m2 = assess_regime_model2(snap, prev_snap)
        st.markdown("---")
        rc1,rc2 = st.columns(2)
        with rc1:
            if m1:
                il = m1["iv_state"].upper(); vl = _vex_label_from_model(m1)
                st.markdown(f'<div style="padding:10px;border-radius:8px;border:1px solid {m1["color"]}33;background:{m1["color"]}22">'
                    f'<span style="color:{CS["text"]};font-size:12px;font-weight:bold">Model 1 (Theta Decay)</span><br>'
                    f'<span style="color:{m1["color"]};font-size:14px;font-weight:bold">● IV {il} · {vl} · {m1["behavior"]}</span><br>'
                    f'<span style="color:{CS["text"]};font-size:11px">ATM IV: {m1["atm_iv"]*100:.1f}% | Open: {m1["open_iv"]*100:.1f}% | Expected: {m1["expected_iv"]*100:.1f}% | VEX: {m1["vex_pct"]:+.0f}% ({m1["net_vex"]:+,.0f})</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:10px;border-radius:8px;border:1px solid {CS["grid"]};background:{CS["plot_bg"]}">'
                    f'<span style="color:{CS["text"]};font-size:12px">Model 1 (Theta Decay): Needs opening snapshot (A period)</span></div>', unsafe_allow_html=True)
        with rc2:
            if m2:
                il = m2["iv_state"].upper(); vl = _vex_label_from_model(m2)
                piv = f" | Prev: {m2['prev_iv']*100:.1f}%" if m2.get("prev_iv") else ""
                st.markdown(f'<div style="padding:10px;border-radius:8px;border:1px solid {m2["color"]}33;background:{m2["color"]}22">'
                    f'<span style="color:{CS["text"]};font-size:12px;font-weight:bold">Model 2 (Periodic)</span><br>'
                    f'<span style="color:{m2["color"]};font-size:14px;font-weight:bold">● IV {il} · {vl} · {m2["behavior"]}</span><br>'
                    f'<span style="color:{CS["text"]};font-size:11px">ATM IV: {m2["atm_iv"]*100:.1f}%{piv} | VEX: {m2["vex_pct"]:+.0f}% ({m2["net_vex"]:+,.0f})</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:10px;border-radius:8px;border:1px solid {CS["grid"]};background:{CS["plot_bg"]}">'
                    f'<span style="color:{CS["text"]};font-size:12px">Model 2 (Periodic): Needs previous snapshot</span></div>', unsafe_allow_html=True)

    # ── Charts (40/60 split) ──
    st.markdown("---")
    col_p, col_g = st.columns([0.40, 0.60])
    with col_p:
        sp = snap.get("price_data")
        pf = create_price_chart(st.session_state.ticker, gex_levels=snap.get("levels"), spot=snap["spot"],
            stored_price_df=sp if isinstance(sp, pd.DataFrame) else None)
        st.plotly_chart(pf, width="stretch", theme=None, key="price_chart")
    with col_g:
        st.plotly_chart(snap["fig1"], width="stretch", theme=None, key="gex_chart_1")
        st.plotly_chart(snap["fig2"], width="stretch", theme=None, key="gex_chart_2")
        st.plotly_chart(snap["fig3"], width="stretch", theme=None, key="gex_chart_3")

    # ── OI + Volume Combined Chart (full width, bottom) ──
    oi_vol_fig = create_oi_volume_combined(
        snap.get("calls"), snap.get("puts"), snap["spot"],
        snap["percent_range"], st.session_state.ticker)
    st.plotly_chart(oi_vol_fig, width="stretch", theme=None, key="oi_vol_combined")

    # ── Key Levels ──
    levels = snap.get("levels",{})
    if levels:
        with st.expander("📐 Key GEX Levels", expanded=False):
            lc1,lc2,lc3,lc4 = st.columns(4)
            with lc1:
                st.markdown("**Gamma / K***")
                for k,l,f in [("gamma_k_star","K*","{:.0f}"),("gamma_forward_price","Fwd F","{:.0f}"),
                              ("gamma_zero_gamma","Γ Flip","{:.0f}"),("gamma_max_gamma","Max Γ","{:.0f}")]:
                    if k in levels: st.markdown(f"{l}: **{f.format(levels[k])}**")
            with lc2:
                st.markdown("**Walls / Peaks**")
                for k,l,f in [("gamma_call_wall","Call Wall","{:.0f}"),("gamma_put_wall","Put Wall","{:.0f}"),
                              ("gamma_sell_peak_1","Sell Peak","{:.0f}"),("gamma_buy_peak_1","Buy Peak","{:.0f}")]:
                    if k in levels: st.markdown(f"{l}: **{f.format(levels[k])}**")
            with lc3:
                st.markdown("**Charm**")
                for k,l,f in [("charm_call_charm_exp","Call Exp","{:,.0f}"),("charm_put_charm_exp","Put Exp","{:,.0f}"),
                              ("charm_net_charm_exp","Net Exp","{:,.0f}"),("charm_max_charm_strike","Max Strike","{:.0f}")]:
                    if k in levels: st.markdown(f"{l}: **{f.format(levels[k])}**")
            with lc4:
                st.markdown("**Pressure / VEX**")
                for k,l,f in [("pressure_max_pressure_strike","Max P","{:.0f}"),("pressure_pressure_eq","Equilibrium","{:.0f}")]:
                    if k in levels: st.markdown(f"{l}: **{f.format(levels[k])}**")
                vx = snap.get("vex_info",{})
                if vx.get("net_vex") is not None: st.markdown(f"VEX: **{vx.get('vex_pct',0):+.0f}%** ({vx['net_vex']:+,.0f})")
                if vx.get("vex_flip") is not None: st.markdown(f"VEX Flip: **${vx['vex_flip']:.1f}**")

    # ── Diagnostics ──
    with st.expander("🔬 Raw Diagnostic Data", expanded=False):
        di = snap.get("diag_info",{}); dt_tbl = snap.get("diag_table",pd.DataFrame())
        dc1,dc2,dc3,dc4 = st.columns(4)
        dc1.metric("Strikes",di.get("unique_strikes","?")); dc2.metric("Calls",di.get("total_calls","?"))
        dc3.metric("Puts",di.get("total_puts","?")); sp_list=di.get("strike_spacings",[])
        dc4.metric("Spacing",", ".join(f"{s}" for s in sp_list) if sp_list else "?")
        if isinstance(dt_tbl, pd.DataFrame) and not dt_tbl.empty:
            st.dataframe(dt_tbl, width="stretch", height=400)

    # ── TradingView Lightweight Charts ──
    with st.expander("📈 TradingView Chart (Experimental)", expanded=False):
        spd = snap.get("price_data"); tvl = snap.get("levels",{}); sv = snap.get("spot",0)
        tv_ticker = TICKER_DISPLAY.get(snap.get("ticker","SPX"), snap.get("ticker","SPX"))
        if isinstance(spd, pd.DataFrame) and not spd.empty:
            pdf = spd.copy()
            if not isinstance(pdf.index, pd.DatetimeIndex):
                for cn in ["datetime","date","time","index"]:
                    if cn in pdf.columns: pdf[cn]=pd.to_datetime(pdf[cn]); pdf=pdf.set_index(cn); break
            cm = {c.lower():c for c in ["Open","High","Low","Close"]}
            pdf = pdf.rename(columns={k:v for k,v in cm.items() if k in pdf.columns and v not in pdf.columns})
            if isinstance(pdf.index, pd.DatetimeIndex):
                if pdf.index.tz is None:
                    try: pdf.index=pdf.index.tz_localize("UTC")
                    except: pass
                try: pdf.index=pdf.index.tz_convert(NYSE_TZ)
                except: pass

                # Filter extended hours: 4AM-8PM ET
                ext_pdf = pdf[(pdf.index.time >= dtime(4,0)) & (pdf.index.time <= dtime(20,0))]
                if ext_pdf.empty: ext_pdf = pdf

                cd=[{"time":int(ts.timestamp()),"open":float(r.get("Open",0)),"high":float(r.get("High",0)),
                     "low":float(r.get("Low",0)),"close":float(r.get("Close",0))} for ts,r in ext_pdf.iterrows()]

                # GEX level price lines
                pls=""
                for key,(lbl,clr) in {"gamma_k_star":("K*","magenta"),"gamma_forward_price":("Fwd F","#ffffff"),
                    "gamma_zero_gamma":("Γ Flip","darkgoldenrod"),"gamma_call_wall":("Call Wall","#00ff88"),
                    "gamma_put_wall":("Put Wall","#ff4757"),"pressure_pressure_eq":("P.Eq","#a855f7")}.items():
                    if key in tvl:
                        val=float(tvl[key])
                        pls+=f"series.createPriceLine({{price:{val},color:'{clr}',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:'{lbl} {val:.0f}'}});\n"
                if sv: pls+=f"series.createPriceLine({{price:{float(sv)},color:'#00d2ff',lineWidth:2,lineStyle:0,axisLabelVisible:true,title:'Spot {float(sv):.0f}'}});\n"

                # $250 increment horizontal grid lines (dim grey)
                prices_all = [r.get("Low",0) for _,r in ext_pdf.iterrows() if r.get("Low",0)>0] + [r.get("High",0) for _,r in ext_pdf.iterrows()]
                price_min = min(prices_all) if prices_all else 0
                price_max = max(prices_all) if prices_all else 0
                round_lines_js = ""
                for rl in range(int(price_min // 250) * 250, int(price_max) + 300, 250):
                    if price_min - 100 < rl < price_max + 100:
                        round_lines_js += f"series.createPriceLine({{price:{rl},color:'rgba(80,100,130,0.5)',lineWidth:1,lineStyle:0,axisLabelVisible:false,title:''}});\n"

                # RTH open/close markers on candles
                markers_data = []
                seen_dates = set()
                for ts in ext_pdf.index:
                    d = ts.date()
                    if d not in seen_dates:
                        seen_dates.add(d)
                        # Find closest candle to 9:30
                        day_candles = ext_pdf[ext_pdf.index.date == d]
                        for ct in day_candles.index:
                            if ct.hour == 9 and 25 <= ct.minute <= 34:
                                markers_data.append({"time":int(ct.timestamp()),"position":"aboveBar","color":"#00d2ff","shape":"circle","text":"Open"})
                                break
                        # Find closest candle to 16:00
                        for ct in day_candles.index:
                            if ct.hour == 15 and 55 <= ct.minute <= 59:
                                markers_data.append({"time":int(ct.timestamp()),"position":"belowBar","color":"#ff8c00","shape":"circle","text":"Close"})
                                break
                            elif ct.hour == 16 and ct.minute <= 5:
                                markers_data.append({"time":int(ct.timestamp()),"position":"belowBar","color":"#ff8c00","shape":"circle","text":"Close"})
                                break

                markers_js = f"series.setMarkers({json_mod.dumps(markers_data)});" if markers_data else ""

                cj=json_mod.dumps(cd)
                html=f"""
                <div id="tv-chart" style="width:100%;height:600px;position:relative;">
                    <div style="position:absolute;top:10px;left:15px;z-index:10;font-family:monospace;font-size:18px;font-weight:bold;color:rgba(200,214,229,0.5);pointer-events:none;">{tv_ticker}</div>
                </div>
                <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
                <script>
                const chart=LightweightCharts.createChart(document.getElementById('tv-chart'),{{
                    width:document.getElementById('tv-chart').clientWidth,
                    height:600,
                    layout:{{background:{{type:'solid',color:'#050b1e'}},textColor:'#c8d6e5'}},
                    grid:{{vertLines:{{visible:false}},horzLines:{{visible:false}}}},
                    crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
                    rightPriceScale:{{borderColor:'#1a3a5c',scaleMargins:{{top:0.05,bottom:0.05}}}},
                    timeScale:{{borderColor:'#1a3a5c',timeVisible:true,secondsVisible:false,rightOffset:40}},
                    handleScroll:{{vertTouchDrag:true}},
                    handleScale:{{axisPressedMouseMove:true,mouseWheel:true,pinch:true}},
                }});
                const series=chart.addCandlestickSeries({{
                    upColor:'#1e90ff',downColor:'#ff00ff',
                    borderUpColor:'#1e90ff',borderDownColor:'#ff00ff',
                    wickUpColor:'#1e90ff',wickDownColor:'#ff00ff',
                }});
                series.setData({cj});
                {round_lines_js}
                {pls}
                {markers_js}
                chart.timeScale().fitContent();
                new ResizeObserver(e=>chart.applyOptions({{width:e[0].contentRect.width}})).observe(document.getElementById('tv-chart'));
                </script>"""
                st.components.v1.html(html, height=620)
            else: st.info("Price data not available for TradingView chart")
        else: st.info("No stored price data — requires snapshot with price data")
else:
    st.warning("⏳ Waiting for data... Click 🔄 Refresh Now in the sidebar.")
