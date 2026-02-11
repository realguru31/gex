"""
GEXdon — Real-time GEX Dashboard
ZERO yfinance. Barchart (options+spot) + tvdatafeed (price chart).
"""
import os
import sys
import time
import shutil
import pickle
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
    DTE_OPTIONS, DTE_LABELS, PERIOD_LABELS,
    CS, SNAPSHOT_DIR,
)
from gex.gex_utils import fetch_options_data, fetch_price_data
from gex.gex_chart_1 import generate_gex_chart
from gex.gex_chart_2 import generate_charm_chart
from gex.gex_chart_3 import generate_pressure_chart


# ─────────────────────────────────────
# Page config
# ─────────────────────────────────────
st.set_page_config(
    page_title="GEXdon", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)


# ─────────────────────────────────────
# Inject dark blue theme
# ─────────────────────────────────────
def load_css():
    css_path = os.path.join(APP_DIR, "theme", "dark_blue.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ─────────────────────────────────────
# NYSE Time Utilities
# ─────────────────────────────────────
def now_et():
    return datetime.now(NYSE_TZ)

def is_market_hours(dt_et=None):
    if dt_et is None:
        dt_et = now_et()
    t = dt_et.time()
    return dt_et.weekday() < 5 and MARKET_OPEN <= t <= MARKET_CLOSE

def current_bucket(dt_et=None):
    if dt_et is None:
        dt_et = now_et()
    minute = (dt_et.minute // SNAPSHOT_INTERVAL_MIN) * SNAPSHOT_INTERVAL_MIN
    return dt_et.replace(minute=minute, second=0, microsecond=0)

def hhmm_to_period_label(hhmm_str):
    """Convert HHMM string to Market Profile period label like 'A (09:30)'."""
    letter = PERIOD_LABELS.get(hhmm_str, "?")
    h, m = hhmm_str[:2], hhmm_str[2:]
    return f"{letter} ({h}:{m})"


# ─────────────────────────────────────
# Snapshot Engine
# ─────────────────────────────────────
def snapshot_dir(ticker, date_str=None):
    if date_str is None:
        date_str = now_et().strftime("%Y-%m-%d")
    path = os.path.join(APP_DIR, SNAPSHOT_DIR, ticker.replace("^", "_"), date_str)
    os.makedirs(path, exist_ok=True)
    return path

def snapshot_filename(bucket_time):
    if isinstance(bucket_time, datetime):
        return bucket_time.strftime("%H%M") + ".pkl"
    return bucket_time.strftime("%H%M") + ".pkl"

def save_snapshot(ticker, bucket_dt, snapshot_data):
    sdir = snapshot_dir(ticker)
    fname = snapshot_filename(bucket_dt)
    fpath = os.path.join(sdir, fname)
    if os.path.exists(fpath):
        return False
    with open(fpath, "wb") as f:
        pickle.dump(snapshot_data, f)
    return True

def load_snapshot(ticker, bucket_dt, date_str=None):
    sdir = snapshot_dir(ticker, date_str)
    fname = snapshot_filename(bucket_dt)
    fpath = os.path.join(sdir, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        return pickle.load(f)

def list_snapshots(ticker, date_str=None):
    """List snapshot HHMM strings for a date, sorted."""
    sdir = snapshot_dir(ticker, date_str)
    if not os.path.exists(sdir):
        return []
    files = sorted([f for f in os.listdir(sdir) if f.endswith(".pkl")])
    return [f.replace(".pkl", "") for f in files]

def delete_all_snapshots(ticker):
    """Delete all snapshot directories for a ticker."""
    base = os.path.join(APP_DIR, SNAPSHOT_DIR, ticker.replace("^", "_"))
    if os.path.exists(base):
        shutil.rmtree(base)
        return True
    return False


# ─────────────────────────────────────
# Compute GEX Snapshot
# ─────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex_snapshot(ticker, percent_range, max_dte):
    """Fetch live data (multi-DTE) and compute all three charts + levels."""
    data = fetch_options_data(ticker, max_dte=max_dte)
    if data is None:
        return None

    fig1, levels1 = generate_gex_chart(data, percent_range)
    fig2, levels2 = generate_charm_chart(data, percent_range)
    fig3, levels3 = generate_pressure_chart(data, percent_range)

    all_levels = {}
    all_levels.update({f"gamma_{k}": v for k, v in levels1.items()})
    all_levels.update({f"charm_{k}": v for k, v in levels2.items()})
    all_levels.update({f"pressure_{k}": v for k, v in levels3.items()})

    return {
        "timestamp": now_et().isoformat(),
        "ticker": ticker,
        "percent_range": percent_range,
        "max_dte": max_dte,
        "spot": data["spot"],
        "expiry": data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage": data["iv_coverage"],
        "fallback_iv": data["fallback_iv"],
        "source": data.get("source", "barchart"),
        "fig1": fig1, "fig2": fig2, "fig3": fig3,
        "levels": all_levels,
        # Store OI data for positions chart
        "calls": data["calls"],
        "puts": data["puts"],
    }


# ─────────────────────────────────────
# Candlestick Price Chart (Plotly)
# ─────────────────────────────────────
def create_price_chart(ticker, gex_levels=None, spot=None):
    """Plotly candlestick: dodgerblue up, magenta down. RTH only (9:30-16:00 ET)."""
    price_df = fetch_price_data(ticker, n_bars=200)

    fig = go.Figure()

    if price_df.empty:
        fig.add_annotation(text="No price data\n(tvdatafeed unavailable)",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=CS["text"], size=14))
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                          height=600, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    # Convert index to ET timezone
    if price_df.index.tz is None:
        try:
            price_df.index = price_df.index.tz_localize("UTC")
        except Exception:
            pass
    try:
        price_df.index = price_df.index.tz_convert(NYSE_TZ)
    except Exception:
        pass

    # Get today's date in ET (or last available trading day)
    today_et = now_et().date()
    available_dates = sorted(set(price_df.index.date))
    if today_et in available_dates:
        target_date = today_et
    elif available_dates:
        target_date = available_dates[-1]
    else:
        target_date = today_et

    # Filter to target date only
    day_df = price_df[price_df.index.date == target_date].copy()

    # Filter RTH: 9:30 to 16:00
    rth_start = dtime(9, 30)
    rth_end = dtime(16, 0)
    rth_df = day_df[(day_df.index.time >= rth_start) & (day_df.index.time <= rth_end)]

    # If no RTH data yet (premarket only), show premarket
    if rth_df.empty:
        rth_df = day_df

    if rth_df.empty:
        fig.add_annotation(text="No RTH data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=CS["text"], size=14))
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                          height=600, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    # Format as HH:MM strings
    time_labels = [t.strftime("%H:%M") for t in rth_df.index]

    # Generate FULL RTH timeline (09:30 to 16:00 in 5-min slots)
    full_rth_labels = []
    h, m = 9, 30
    while (h < 16) or (h == 16 and m == 0):
        full_rth_labels.append(f"{h:02d}:{m:02d}")
        m += 5
        if m >= 60:
            m = 0
            h += 1

    fig.add_trace(go.Candlestick(
        x=time_labels,
        open=rth_df["Open"], high=rth_df["High"],
        low=rth_df["Low"], close=rth_df["Close"],
        increasing=dict(line=dict(color="dodgerblue"), fillcolor="dodgerblue"),
        decreasing=dict(line=dict(color="magenta"), fillcolor="magenta"),
        name="Price",
    ))

    # Spot line
    if spot:
        fig.add_hline(y=spot, line=dict(color=CS["cyan"], width=1.5),
                      annotation_text=f"${spot:.2f}",
                      annotation_font_color=CS["cyan"],
                      annotation_position="right")

    # Key GEX levels overlay
    if gex_levels:
        level_styles = {
            "gamma_max_gamma":           ("Max Γ",           CS["gold"],   "solid"),
            "gamma_zero_gamma":          ("Gamma Flip",      "#ffffff",    "dash"),
            "gamma_call_wall":           ("Call Wall",        CS["green"],  "dot"),
            "gamma_put_wall":            ("Put Wall",         CS["red"],    "dot"),
            "pressure_pressure_eq":      ("P.Eq",             CS["purple"], "dash"),
            "charm_max_charm_strike":    ("Max Charm",        CS["orange"], "dot"),
            "pressure_max_pressure_strike": ("Max Pressure",  "#e040fb",    "dashdot"),
            "pressure_max_accel_strike":    ("Max Acceleration","#76ff03",   "dashdot"),
        }
        for key, (label, color, dash) in level_styles.items():
            if key in gex_levels:
                val = gex_levels[key]
                fig.add_hline(y=val, line=dict(color=color, width=1.2, dash=dash),
                              annotation_text=f"{label} ${val:.0f}",
                              annotation_font_color=color,
                              annotation_font_size=9,
                              annotation_position="left")

    display_ticker = TICKER_DISPLAY.get(ticker, ticker)
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{display_ticker} RTH", font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=10),
        xaxis=dict(gridcolor=CS["grid"], rangeslider_visible=False,
                   categoryorder="array", categoryarray=full_rth_labels,
                   range=[-0.5, len(full_rth_labels) - 0.5],
                   nticks=14),
        yaxis=dict(gridcolor=CS["grid"], title="Price", tickformat="$,.2f"),
        margin=dict(l=60, r=10, t=35, b=30),
        height=600,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────
# Positions by Strike Chart (Plotly)
# ─────────────────────────────────────
def create_positions_chart(calls_df, puts_df, spot, percent_range=0.03):
    """
    Horizontal bar chart:
      Put OI  = orange bars going LEFT (negative x)
      Call OI = dodgerblue bars going RIGHT (positive x)
      Spot    = gold horizontal line
    Uses NUMERIC y-axis (strike prices) so spot line renders correctly.
    """
    fig = go.Figure()

    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        fig.add_annotation(text="No OI data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=CS["text"], size=14))
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                          height=500, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    rng = spot * percent_range

    # Aggregate OI by strike across all expiries — SEPARATELY
    call_oi = calls_df[
        (calls_df["strikePrice"] >= spot - rng) &
        (calls_df["strikePrice"] <= spot + rng)
    ].groupby("strikePrice")["openInterest"].sum().reset_index()
    call_oi.columns = ["strike", "call_oi"]

    put_oi = puts_df[
        (puts_df["strikePrice"] >= spot - rng) &
        (puts_df["strikePrice"] <= spot + rng)
    ].groupby("strikePrice")["openInterest"].sum().reset_index()
    put_oi.columns = ["strike", "put_oi"]

    # Merge — keep separate columns, no netting
    oi = pd.merge(call_oi, put_oi, on="strike", how="outer").fillna(0)
    oi = oi.sort_values("strike")

    if len(oi) < 1:
        fig.add_annotation(text="No OI in range", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=CS["text"], size=14))
        fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                          height=500, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    strikes = oi["strike"].values
    c_oi = oi["call_oi"].values.astype(float)
    p_oi = oi["put_oi"].values.astype(float)

    # Put OI as NEGATIVE (left), Call OI as POSITIVE (right)
    fig.add_trace(go.Bar(
        y=strikes,
        x=-p_oi,
        orientation="h",
        name="Put OI",
        marker_color=CS["orange"],
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=strikes,
        x=c_oi,
        orientation="h",
        name="Call OI",
        marker_color="dodgerblue",
        opacity=0.85,
    ))

    # Spot line — numeric y-axis, so hline works directly
    fig.add_hline(y=spot, line=dict(color=CS["gold"], width=3),
                  annotation_text=f"Spot ${spot:.2f}",
                  annotation_font_color=CS["gold"],
                  annotation_font_size=10,
                  annotation_position="top right")

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Positions by Strike", font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=9),
        xaxis=dict(gridcolor=CS["grid"], title="Open Interest",
                   zeroline=True, zerolinecolor=CS["text"], zerolinewidth=1),
        yaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat="$,.0f",
                   dtick=5),
        barmode="relative",
        legend=dict(bgcolor="rgba(13,31,60,0.9)", bordercolor=CS["grid"],
                    font=dict(size=10, color="#ffffff"), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=55, r=10, t=40, b=30),
        height=500,
        bargap=0.15,
    )
    return fig


# ─────────────────────────────────────
# Session State
# ─────────────────────────────────────
def init_state():
    defaults = {
        "ticker": "SPY",
        "strike_range": 0.03,
        "max_dte": 0,
        "slider_time": None,
        "last_refresh": 0,
        "last_auto_refresh": 0,
        "live_snapshot": None,
        "viewing_mode": "live",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────
# Sidebar
# ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ GEXdon")
    st.markdown("---")

    # Ticker (no ^SPX)
    ticker = st.selectbox(
        "Ticker", TICKERS,
        format_func=lambda x: TICKER_DISPLAY.get(x, x),
        index=TICKERS.index(st.session_state.ticker),
        key="ticker_select",
    )
    if ticker != st.session_state.ticker:
        st.session_state.ticker = ticker
        st.session_state.live_snapshot = None
        st.session_state.slider_time = None

    # Strike range
    strike_range = st.selectbox(
        "Strike Range", STRIKE_RANGES,
        format_func=lambda x: STRIKE_RANGE_LABELS[x],
        index=STRIKE_RANGES.index(st.session_state.strike_range),
        key="range_select",
    )
    if strike_range != st.session_state.strike_range:
        st.session_state.strike_range = strike_range
        st.session_state.live_snapshot = None

    # DTE selector
    max_dte = st.selectbox(
        "DTE Range", DTE_OPTIONS,
        format_func=lambda x: DTE_LABELS[x],
        index=DTE_OPTIONS.index(st.session_state.max_dte),
        key="dte_select",
    )
    if max_dte != st.session_state.max_dte:
        st.session_state.max_dte = max_dte
        st.session_state.live_snapshot = None

    st.markdown("---")

    # Refresh
    if st.button("🔄 Refresh Now", use_container_width=True):
        compute_gex_snapshot.clear()
        st.session_state.live_snapshot = None
        st.session_state.last_refresh = time.time()

    # Status
    et = now_et()
    market_status = "🟢 MARKET OPEN" if is_market_hours(et) else "🔴 MARKET CLOSED"
    st.markdown(f"**{market_status}**")
    st.markdown(f"**ET:** {et.strftime('%H:%M:%S')}")

    # Snapshots info
    st.markdown("---")
    st.markdown("### 📁 Snapshots")
    snaps_hhmm = list_snapshots(st.session_state.ticker)
    if snaps_hhmm:
        st.markdown(f"{len(snaps_hhmm)} snapshots today")
    else:
        st.markdown("No snapshots today")

    # Delete all snapshots button
    if st.button("🗑️ Delete All Snapshots", use_container_width=True):
        deleted = delete_all_snapshots(st.session_state.ticker)
        if deleted:
            st.toast(f"🗑️ All snapshots deleted for {st.session_state.ticker}")
            st.session_state.slider_time = None
            st.rerun()
        else:
            st.toast("No snapshots to delete")

    # Auto-refresh
    if is_market_hours(et):
        elapsed = time.time() - st.session_state.last_auto_refresh
        if elapsed > AUTO_REFRESH_SEC:
            st.session_state.last_auto_refresh = time.time()
            compute_gex_snapshot.clear()
            st.session_state.live_snapshot = None


# ─────────────────────────────────────
# Compute Live Data
# ─────────────────────────────────────
def compute_live():
    with st.spinner("Fetching from Barchart..."):
        snap = compute_gex_snapshot(
            st.session_state.ticker,
            st.session_state.strike_range,
            st.session_state.max_dte,
        )
    if snap is None:
        st.error("❌ Failed to fetch options data. Barchart may be unavailable.")
        return None

    st.session_state.live_snapshot = snap

    # Auto-save snapshot during market hours
    if is_market_hours():
        bucket = current_bucket()
        saved = save_snapshot(st.session_state.ticker, bucket, snap)
        if saved:
            hhmm = bucket.strftime("%H%M")
            label = hhmm_to_period_label(hhmm)
            st.toast(f"💾 Snapshot saved: {label}")

    return snap


# ─────────────────────────────────────
# Main Layout
# ─────────────────────────────────────
st.markdown("# 📊 GEXdon")

snap = st.session_state.live_snapshot
if snap is None:
    snap = compute_live()

if snap is not None:
    # Header metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ticker", TICKER_DISPLAY.get(snap["ticker"], snap["ticker"]))
    c2.metric("Spot", f"${snap['spot']:.2f}")
    c3.metric("Expiry", snap["expiry_label"])
    c4.metric("IV Coverage", f"{snap['iv_coverage']:.0f}%")
    c5.metric("Range", STRIKE_RANGE_LABELS.get(snap["percent_range"], "3%"))
    c6.metric("Source", snap.get("source", "barchart").upper())

    # Time slider with Market Profile period labels (A, B, C...)
    snaps_hhmm = list_snapshots(st.session_state.ticker)
    if snaps_hhmm:
        st.markdown("---")
        period_options = ["LIVE"] + [hhmm_to_period_label(h) for h in snaps_hhmm]
        selected = st.select_slider(
            "⏱ Period",
            options=period_options,
            value="LIVE",
            key="time_slider",
        )

        if selected != "LIVE":
            # Parse back: "A (09:30)" → "0930"
            idx = period_options.index(selected) - 1  # -1 for LIVE offset
            hhmm = snaps_hhmm[idx]
            h, m = int(hhmm[:2]), int(hhmm[2:])
            target_dt = datetime.combine(datetime.today(), dtime(h, m))
            hist_snap = load_snapshot(st.session_state.ticker, target_dt)
            if hist_snap:
                snap = hist_snap
                st.info(f"📷 Viewing snapshot: {selected}")

    # Two-column layout
    st.markdown("---")
    col_price, col_gex = st.columns([0.32, 0.68])

    with col_price:
        # Candlestick price chart
        price_fig = create_price_chart(
            st.session_state.ticker,
            gex_levels=snap.get("levels"),
            spot=snap["spot"],
        )
        st.plotly_chart(price_fig, width="stretch", theme=None, key="price_chart")

        # Positions by Strike chart (below price chart)
        pos_fig = create_positions_chart(
            snap.get("calls"), snap.get("puts"),
            snap["spot"], snap["percent_range"],
        )
        st.plotly_chart(pos_fig, width="stretch", theme=None, key="pos_chart")

    with col_gex:
        # Chart 1: Gamma Density
        st.plotly_chart(snap["fig1"], width="stretch", theme=None, key="gex_chart_1")
        # Chart 2: Charm Density
        st.plotly_chart(snap["fig2"], width="stretch", theme=None, key="gex_chart_2")
        # Chart 3: Charm Pressure & Acceleration
        st.plotly_chart(snap["fig3"], width="stretch", theme=None, key="gex_chart_3")

    # Key Levels Summary
    levels = snap.get("levels", {})
    if levels:
        with st.expander("📐 Key GEX Levels", expanded=False):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                st.markdown("**Gamma**")
                if "gamma_max_gamma" in levels:
                    st.markdown(f"Max Γ: **${levels['gamma_max_gamma']:.1f}**")
                if "gamma_zero_gamma" in levels:
                    st.markdown(f"Gamma Flip: **${levels['gamma_zero_gamma']:.1f}**")
                if "gamma_call_wall" in levels:
                    st.markdown(f"Call Wall: **${levels['gamma_call_wall']:.1f}**")
                if "gamma_put_wall" in levels:
                    st.markdown(f"Put Wall: **${levels['gamma_put_wall']:.1f}**")
            with lc2:
                st.markdown("**Charm**")
                if "charm_call_charm_exp" in levels:
                    st.markdown(f"Call Exp: **{levels['charm_call_charm_exp']:,.0f}**")
                if "charm_put_charm_exp" in levels:
                    st.markdown(f"Put Exp: **{levels['charm_put_charm_exp']:,.0f}**")
                if "charm_net_charm_exp" in levels:
                    st.markdown(f"Net Exp: **{levels['charm_net_charm_exp']:,.0f}**")
                if "charm_max_charm_strike" in levels:
                    st.markdown(f"Max Strike: **${levels['charm_max_charm_strike']:.1f}**")
            with lc3:
                st.markdown("**Pressure**")
                if "pressure_max_pressure_strike" in levels:
                    st.markdown(f"Max P: **${levels['pressure_max_pressure_strike']:.1f}**")
                if "pressure_max_accel_strike" in levels:
                    st.markdown(f"Max Accel: **${levels['pressure_max_accel_strike']:.1f}**")
                if "pressure_pressure_eq" in levels:
                    st.markdown(f"Equilibrium: **${levels['pressure_pressure_eq']:.1f}**")

else:
    st.warning("⏳ Waiting for data... Click 🔄 Refresh Now in the sidebar.")
