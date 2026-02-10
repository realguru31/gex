"""
GEXdon — Real-time GEX Dashboard
Main Streamlit application. ZERO yfinance dependency.
Data: Barchart (options+spot) + tvdatafeed (price chart).
"""
import os
import sys
import time
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
import pytz

# Ensure gexdon root is on path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config import (
    NYSE_TZ, MARKET_OPEN, MARKET_CLOSE,
    SNAPSHOT_INTERVAL_MIN, AUTO_REFRESH_SEC,
    TICKERS, TICKER_DISPLAY, STRIKE_RANGES, STRIKE_RANGE_LABELS,
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
    page_title="GEXdon",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
    wd = dt_et.weekday()
    return wd < 5 and MARKET_OPEN <= t <= MARKET_CLOSE

def current_bucket(dt_et=None):
    if dt_et is None:
        dt_et = now_et()
    minute = (dt_et.minute // SNAPSHOT_INTERVAL_MIN) * SNAPSHOT_INTERVAL_MIN
    return dt_et.replace(minute=minute, second=0, microsecond=0)

def bucket_label(dt_et):
    return dt_et.strftime("%H:%M ET")

def get_trading_buckets():
    buckets = []
    current = datetime.combine(datetime.today(), MARKET_OPEN)
    end = datetime.combine(datetime.today(), MARKET_CLOSE)
    while current <= end:
        buckets.append(current.time())
        current += timedelta(minutes=SNAPSHOT_INTERVAL_MIN)
    return buckets

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
    sdir = snapshot_dir(ticker, date_str)
    if not os.path.exists(sdir):
        return []
    files = sorted([f for f in os.listdir(sdir) if f.endswith(".pkl")])
    times = []
    for f in files:
        hhmm = f.replace(".pkl", "")
        try:
            t = dtime(int(hhmm[:2]), int(hhmm[2:]))
            times.append(t)
        except (ValueError, IndexError):
            continue
    return times

# ─────────────────────────────────────
# Compute & Package GEX Data
# ─────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex_snapshot(ticker, percent_range):
    data = fetch_options_data(ticker)
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
        "spot": data["spot"],
        "expiry": data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage": data["iv_coverage"],
        "fallback_iv": data["fallback_iv"],
        "source": data.get("source", "barchart"),
        "fig1": fig1,
        "fig2": fig2,
        "fig3": fig3,
        "levels": all_levels,
    }

# ─────────────────────────────────────
# Price Chart (tvdatafeed)
# ─────────────────────────────────────
def create_price_chart(ticker, gex_levels=None, spot=None):
    price_df = fetch_price_data(ticker)

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.patch.set_facecolor(CS["bg"])
    ax.set_facecolor(CS["plot_bg"])

    if price_df.empty:
        ax.text(0.5, 0.5, "No price data\n(tvdatafeed unavailable)",
                transform=ax.transAxes, ha="center", va="center",
                color=CS["text"], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    close = price_df["Close"].values
    high = price_df["High"].values
    low = price_df["Low"].values
    idx = range(len(close))

    ax.plot(idx, close, color=CS["cyan"], lw=1.5, alpha=0.9, zorder=3)
    ax.fill_between(idx, low, high, color=CS["blue"], alpha=0.08, zorder=1)

    if spot:
        ax.axhline(y=spot, color=CS["cyan"], ls="-", lw=1, alpha=0.4)
        ax.text(len(close) - 1, spot, f" ${spot:.2f}", color=CS["cyan"],
                fontsize=8, va="bottom", fontweight="bold")

    if gex_levels:
        level_styles = {
            "gamma_max_gamma":             ("Max Γ",     CS["gold"],   "-"),
            "gamma_zero_gamma":            ("Zero Γ",    "#ffffff",    "--"),
            "gamma_call_wall":             ("Call Wall",  CS["green"],  ":"),
            "gamma_put_wall":              ("Put Wall",   CS["red"],    ":"),
            "pressure_pressure_eq":        ("P.Eq",       CS["purple"], "--"),
            "charm_max_charm_strike":      ("Max Charm",  CS["orange"], ":"),
        }
        for key, (label, color, ls) in level_styles.items():
            if key in gex_levels:
                val = gex_levels[key]
                ax.axhline(y=val, color=color, ls=ls, lw=1.2, alpha=0.7, zorder=2)
                ax.text(0, val, f" {label} ${val:.0f}", color=color,
                        fontsize=7, va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=CS["bg"],
                                  edgecolor=color, alpha=0.8))

    ax.grid(True, alpha=0.15, color=CS["grid"])
    ax.tick_params(colors=CS["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(CS["grid"])

    n = len(price_df)
    if n > 0:
        step = max(1, n // 8)
        tick_idx = list(range(0, n, step))
        tick_labels = []
        for i in tick_idx:
            idx_val = price_df.index[i]
            if hasattr(idx_val, "strftime"):
                tick_labels.append(idx_val.strftime("%H:%M"))
            else:
                tick_labels.append(str(i))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    display_ticker = TICKER_DISPLAY.get(ticker, ticker)
    ax.set_title(f"{display_ticker} Intraday", color=CS["text"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Price", color=CS["text"], fontsize=9)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────
# Session State Init
# ─────────────────────────────────────
def init_state():
    defaults = {
        "ticker": "SPY",
        "strike_range": 0.03,
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
# Sidebar Controls
# ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ GEXdon")
    st.markdown("---")

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

    strike_range = st.selectbox(
        "Strike Range", STRIKE_RANGES,
        format_func=lambda x: STRIKE_RANGE_LABELS[x],
        index=STRIKE_RANGES.index(st.session_state.strike_range),
        key="range_select",
    )
    if strike_range != st.session_state.strike_range:
        st.session_state.strike_range = strike_range
        st.session_state.live_snapshot = None

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Now", use_container_width=True):
        compute_gex_snapshot.clear()
        st.session_state.live_snapshot = None
        st.session_state.last_refresh = time.time()

    # Status
    et = now_et()
    market_status = "🟢 MARKET OPEN" if is_market_hours(et) else "🔴 MARKET CLOSED"
    st.markdown(f"**{market_status}**")
    st.markdown(f"**ET:** {et.strftime('%H:%M:%S')}")

    # Historical snapshots
    st.markdown("---")
    st.markdown("### 📁 History")
    snaps = list_snapshots(st.session_state.ticker)
    if snaps:
        st.markdown(f"{len(snaps)} snapshots today")
        if st.button("📷 View Latest Snapshot"):
            st.session_state.slider_time = snaps[-1]
            st.session_state.viewing_mode = "historical"
    else:
        st.markdown("No snapshots yet")

    # Auto-refresh logic
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
    with st.spinner("Fetching data from Barchart..."):
        snap = compute_gex_snapshot(st.session_state.ticker, st.session_state.strike_range)
    if snap is None:
        st.error("❌ Failed to fetch options data. Barchart may be unavailable.")
        return None

    st.session_state.live_snapshot = snap

    # Auto-save snapshot if market hours and new bucket
    if is_market_hours():
        bucket = current_bucket()
        saved = save_snapshot(st.session_state.ticker, bucket, snap)
        if saved:
            st.toast(f"💾 Snapshot saved: {bucket_label(bucket)}")

    return snap

# ─────────────────────────────────────
# Main Layout
# ─────────────────────────────────────
st.markdown("# 📊 GEXdon")

# Header metrics row
snap = st.session_state.live_snapshot
if snap is None:
    snap = compute_live()

if snap is not None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ticker", TICKER_DISPLAY.get(snap["ticker"], snap["ticker"]))
    c2.metric("Spot", f"${snap['spot']:.2f}")
    c3.metric("Expiry", snap["expiry_label"])
    c4.metric("IV Coverage", f"{snap['iv_coverage']:.0f}%")
    c5.metric("Range", STRIKE_RANGE_LABELS.get(snap["percent_range"], "3%"))
    c6.metric("Source", snap.get("source", "barchart").upper())

    # Time slider
    snaps_today = list_snapshots(st.session_state.ticker)
    if snaps_today:
        st.markdown("---")
        bucket_options = ["LIVE"] + [t.strftime("%H:%M ET") for t in snaps_today]
        selected = st.select_slider(
            "⏱ Time",
            options=bucket_options,
            value="LIVE",
            key="time_slider",
        )

        if selected != "LIVE":
            # Load historical snapshot
            hhmm = selected.replace(" ET", "")
            h, m = int(hhmm[:2]), int(hhmm[3:5])
            target_time = dtime(h, m)
            target_dt = datetime.combine(datetime.today(), target_time)
            hist_snap = load_snapshot(st.session_state.ticker, target_dt)
            if hist_snap:
                snap = hist_snap
                st.info(f"📷 Viewing snapshot: {selected}")

    # Two-column layout
    st.markdown("---")
    col_price, col_gex = st.columns([0.3, 0.7])

    with col_price:
        price_fig = create_price_chart(
            st.session_state.ticker,
            gex_levels=snap.get("levels"),
            spot=snap["spot"],
        )
        st.pyplot(price_fig, use_container_width=True)
        plt.close(price_fig)

    with col_gex:
        # Chart 1: Gamma Density
        st.pyplot(snap["fig1"], use_container_width=True)

        # Chart 2: Charm Density
        st.pyplot(snap["fig2"], use_container_width=True)

        # Chart 3: Charm Pressure & Acceleration
        st.pyplot(snap["fig3"], use_container_width=True)

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
                    st.markdown(f"Zero Γ: **${levels['gamma_zero_gamma']:.1f}**")
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

    # Close all figures to prevent memory leak
    plt.close("all")

else:
    st.warning("⏳ Waiting for data... Click 🔄 Refresh Now in the sidebar.")
