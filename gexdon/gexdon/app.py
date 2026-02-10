"""
GEXdon — Real-Time GEX Dashboard
Main Streamlit application with snapshot engine, time slider, price chart overlay.
"""
import os, sys, time, pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
import pytz

# ── ensure app root is on sys.path ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config import (
    NYSE_TZ, MARKET_OPEN, MARKET_CLOSE, SNAPSHOT_INTERVAL_MIN,
    AUTO_REFRESH_SEC, TICKERS, TICKER_DISPLAY,
    STRIKE_RANGES, STRIKE_RANGE_LABELS, SNAPSHOT_DIR, CS,
)
from gex.gex_utils import fetch_options_data, fetch_price_data
from gex.gex_chart_1 import generate_gex_chart
from gex.gex_chart_2 import generate_charm_chart
from gex.gex_chart_3 import generate_pressure_chart

# ══════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════
st.set_page_config(page_title="GEXdon", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ── Inject dark-blue CSS ──
css_path = os.path.join(APP_DIR, "theme", "dark_blue.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# NYSE Time Helpers
# ══════════════════════════════════════════
def now_et():
    return datetime.now(NYSE_TZ)

def is_market_open(dt=None):
    dt = dt or now_et()
    return dt.weekday() < 5 and MARKET_OPEN <= dt.time() <= MARKET_CLOSE

def current_bucket(dt=None):
    """Round DOWN to nearest 30-min bucket."""
    dt = dt or now_et()
    m = (dt.minute // SNAPSHOT_INTERVAL_MIN) * SNAPSHOT_INTERVAL_MIN
    return dt.replace(minute=m, second=0, microsecond=0)

def bucket_label(t):
    """Format a time object or datetime as 'HH:MM ET'."""
    if isinstance(t, datetime):
        return t.strftime("%H:%M ET")
    return t.strftime("%H:%M ET")


# ══════════════════════════════════════════
# Snapshot Engine
# ══════════════════════════════════════════
def _snap_dir(ticker, date_str=None):
    """Return snapshot directory path, creating it if needed."""
    if date_str is None:
        date_str = now_et().strftime("%Y-%m-%d")
    d = os.path.join(APP_DIR, SNAPSHOT_DIR, ticker.replace("^", "_"), date_str)
    os.makedirs(d, exist_ok=True)
    return d

def _snap_fname(bucket):
    """Convert bucket (datetime or time) to 'HHMM.pkl'."""
    if isinstance(bucket, datetime):
        return bucket.strftime("%H%M") + ".pkl"
    return bucket.strftime("%H%M") + ".pkl"

def save_snapshot(ticker, bucket_dt, payload):
    """Save snapshot ONLY if file does not exist (never overwrite)."""
    fpath = os.path.join(_snap_dir(ticker), _snap_fname(bucket_dt))
    if os.path.exists(fpath):
        return False
    with open(fpath, "wb") as f:
        pickle.dump(payload, f)
    return True

def load_snapshot(ticker, bucket_time, date_str=None):
    """Load snapshot from disk. bucket_time can be datetime or time."""
    fpath = os.path.join(_snap_dir(ticker, date_str), _snap_fname(bucket_time))
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        return pickle.load(f)

def list_snapshots(ticker, date_str=None):
    """Return sorted list of time objects for available snapshots."""
    d = _snap_dir(ticker, date_str)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pkl"):
            continue
        hhmm = fn.replace(".pkl", "")
        try:
            out.append(dtime(int(hhmm[:2]), int(hhmm[2:])))
        except (ValueError, IndexError):
            pass
    return out


# ══════════════════════════════════════════
# GEX Computation (live)
# ══════════════════════════════════════════
def compute_live(ticker, pct_range):
    """Fetch data + compute all 3 charts. Returns snapshot payload dict or None."""
    data = fetch_options_data(ticker)
    if data is None:
        return None
    fig1, lv1 = generate_gex_chart(data, pct_range)
    fig2, lv2 = generate_charm_chart(data, pct_range)
    fig3, lv3 = generate_pressure_chart(data, pct_range)

    # merge levels with prefixes
    all_levels = {}
    for k, v in lv1.items(): all_levels[f"g_{k}"] = v
    for k, v in lv2.items(): all_levels[f"c_{k}"] = v
    for k, v in lv3.items(): all_levels[f"p_{k}"] = v

    return {
        "timestamp":    now_et().isoformat(),
        "ticker":       ticker,
        "pct_range":    pct_range,
        "spot":         data["spot"],
        "expiry":       data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage":  data["iv_coverage"],
        "fallback_iv":  data["fallback_iv"],
        "fig1": fig1, "fig2": fig2, "fig3": fig3,
        "levels": all_levels,
    }


# ══════════════════════════════════════════
# Price Chart with GEX Level Overlay
# ══════════════════════════════════════════
def build_price_chart(ticker, levels=None, spot=None):
    """Create intraday price chart with GEX level overlays."""
    pdf = fetch_price_data(ticker)
    fig, ax = plt.subplots(figsize=(6.5, 12))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])

    if pdf.empty:
        ax.text(.5, .5, "No price data", transform=ax.transAxes,
                ha="center", va="center", color=CS["text"], fontsize=14)
        return fig

    close = pdf["Close"].values
    high  = pdf["High"].values
    low   = pdf["Low"].values
    x     = np.arange(len(close))

    ax.plot(x, close, color=CS["cyan"], lw=1.4, alpha=.9, zorder=3)
    ax.fill_between(x, low, high, color=CS["blue"], alpha=.06, zorder=1)

    if spot:
        ax.axhline(spot, color=CS["cyan"], ls="-", lw=.8, alpha=.35)
        ax.text(len(x) - 1, spot, f"  ${spot:.2f}", color=CS["cyan"],
                fontsize=7, va="bottom", fontweight="bold")

    # ── overlay GEX levels ──
    if levels:
        overlay = {
            "g_max_gamma":   ("Max Γ",     CS["gold"],   "-"),
            "g_zero_gamma":  ("Zero Γ",    "#ffffff",    "--"),
            "g_call_wall":   ("Call Wall",  CS["green"],  ":"),
            "g_put_wall":    ("Put Wall",   CS["red"],    ":"),
            "p_pressure_eq": ("P.Equil",    CS["purple"], "--"),
            "c_max_charm_strike": ("Max Charm", CS["orange"], ":"),
        }
        for key, (lbl, col, ls) in overlay.items():
            val = levels.get(key)
            if val is None:
                continue
            ax.axhline(val, color=col, ls=ls, lw=1.1, alpha=.7, zorder=2)
            ax.text(0, val, f" {lbl} ${val:.0f}", color=col, fontsize=6.5,
                    va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=.12", fc=CS["bg"],
                              ec=col, alpha=.85))

    # ── x-axis time ticks ──
    n = len(pdf)
    if n > 0:
        step = max(1, n // 8)
        ti = list(range(0, n, step))
        tl = []
        for i in ti:
            idx = pdf.index[i]
            tl.append(idx.strftime("%H:%M") if hasattr(idx, "strftime") else str(i))
        ax.set_xticks(ti); ax.set_xticklabels(tl, rotation=45, fontsize=6)

    ax.grid(True, alpha=.12, color=CS["grid"])
    ax.tick_params(colors=CS["text"], labelsize=6)
    for sp in ax.spines.values(): sp.set_color(CS["grid"])
    disp = TICKER_DISPLAY.get(ticker, ticker)
    ax.set_title(f"{disp} Intraday", color=CS["text"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Price ($)", color=CS["text"], fontsize=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════
_DEFAULTS = {
    "ticker": "SPY",
    "pct_range": 0.03,
    "slider_idx": None,
    "last_auto": 0.0,
    "snap_cache": None,     # current displayed snapshot payload
    "mode": "live",         # "live" | "historical"
    "force_refresh": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ GEXdon")
    st.markdown("---")

    sel_ticker = st.selectbox("Ticker", TICKERS,
                              format_func=lambda x: TICKER_DISPLAY.get(x, x),
                              index=TICKERS.index(st.session_state.ticker))
    if sel_ticker != st.session_state.ticker:
        st.session_state.ticker = sel_ticker
        st.session_state.snap_cache = None
        st.session_state.slider_idx = None

    sel_range = st.selectbox("Strike Range", STRIKE_RANGES,
                             format_func=lambda x: STRIKE_RANGE_LABELS[x],
                             index=STRIKE_RANGES.index(st.session_state.pct_range))
    if sel_range != st.session_state.pct_range:
        st.session_state.pct_range = sel_range
        st.session_state.snap_cache = None
        st.session_state.slider_idx = None

    st.markdown("---")

    # Manual refresh button
    if st.button("🔄  Refresh Now", use_container_width=True):
        st.session_state.force_refresh = True
        st.session_state.snap_cache = None

    st.markdown("---")

    # Status
    et = now_et()
    mkt = is_market_open(et)
    st.markdown(f"**🕐 {et.strftime('%H:%M:%S ET')}**")
    st.markdown(f"**Market:** {'🟢 Open' if mkt else '🔴 Closed'}")
    st.markdown(f"**Bucket:** {bucket_label(current_bucket(et))}")

    snaps = list_snapshots(st.session_state.ticker)
    st.markdown(f"**Snapshots today:** {len(snaps)}")

    st.markdown("---")
    st.caption("Auto-refreshes every 5 min during market hours.")


# ══════════════════════════════════════════
# Auto-refresh Logic
# ══════════════════════════════════════════
now_ts = time.time()
if (is_market_open()
        and now_ts - st.session_state.last_auto > AUTO_REFRESH_SEC
        and not st.session_state.force_refresh):
    st.session_state.last_auto = now_ts
    st.session_state.snap_cache = None

# ══════════════════════════════════════════
# Load or Compute Data
# ══════════════════════════════════════════
ticker    = st.session_state.ticker
pct_range = st.session_state.pct_range

# Determine available snapshots for today
available_snaps = list_snapshots(ticker)

# ── Compute live data if needed ──
if st.session_state.snap_cache is None:
    with st.spinner("Computing GEX levels…"):
        payload = compute_live(ticker, pct_range)
    if payload:
        st.session_state.snap_cache = payload
        st.session_state.mode = "live"

        # Save snapshot if market open + new bucket
        if is_market_open():
            bucket = current_bucket()
            saved = save_snapshot(ticker, bucket, payload)
            if saved:
                available_snaps = list_snapshots(ticker)  # refresh list
    st.session_state.force_refresh = False
    st.session_state.last_auto = time.time()


# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
snap = st.session_state.snap_cache

st.markdown("# 📊 GEXdon")
if snap:
    disp = TICKER_DISPLAY.get(snap["ticker"], snap["ticker"])
    cols = st.columns(5)
    cols[0].metric("Ticker", disp)
    cols[1].metric("Spot", f"${snap['spot']:.2f}")
    cols[2].metric("Expiry", snap["expiry_label"])
    iv_txt = f"{snap['iv_coverage']:.0f}%" + (f" (fb {snap['fallback_iv']:.3f})" if snap["fallback_iv"] else "")
    cols[3].metric("IV Coverage", iv_txt)
    cols[4].metric("Range", STRIKE_RANGE_LABELS[pct_range])

st.markdown("---")


# ══════════════════════════════════════════
# TIME SLIDER (above GEX charts)
# ══════════════════════════════════════════
slider_snap = None  # will hold the snapshot to display (live or historical)

if available_snaps:
    snap_labels = [bucket_label(t) for t in available_snaps]
    # Default to latest
    default_idx = len(available_snaps) - 1
    if st.session_state.slider_idx is not None:
        default_idx = min(st.session_state.slider_idx, len(available_snaps) - 1)

    chosen_idx = st.select_slider(
        "📅 Snapshot Time",
        options=list(range(len(available_snaps))),
        format_func=lambda i: snap_labels[i],
        value=default_idx,
        key="time_slider",
    )
    st.session_state.slider_idx = chosen_idx

    # If user picked a historical bucket different from live, load from disk
    chosen_time = available_snaps[chosen_idx]
    cb = current_bucket()
    is_latest = (chosen_time.hour == cb.hour and chosen_time.minute == cb.minute)

    if is_latest and snap:
        slider_snap = snap
        st.session_state.mode = "live"
    else:
        loaded = load_snapshot(ticker, chosen_time)
        if loaded:
            slider_snap = loaded
            st.session_state.mode = "historical"
        elif snap:
            slider_snap = snap
else:
    slider_snap = snap


# ══════════════════════════════════════════
# MAIN LAYOUT: Price (left) | GEX (right)
# ══════════════════════════════════════════
if slider_snap is None:
    st.warning("⚠️ No data available. Click **Refresh Now** or wait for market hours.")
    st.stop()

# Mode badge
mode_txt = "🟢 LIVE" if st.session_state.mode == "live" else "🕐 HISTORICAL"
ts_txt = slider_snap.get("timestamp", "")
st.markdown(f"**{mode_txt}** &nbsp;|&nbsp; Computed: {ts_txt}")

col_price, col_gex = st.columns([0.32, 0.68])

# ── LEFT: Price Chart ──
with col_price:
    st.markdown("### Price Chart")
    price_fig = build_price_chart(
        ticker,
        levels=slider_snap.get("levels"),
        spot=slider_snap.get("spot"),
    )
    st.pyplot(price_fig, use_container_width=True)
    plt.close(price_fig)

    # Key levels summary
    lvls = slider_snap.get("levels", {})
    if lvls:
        with st.expander("🔑 Key GEX Levels", expanded=True):
            level_names = {
                "g_max_gamma":        "Max Gamma",
                "g_zero_gamma":       "Zero Gamma",
                "g_call_wall":        "Call Wall",
                "g_put_wall":         "Put Wall",
                "c_max_charm_strike": "Max Charm",
                "c_net_charm_exp":    "Net Charm Exp",
                "p_pressure_eq":      "Pressure Eq",
                "p_max_pressure_strike": "Max Pressure",
            }
            for key, name in level_names.items():
                val = lvls.get(key)
                if val is not None:
                    if "exp" in key.lower():
                        st.markdown(f"**{name}:** {val:,.0f}")
                    else:
                        st.markdown(f"**{name}:** ${val:.1f}")

# ── RIGHT: Three GEX Charts ──
with col_gex:
    st.markdown("### Gamma Density")
    st.pyplot(slider_snap["fig1"], use_container_width=True)

    st.markdown("### Charm Density")
    st.pyplot(slider_snap["fig2"], use_container_width=True)

    st.markdown("### Charm Pressure & Acceleration")
    st.pyplot(slider_snap["fig3"], use_container_width=True)


# ══════════════════════════════════════════
# Footer
# ══════════════════════════════════════════
st.markdown("---")
st.caption(
    f"GEXdon v1.0 &nbsp;|&nbsp; Data: Yahoo Finance &nbsp;|&nbsp; "
    f"Snapshots: {len(available_snaps)} today &nbsp;|&nbsp; "
    f"Next auto-refresh: ~{max(0, AUTO_REFRESH_SEC - int(time.time() - st.session_state.last_auto))}s"
)

# ── Schedule next auto-refresh via rerun ──
if is_market_open():
    elapsed = time.time() - st.session_state.last_auto
    remaining = max(1, AUTO_REFRESH_SEC - int(elapsed))
    time.sleep(min(remaining, 2))  # brief sleep before potential rerun check
    if time.time() - st.session_state.last_auto >= AUTO_REFRESH_SEC:
        st.rerun()
