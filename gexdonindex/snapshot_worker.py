"""
GEXdon Snapshot Worker — Headless chain fetcher for GitHub Actions.
Saves snapshots as JSON. Includes price data and VEX computation.
"""
import os, sys, json, pytz
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import NYSE_TZ, MARKET_OPEN, MARKET_CLOSE, TICKERS, SNAPSHOT_DIR, PERIOD_LABELS
from gex.gex_utils import fetch_options_data, fetch_price_data, build_diagnostic_table, compute_net_vex
from gex.gex_chart_1 import generate_gex_chart
from gex.gex_chart_2 import generate_charm_chart
from gex.gex_chart_3 import generate_pressure_chart

SNAPSHOT_TICKERS = ["SPY", "QQQ", "SPX"]
DEFAULT_STRIKE_RANGE = 0.03
DEFAULT_MAX_DTE = 0

def now_et(): return datetime.now(NYSE_TZ)
def is_market_hours(dt_et=None):
    if dt_et is None: dt_et = now_et()
    return dt_et.weekday() < 5 and MARKET_OPEN <= dt_et.time() <= MARKET_CLOSE
def current_bucket(dt_et=None):
    if dt_et is None: dt_et = now_et()
    minute = (dt_et.minute // 30) * 30
    return dt_et.replace(minute=minute, second=0, microsecond=0)
def hhmm_to_period(hhmm): return PERIOD_LABELS.get(hhmm, "?")
def snapshot_dir(ticker, date_str=None):
    if date_str is None: date_str = now_et().strftime("%Y-%m-%d")
    path = os.path.join(SCRIPT_DIR, SNAPSHOT_DIR, ticker.replace("^","_"), date_str)
    os.makedirs(path, exist_ok=True)
    return path

def compute_and_save(ticker, percent_range, max_dte, bucket_dt):
    hhmm = bucket_dt.strftime("%H%M")
    period = hhmm_to_period(hhmm)
    date_str = bucket_dt.strftime("%Y-%m-%d")
    sdir = snapshot_dir(ticker, date_str)
    fpath = os.path.join(sdir, f"{hhmm}.json")
    if os.path.exists(fpath):
        print(f"  ⏭  {ticker} {period}({hhmm}) — exists, skipping")
        return True

    data = fetch_options_data(ticker, max_dte=max_dte)
    if data is None:
        print(f"  ✗  {ticker} {period}({hhmm}) — fetch failed")
        return False

    spot = data["spot"]
    price_df = fetch_price_data(ticker, n_bars=600)
    fig1, levels1 = generate_gex_chart(data, percent_range)
    fig2, levels2 = generate_charm_chart(data, percent_range)
    fig3, levels3 = generate_pressure_chart(data, percent_range)

    all_levels = {}
    all_levels.update({f"gamma_{k}": v for k, v in levels1.items()})
    all_levels.update({f"charm_{k}": v for k, v in levels2.items()})
    all_levels.update({f"pressure_{k}": v for k, v in levels3.items()})

    diag_table = build_diagnostic_table(data, percent_range)
    diag_info = data.get("diagnostics", {})
    vex_data = compute_net_vex(data, percent_range)

    def _df_to_dict(df):
        if df is None: return None
        d = df.copy()
        if not isinstance(d.index, pd.RangeIndex): d = d.reset_index()
        return {"__df__": True, "data": d.to_dict(orient="list")}

    def _fig_to_dict(fig): return {"__plotly__": True, "data": fig.to_plotly_json()}

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)): return str(obj)
        return str(obj)

    snapshot = {
        "timestamp": now_et().isoformat(), "ticker": ticker,
        "percent_range": percent_range, "max_dte": max_dte,
        "spot": float(spot), "expiry": data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage": float(data["iv_coverage"]),
        "fallback_iv": float(data["fallback_iv"]) if data["fallback_iv"] is not None else None,
        "source": data.get("source", "barchart"),
        "fetched_expiries": data.get("fetched_expiries", []),
        "fig1": _fig_to_dict(fig1), "fig2": _fig_to_dict(fig2), "fig3": _fig_to_dict(fig3),
        "levels": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                   for k, v in all_levels.items() if v is not None and not isinstance(v, list)},
        "calls": _df_to_dict(data["calls"]), "puts": _df_to_dict(data["puts"]),
        "price_data": _df_to_dict(price_df) if price_df is not None and not price_df.empty else None,
        "vex_info": vex_data,
        "diag_table": _df_to_dict(diag_table), "diag_info": diag_info,
    }

    with open(fpath, "w") as f: json.dump(snapshot, f, default=_default)
    print(f"  ✓  {ticker} {period}({hhmm}) — spot=${spot:.2f}, {len(data['calls'])}C/{len(data['puts'])}P, IV={data['iv_coverage']:.0f}%")
    return True

def main():
    et = now_et(); bucket = current_bucket(et)
    hhmm = bucket.strftime("%H%M"); period = hhmm_to_period(hhmm)
    print(f"{'='*60}\n  GEXdon Snapshot Worker\n  ET: {et.strftime('%Y-%m-%d %H:%M:%S')}\n  Bucket: {period} ({hhmm})\n  Market: {is_market_hours(et)}\n{'='*60}")

    if not is_market_hours(et):
        print("  Market closed")
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        if event != "workflow_dispatch": return
        print("  Manual trigger — proceeding")

    ok, fail = 0, 0
    for t in SNAPSHOT_TICKERS:
        if compute_and_save(t, DEFAULT_STRIKE_RANGE, DEFAULT_MAX_DTE, bucket): ok += 1
        else: fail += 1
    print(f"\n  Done: {ok} saved, {fail} failed")

if __name__ == "__main__": main()
