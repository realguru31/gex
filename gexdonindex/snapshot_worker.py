"""
GEXdon Snapshot Worker — Headless chain fetcher for GitHub Actions.
No Streamlit dependency. Fetches options chains for all tickers,
computes GEX/Charm, and saves snapshots as pickle files.

Usage: python snapshot_worker.py
"""
import os
import sys
import pickle
import pytz
from datetime import datetime, time as dtime

# Ensure we can import from gex/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    NYSE_TZ, MARKET_OPEN, MARKET_CLOSE,
    TICKERS, SNAPSHOT_DIR, PERIOD_LABELS,
    STRIKE_RANGES, DTE_OPTIONS,
)
from gex.gex_utils import fetch_options_data, build_diagnostic_table
from gex.gex_chart_1 import generate_gex_chart
from gex.gex_chart_2 import generate_charm_chart
from gex.gex_chart_3 import generate_pressure_chart

# ─────────────────────────────────────
# Config — what to capture
# ─────────────────────────────────────
SNAPSHOT_TICKERS = TICKERS  # All 4: SPY, QQQ, SPX, NDX
DEFAULT_STRIKE_RANGE = 0.03
DEFAULT_MAX_DTE = 0  # 0DTE by default


def now_et():
    return datetime.now(NYSE_TZ)


def is_market_hours(dt_et=None):
    if dt_et is None:
        dt_et = now_et()
    t = dt_et.time()
    return dt_et.weekday() < 5 and MARKET_OPEN <= t <= MARKET_CLOSE


def current_bucket(dt_et=None):
    """Round to nearest 30-min bucket."""
    if dt_et is None:
        dt_et = now_et()
    minute = (dt_et.minute // 30) * 30
    return dt_et.replace(minute=minute, second=0, microsecond=0)


def hhmm_to_period(hhmm):
    return PERIOD_LABELS.get(hhmm, "?")


def snapshot_dir(ticker, date_str=None):
    if date_str is None:
        date_str = now_et().strftime("%Y-%m-%d")
    path = os.path.join(SCRIPT_DIR, SNAPSHOT_DIR, ticker.replace("^", "_"), date_str)
    os.makedirs(path, exist_ok=True)
    return path


def compute_and_save(ticker, percent_range, max_dte, bucket_dt):
    """Fetch chain, compute charts, save snapshot. Returns True on success."""
    hhmm = bucket_dt.strftime("%H%M")
    period = hhmm_to_period(hhmm)
    date_str = bucket_dt.strftime("%Y-%m-%d")

    # Check if already saved
    sdir = snapshot_dir(ticker, date_str)
    fpath = os.path.join(sdir, f"{hhmm}.pkl")
    if os.path.exists(fpath):
        print(f"  ⏭  {ticker} {period}({hhmm}) — already exists, skipping")
        return True

    # Fetch data
    data = fetch_options_data(ticker, max_dte=max_dte)
    if data is None:
        print(f"  ✗  {ticker} {period}({hhmm}) — fetch failed")
        return False

    spot = data["spot"]

    # Generate charts (these return Plotly Figure objects)
    fig1, levels1 = generate_gex_chart(data, percent_range)
    fig2, levels2 = generate_charm_chart(data, percent_range)
    fig3, levels3 = generate_pressure_chart(data, percent_range)

    # Merge levels
    all_levels = {}
    all_levels.update({f"gamma_{k}": v for k, v in levels1.items()})
    all_levels.update({f"charm_{k}": v for k, v in levels2.items()})
    all_levels.update({f"pressure_{k}": v for k, v in levels3.items()})

    # Build diagnostic table
    diag_table = build_diagnostic_table(data, percent_range)
    diag_info = data.get("diagnostics", {})

    snapshot = {
        "timestamp": now_et().isoformat(),
        "ticker": ticker,
        "percent_range": percent_range,
        "max_dte": max_dte,
        "spot": spot,
        "expiry": data["expiry"],
        "expiry_label": data["expiry_label"],
        "iv_coverage": data["iv_coverage"],
        "fallback_iv": data["fallback_iv"],
        "source": data.get("source", "barchart"),
        "fetched_expiries": data.get("fetched_expiries", []),
        "fig1": fig1, "fig2": fig2, "fig3": fig3,
        "levels": all_levels,
        "calls": data["calls"],
        "puts": data["puts"],
        "diag_table": diag_table,
        "diag_info": diag_info,
    }

    # Save
    with open(fpath, "wb") as f:
        pickle.dump(snapshot, f)

    print(f"  ✓  {ticker} {period}({hhmm}) — spot=${spot:.2f}, "
          f"{len(data['calls'])}C/{len(data['puts'])}P, "
          f"IV={data['iv_coverage']:.0f}%")
    return True


def main():
    et = now_et()
    bucket = current_bucket(et)
    hhmm = bucket.strftime("%H%M")
    period = hhmm_to_period(hhmm)
    date_str = et.strftime("%Y-%m-%d")

    print(f"{'='*60}")
    print(f"  GEXdon Snapshot Worker")
    print(f"  ET: {et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bucket: {period} ({hhmm})")
    print(f"  Market hours: {is_market_hours(et)}")
    print(f"{'='*60}")

    if not is_market_hours(et):
        print("  Market closed — skipping snapshot capture")
        print("  (Use workflow_dispatch to force a capture)")
        # Still allow manual trigger via workflow_dispatch
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--force", action="store_true")
        args, _ = parser.parse_known_args()
        if not args.force and "GITHUB_EVENT_NAME" in os.environ:
            event = os.environ.get("GITHUB_EVENT_NAME", "")
            if event == "workflow_dispatch":
                print("  Manual trigger detected — proceeding anyway")
            else:
                return

    success = 0
    failed = 0
    for ticker in SNAPSHOT_TICKERS:
        ok = compute_and_save(ticker, DEFAULT_STRIKE_RANGE, DEFAULT_MAX_DTE, bucket)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n  Done: {success} saved, {failed} failed")
    print(f"  Snapshots dir: {os.path.join(SCRIPT_DIR, SNAPSHOT_DIR)}")

    # List what we have for today
    print(f"\n  Today's snapshots ({date_str}):")
    for ticker in SNAPSHOT_TICKERS:
        sdir = os.path.join(SCRIPT_DIR, SNAPSHOT_DIR, ticker, date_str)
        if os.path.exists(sdir):
            files = sorted(os.listdir(sdir))
            periods = [f"{hhmm_to_period(f.replace('.pkl',''))}({f.replace('.pkl','')})" for f in files]
            print(f"    {ticker}: {', '.join(periods)}")
        else:
            print(f"    {ticker}: none")


if __name__ == "__main__":
    main()
