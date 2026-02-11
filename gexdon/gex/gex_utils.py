"""
GEXdon Utilities — ZERO yfinance.
  Barchart   → spot, expiry dates, options chains (multi-DTE aggregation)
  tvdatafeed → intraday candlestick chart only
  BS Greeks  → delta, gamma, charm
"""
import re
import requests
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime, timedelta
from urllib.parse import unquote
from config import (
    RISK_FREE_RATE, BC_BASE_SYM, BC_PAGE_TYPE,
    TV_EXCHANGE, TV_SYMBOL, TV_EXCHANGE_FALLBACKS,
)
import logging

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
MIN_T = 1 / (24 * 60)


# ═══════════════════════════════════════
# Black-Scholes Greeks
# ═══════════════════════════════════════
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    n_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = n_d1 / (S * sigma * sqrt(T))
    charm = -exp(-r * T) * n_d1 * (r / (sigma * sqrt(T)) - d1 / (2 * T))
    return delta, gamma, charm


def resolve_iv(row_iv, fallback_iv):
    if row_iv and row_iv > 0:
        return row_iv
    if fallback_iv is not None and fallback_iv > 0:
        return fallback_iv
    return None


# ═══════════════════════════════════════
# Expiry Date Helpers
# ═══════════════════════════════════════
def _calculate_expiry_dates(n=30):
    dates = []
    d = datetime.now().date()
    for i in range(90):
        check = d + timedelta(days=i)
        if check.weekday() < 5:
            dates.append(check.strftime("%Y-%m-%d"))
        if len(dates) >= n:
            break
    return dates


def _parse_expiry_dates_from_html(html):
    today_str = datetime.now().strftime("%Y-%m-%d")
    all_dates = re.findall(r'20\d{2}-\d{2}-\d{2}', html)
    valid = sorted(set(d for d in all_dates if d >= today_str))
    return valid if valid else None


# ═══════════════════════════════════════
# Barchart Session + Spot
# ═══════════════════════════════════════
def _create_barchart_session(ticker):
    """Create Barchart session with XSRF token. Returns (sess, xsrf, api_headers, page_html, base_sym) or None."""
    base_sym = BC_BASE_SYM.get(ticker, ticker)
    page_type = BC_PAGE_TYPE.get(ticker, "etfs-funds")
    page_url = f"https://www.barchart.com/{page_type}/quotes/{base_sym}/volatility-greeks"

    try:
        sess = requests.Session()
        r = sess.get(page_url, params={"page": "all"}, headers={
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "max-age=0",
            "upgrade-insecure-requests": "1",
            "user-agent": _UA,
        }, timeout=15)
        r.raise_for_status()
        page_html = r.text

        cookies = sess.cookies.get_dict()
        if "XSRF-TOKEN" not in cookies:
            return None
        xsrf = unquote(cookies["XSRF-TOKEN"])

        api_headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "referer": page_url,
            "user-agent": _UA,
            "x-xsrf-token": xsrf,
        }
        return sess, xsrf, api_headers, page_html, base_sym
    except Exception as e:
        logger.error("Barchart session failed for %s: %s", ticker, e)
        return None


def _get_barchart_spot(session, api_headers, ticker):
    base = BC_BASE_SYM.get(ticker, ticker)
    try:
        r = session.get(
            "https://www.barchart.com/proxies/core-api/v1/quotes/get",
            params={"symbols": base, "fields": "lastPrice,previousClose"},
            headers=api_headers, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        items = data.get("data", [])
        if items:
            raw = items if isinstance(items, list) else [items]
            for item in raw:
                raw_item = item.get("raw", item)
                lp = raw_item.get("lastPrice")
                if lp:
                    return float(lp)
    except Exception:
        pass
    return None


def _parse_spot_from_html(html):
    for pat in [r'"lastPrice"\s*:\s*"?([\d.]+)"?', r'"last"\s*:\s*"?([\d.]+)"?']:
        m = re.search(pat, html)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


# ═══════════════════════════════════════
# Fetch Single Chain from Barchart
# ═══════════════════════════════════════
def _fetch_single_chain(session, api_headers, base_sym, expiry):
    """Fetch one expiry's chain. Returns DataFrame with optionType column, or None."""
    api_url = "https://www.barchart.com/proxies/core-api/v1/options/get"
    try:
        r = session.get(api_url, params={
            "baseSymbol": base_sym,
            "groupBy": "optionType",
            "expirationDate": expiry,
            "orderBy": "strikePrice",
            "orderDir": "asc",
            "raw": "1",
            "fields": "symbol,strikePrice,lastPrice,volatility,delta,gamma,theta,vega,volume,openInterest,optionType",
        }, headers=api_headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        rows = []
        for opt_type, options in data.get("data", {}).items():
            for opt in options:
                opt["optionType"] = opt_type
                rows.append(opt)
        if not rows:
            return None

        df = pd.DataFrame(rows)
        for col in ["strikePrice", "lastPrice", "volatility", "delta",
                     "gamma", "theta", "vega", "volume", "openInterest"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["openInterest"] = df["openInterest"].astype(int)
        df["iv_decimal"] = df["volatility"] / 100.0
        df["expiry"] = expiry

        # Compute t_expiry for this chain
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        t_exp = max(MIN_T, (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
        df["t_expiry"] = t_exp

        return df
    except Exception as e:
        logger.debug("Chain fetch failed for %s: %s", expiry, e)
        return None


# ═══════════════════════════════════════
# Fetch Aggregated Options (multi-DTE)
# ═══════════════════════════════════════
def fetch_options_data(ticker, max_dte=0):
    """
    Fetch and aggregate options from 0 DTE through max_dte.
    Returns unified dict with combined calls/puts DataFrames.
    Each row has its own t_expiry column for per-expiry Greeks.
    """
    # Create session
    result = _create_barchart_session(ticker)
    if result is None:
        return None
    sess, xsrf, api_headers, page_html, base_sym = result

    # Spot price
    spot = _get_barchart_spot(sess, api_headers, ticker)
    if spot is None:
        spot = _parse_spot_from_html(page_html)
    if spot is None:
        spot = _get_tv_spot(ticker)
    if spot is None:
        logger.error("Cannot get spot for %s", ticker)
        return None

    # Expiry dates
    expiry_dates = _parse_expiry_dates_from_html(page_html)
    if not expiry_dates:
        expiry_dates = _calculate_expiry_dates()
    if not expiry_dates:
        return None

    # Determine which expiries to fetch (0 through max_dte)
    num_expiries = min(max_dte + 1, len(expiry_dates))
    target_expiries = expiry_dates[:num_expiries]

    # Fetch each chain
    all_calls = []
    all_puts = []
    fetched_expiries = []

    for expiry in target_expiries:
        df = _fetch_single_chain(sess, api_headers, base_sym, expiry)
        if df is not None and len(df) > 0:
            calls_df = df[df["optionType"] == "Call"].copy()
            puts_df  = df[df["optionType"] == "Put"].copy()
            if len(calls_df) > 0:
                all_calls.append(calls_df)
            if len(puts_df) > 0:
                all_puts.append(puts_df)
            fetched_expiries.append(expiry)

    if not all_calls and not all_puts:
        logger.warning("No chain data for %s", ticker)
        return None

    calls = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts  = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    # IV quality
    n_valid = 0
    n_total = max(len(calls) + len(puts), 1)
    if len(calls) > 0:
        n_valid += len(calls[calls["iv_decimal"] > 0])
    if len(puts) > 0:
        n_valid += len(puts[puts["iv_decimal"] > 0])
    iv_pct = n_valid / n_total * 100.0
    fallback_iv = 0.15 if iv_pct < 50 else None

    # Expiry label
    if max_dte == 0:
        first_exp = fetched_expiries[0] if fetched_expiries else target_expiries[0]
        exp_dt = datetime.strptime(first_exp, "%Y-%m-%d")
        t0 = max(MIN_T, (exp_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
        expiry_label = "0DTE" if exp_dt.date() == datetime.now().date() else f"{t0*365:.1f}d"
    else:
        expiry_label = f"0-{max_dte} DTE ({len(fetched_expiries)} chains)"

    # t_expiry for single-DTE backward compat
    first_exp = fetched_expiries[0] if fetched_expiries else target_expiries[0]
    exp_dt = datetime.strptime(first_exp, "%Y-%m-%d")
    t_expiry_single = max(MIN_T, (exp_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))

    return {
        "ticker": ticker, "spot": spot,
        "calls": calls, "puts": puts,
        "expiry": first_exp, "expiry_dt": exp_dt,
        "expiry_dates": expiry_dates,
        "t_expiry": t_expiry_single,
        "expiry_label": expiry_label,
        "fallback_iv": fallback_iv, "iv_coverage": iv_pct,
        "source": "barchart",
        "multi_dte": max_dte > 0,
        "fetched_expiries": fetched_expiries,
    }


# ═══════════════════════════════════════
# tvdatafeed (price chart only)
# ═══════════════════════════════════════
def _get_tv_spot(ticker):
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        sym = TV_SYMBOL.get(ticker, ticker.replace("^", ""))
        primary_ex = TV_EXCHANGE.get(ticker, "AMEX")
        exchanges = [primary_ex] + [e for e in TV_EXCHANGE_FALLBACKS if e != primary_ex]
        for ex in exchanges:
            try:
                df = tv.get_hist(symbol=sym, exchange=ex,
                                 interval=Interval.in_1_minute, n_bars=1)
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                continue
    except ImportError:
        pass
    return None


def fetch_price_data(ticker, n_bars=100):
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        sym = TV_SYMBOL.get(ticker, ticker.replace("^", ""))
        primary_ex = TV_EXCHANGE.get(ticker, "AMEX")
        exchanges = [primary_ex] + [e for e in TV_EXCHANGE_FALLBACKS if e != primary_ex]
        for ex in exchanges:
            try:
                df = tv.get_hist(symbol=sym, exchange=ex,
                                 interval=Interval.in_5_minute, n_bars=n_bars)
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "volume": "Volume",
                    })
                    return df
            except Exception:
                continue
    except ImportError:
        logger.warning("tvdatafeed not installed")
    return pd.DataFrame()
