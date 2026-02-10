"""
GEXdon Utilities — ZERO yfinance.
  Barchart   → spot price, expiry dates, options chain (OI, gamma, IV, delta)
  tvdatafeed → intraday price chart only (candlesticks)
  BS Greeks  → charm (not provided by any data source)
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


# ═══════════════════════════════════════
# Black-Scholes Greeks
# ═══════════════════════════════════════
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """(delta, gamma, charm) for a European option."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    n_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = n_d1 / (S * sigma * sqrt(T))
    charm = -exp(-r * T) * n_d1 * (r / (sigma * sqrt(T)) - d1 / (2 * T))
    return delta, gamma, charm


def resolve_iv(row_iv, fallback_iv):
    """Return usable IV (decimal), or None."""
    if row_iv and row_iv > 0:
        return row_iv
    if fallback_iv is not None and fallback_iv > 0:
        return fallback_iv
    return None


# ═══════════════════════════════════════
# Expiry Date Helpers (no yfinance)
# ═══════════════════════════════════════
def _calculate_expiry_dates(n=30):
    """Generate every weekday for next 90 calendar days as potential expiries."""
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
    """Parse available expiry dates from Barchart page HTML."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    all_dates = re.findall(r'20\d{2}-\d{2}-\d{2}', html)
    valid = sorted(set(d for d in all_dates if d >= today_str))
    return valid if valid else None


# ═══════════════════════════════════════
# Barchart Spot Price
# ═══════════════════════════════════════
def _get_barchart_spot(session, api_headers, ticker):
    """Get spot price from Barchart quotes API."""
    base = BC_BASE_SYM.get(ticker, ticker)
    try:
        r = session.get(
            "https://www.barchart.com/proxies/core-api/v1/quotes/get",
            params={"symbols": base, "fields": "lastPrice,previousClose"},
            headers=api_headers,
            timeout=10,
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
    except Exception as e:
        logger.debug("Barchart quotes API failed: %s", e)
    return None


def _parse_spot_from_html(html):
    """Fallback: parse spot price from Barchart page HTML."""
    patterns = [
        r'"lastPrice"\s*:\s*"?([\d.]+)"?',
        r'"last"\s*:\s*"?([\d.]+)"?',
        r'data-last-price="([\d.]+)"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


# ═══════════════════════════════════════
# BARCHART Full Fetch (PRIMARY DATA SOURCE)
# ═══════════════════════════════════════
def fetch_barchart_options(ticker, expiry_offset=0):
    """
    Fetch everything from Barchart: spot, expiry dates, options chain.
    Returns unified dict or None on failure.

    Columns in calls/puts DataFrames:
        strikePrice, openInterest, gamma, iv_decimal,
        delta, theta, vega, volume, lastPrice, volatility
    """
    MIN_T = 1 / (24 * 60)
    base_sym = BC_BASE_SYM.get(ticker, ticker)
    page_type = BC_PAGE_TYPE.get(ticker, "etfs-funds")
    page_url = f"https://www.barchart.com/{page_type}/quotes/{base_sym}/volatility-greeks"
    api_url = "https://www.barchart.com/proxies/core-api/v1/options/get"

    try:
        # ── Step 1: Session + XSRF token ──
        sess = requests.Session()
        r = sess.get(
            page_url,
            params={"page": "all"},
            headers={
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "upgrade-insecure-requests": "1",
                "user-agent": _UA,
            },
            timeout=15,
        )
        r.raise_for_status()
        page_html = r.text

        cookies = sess.cookies.get_dict()
        if "XSRF-TOKEN" not in cookies:
            logger.warning("No XSRF-TOKEN for %s", ticker)
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

        # ── Step 2: Spot price from Barchart ──
        spot = _get_barchart_spot(sess, api_headers, ticker)
        if spot is None:
            spot = _parse_spot_from_html(page_html)
        if spot is None:
            spot = _get_tv_spot(ticker)  # tvdatafeed fallback
        if spot is None:
            logger.error("Cannot get spot price for %s", ticker)
            return None

        # ── Step 3: Expiry dates ──
        expiry_dates = _parse_expiry_dates_from_html(page_html)
        if not expiry_dates:
            expiry_dates = _calculate_expiry_dates()
        if not expiry_dates:
            return None

        expiry_offset = min(expiry_offset, len(expiry_dates) - 1)
        expiry = expiry_dates[expiry_offset]

        # Time to expiry
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        t_exp = max(MIN_T, (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
        expiry_label = "0DTE" if expiry_dt.date() == datetime.now().date() else f"{t_exp * 365:.1f}d"

        # ── Step 4: Options chain API ──
        r = sess.get(
            api_url,
            params={
                "baseSymbol": base_sym,
                "groupBy": "optionType",
                "expirationDate": expiry,
                "orderBy": "strikePrice",
                "orderDir": "asc",
                "raw": "1",
                "fields": "symbol,strikePrice,lastPrice,volatility,delta,gamma,theta,vega,volume,openInterest,optionType",
            },
            headers=api_headers,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        # ── Step 5: Parse DataFrame ──
        rows = []
        for opt_type, options in data.get("data", {}).items():
            for opt in options:
                opt["optionType"] = opt_type
                rows.append(opt)
        if not rows:
            logger.warning("Barchart: empty chain for %s exp %s", ticker, expiry)
            return None

        df = pd.DataFrame(rows)
        for col in ["strikePrice", "lastPrice", "volatility", "delta",
                     "gamma", "theta", "vega", "volume", "openInterest"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["openInterest"] = df["openInterest"].astype(int)
        # Barchart volatility is IV in % → convert to decimal
        df["iv_decimal"] = df["volatility"] / 100.0

        calls = df[df["optionType"] == "Call"].copy().reset_index(drop=True)
        puts  = df[df["optionType"] == "Put"].copy().reset_index(drop=True)

        # IV quality check
        n_valid = len(calls[calls["iv_decimal"] > 0]) + len(puts[puts["iv_decimal"] > 0])
        n_total = max(len(calls) + len(puts), 1)
        iv_pct = n_valid / n_total * 100.0
        fallback_iv = None
        if iv_pct < 50:
            fallback_iv = 0.15  # safe default

        logger.info("Barchart OK: %s spot=$%.2f exp=%s #C=%d #P=%d iv=%.0f%%",
                     ticker, spot, expiry, len(calls), len(puts), iv_pct)

        return {
            "ticker": ticker, "spot": spot,
            "calls": calls, "puts": puts,
            "expiry": expiry, "expiry_dt": expiry_dt,
            "expiry_dates": expiry_dates,
            "t_expiry": t_exp, "expiry_label": expiry_label,
            "fallback_iv": fallback_iv, "iv_coverage": iv_pct,
            "source": "barchart",
        }

    except Exception as e:
        logger.error("Barchart fetch failed for %s: %s", ticker, e)
        return None


# ═══════════════════════════════════════
# tvdatafeed Helpers
# ═══════════════════════════════════════
def _get_tv_spot(ticker):
    """Get spot price from tvdatafeed (fallback for Barchart spot)."""
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
    """
    Fetch intraday 5-min candlestick data from tvdatafeed.
    Returns DataFrame with Open/High/Low/Close/Volume columns, or empty DF.
    """
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
        logger.warning("tvdatafeed not installed — no price chart")
    return pd.DataFrame()


# ═══════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════
def fetch_options_data(ticker_str, expiry_offset=0):
    """Fetch from Barchart. No yfinance fallback."""
    return fetch_barchart_options(ticker_str, expiry_offset)
