"""
Shared utilities: Black-Scholes Greeks, Barchart data fetching, IV fallback.
Primary data source: Barchart (options chain with pre-computed Greeks).
Fallback: yfinance (if Barchart fails).

Unified columns downstream: strikePrice, openInterest, gamma, iv_decimal,
                             delta, volatility, theta, vega
"""
import requests
import yfinance as yf
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
from urllib.parse import unquote
from config import RISK_FREE_RATE
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════
# Black-Scholes Greeks
# ═══════════════════════════════════════
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Return (delta, gamma, charm) for a European option."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    n_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = n_d1 / (S * sigma * sqrt(T))
    charm = -exp(-r * T) * n_d1 * (r / (sigma * sqrt(T)) - d1 / (2 * T))
    return delta, gamma, charm


# ═══════════════════════════════════════
# IV Fallbacks
# ═══════════════════════════════════════
def get_vix_as_iv():
    try:
        return float(yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]) / 100.0
    except Exception:
        return None


def estimate_iv_historical(ticker_obj, days=30):
    try:
        hist = ticker_obj.history(period=f"{days}d")
        if len(hist) < 2:
            return None
        rets = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        return float(rets.std() * np.sqrt(252))
    except Exception:
        return None


def resolve_iv(row_iv, fallback_iv):
    """Return usable IV: chain IV if valid, else fallback."""
    if row_iv and row_iv > 0:
        return row_iv
    if fallback_iv is not None and fallback_iv > 0:
        return fallback_iv
    return None


# ═══════════════════════════════════════
# BARCHART Data Fetching (PRIMARY)
# ═══════════════════════════════════════
def _bc_page_url(sym):
    if sym in ("SPX", "^SPX", "$SPX"):
        return "https://www.barchart.com/stocks/quotes/$SPX/volatility-greeks"
    return f"https://www.barchart.com/etfs-funds/quotes/{sym}/volatility-greeks"


def _bc_base_sym(sym):
    if sym in ("SPX", "^SPX"):
        return "$SPX"
    return sym


def _yf_sym(sym):
    """Convert display ticker to yfinance symbol."""
    if sym in ("SPX", "$SPX"):
        return "^SPX"
    return sym


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    """
    Fetch options chain from Barchart with pre-computed Greeks.

    Returns unified dict:
        spot, expiry, calls, puts, t_expiry, expiry_label,
        fallback_iv, iv_coverage, source="barchart"

    Columns in calls/puts:
        strikePrice, openInterest, gamma, iv_decimal,
        delta, theta, vega, volume, lastPrice, volatility
    """
    MIN_T = 1 / (24 * 60)

    # ── 1. Expiry dates + spot via yfinance (lightweight call) ──
    yf_tkr = yf.Ticker(_yf_sym(ticker_symbol))
    try:
        expiry_dates = list(yf_tkr.options)
    except Exception:
        expiry_dates = []
    if not expiry_dates:
        logger.warning("No expiry dates for %s", ticker_symbol)
        return None

    expiry_offset = min(expiry_offset, len(expiry_dates) - 1)
    expiry = expiry_dates[expiry_offset]

    try:
        hist = yf_tkr.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
    except Exception:
        return None

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    t_exp = max(MIN_T, (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
    expiry_label = "0DTE" if expiry_dt.date() == datetime.now().date() else f"{t_exp * 365:.1f}d"

    # ── 2. Barchart session + XSRF token ──
    page_url = _bc_page_url(ticker_symbol)
    api_url = "https://www.barchart.com/proxies/core-api/v1/options/get"

    try:
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

        cookies = sess.cookies.get_dict()
        if "XSRF-TOKEN" not in cookies:
            logger.warning("No XSRF-TOKEN in Barchart cookies")
            return None
        xsrf = unquote(cookies["XSRF-TOKEN"])

        # ── 3. API call ──
        r = sess.get(
            api_url,
            params={
                "baseSymbol": _bc_base_sym(ticker_symbol),
                "groupBy": "optionType",
                "expirationDate": expiry,
                "orderBy": "strikePrice",
                "orderDir": "asc",
                "raw": "1",
                "fields": "symbol,strikePrice,lastPrice,volatility,delta,gamma,theta,vega,volume,openInterest,optionType",
            },
            headers={
                "accept": "application/json",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "en-US,en;q=0.9",
                "referer": page_url,
                "user-agent": _UA,
                "x-xsrf-token": xsrf,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        # ── 4. Parse ──
        rows = []
        for opt_type, options in data.get("data", {}).items():
            for opt in options:
                opt["optionType"] = opt_type
                rows.append(opt)
        if not rows:
            logger.warning("Barchart: empty data for %s", ticker_symbol)
            return None

        df = pd.DataFrame(rows)
        for col in ["strikePrice", "lastPrice", "volatility", "delta",
                     "gamma", "theta", "vega", "volume", "openInterest"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["openInterest"] = df["openInterest"].astype(int)

        # Barchart 'volatility' is IV in % (e.g. 15.2 = 15.2%), convert to decimal
        df["iv_decimal"] = df["volatility"] / 100.0

        calls = df[df["optionType"] == "Call"].copy().reset_index(drop=True)
        puts = df[df["optionType"] == "Put"].copy().reset_index(drop=True)

        # IV quality check
        n_valid = len(calls[calls["iv_decimal"] > 0]) + len(puts[puts["iv_decimal"] > 0])
        n_total = max(len(calls) + len(puts), 1)
        iv_pct = n_valid / n_total * 100.0
        fallback_iv = None
        if iv_pct < 50:
            fallback_iv = get_vix_as_iv() or estimate_iv_historical(yf_tkr) or 0.15

        logger.info("Barchart OK: %s spot=$%.2f exp=%s calls=%d puts=%d iv=%.0f%%",
                     ticker_symbol, spot, expiry, len(calls), len(puts), iv_pct)

        return {
            "ticker": ticker_symbol, "spot": spot,
            "calls": calls, "puts": puts,
            "expiry": expiry, "expiry_dt": expiry_dt,
            "expiry_dates": expiry_dates,
            "t_expiry": t_exp, "expiry_label": expiry_label,
            "fallback_iv": fallback_iv, "iv_coverage": iv_pct,
            "source": "barchart",
        }

    except Exception as e:
        logger.error("Barchart fetch failed for %s: %s", ticker_symbol, e)
        return None


# ═══════════════════════════════════════
# YFINANCE Fallback
# ═══════════════════════════════════════
def fetch_yfinance_options(ticker_symbol, expiry_offset=0):
    """
    Fallback: yfinance options chain.
    Normalizes columns to match Barchart naming convention.
    """
    MIN_T = 1 / (24 * 60)
    yf_s = _yf_sym(ticker_symbol)
    ticker = yf.Ticker(yf_s)

    try:
        spot = float(ticker.history(period="1d", interval="1m")["Close"].iloc[-1])
    except Exception:
        try:
            spot = float(ticker.history(period="5d")["Close"].iloc[-1])
        except Exception:
            return None

    exps = ticker.options
    if not exps:
        return None
    idx = min(expiry_offset, len(exps) - 1)
    expiry = exps[idx]
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    chain = ticker.option_chain(expiry)

    t_exp = max(MIN_T, (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
    expiry_label = "0DTE" if expiry_dt.date() == datetime.now().date() else f"{t_exp * 365:.1f}d"

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    # Normalize columns to Barchart naming
    rename_map = {"strike": "strikePrice", "impliedVolatility": "iv_decimal"}
    for df in [calls, puts]:
        df.rename(columns=rename_map, inplace=True)
        for col in ["gamma", "delta", "theta", "vega"]:
            if col not in df.columns:
                df[col] = 0.0
        if "volatility" not in df.columns:
            df["volatility"] = df["iv_decimal"] * 100.0

    n_valid = len(calls[calls["iv_decimal"] > 0]) + len(puts[puts["iv_decimal"] > 0])
    n_total = max(len(calls) + len(puts), 1)
    iv_pct = n_valid / n_total * 100.0
    fallback_iv = None
    if iv_pct < 50:
        fallback_iv = get_vix_as_iv() or estimate_iv_historical(ticker) or 0.15

    return {
        "ticker": ticker_symbol, "spot": spot,
        "calls": calls, "puts": puts,
        "expiry": expiry, "expiry_dt": expiry_dt,
        "expiry_dates": list(exps),
        "t_expiry": t_exp, "expiry_label": expiry_label,
        "fallback_iv": fallback_iv, "iv_coverage": iv_pct,
        "source": "yfinance",
    }


# ═══════════════════════════════════════
# UNIFIED FETCH: Barchart → yfinance
# ═══════════════════════════════════════
def fetch_options_data(ticker_str, expiry_offset=0):
    """Try Barchart first; fall back to yfinance."""
    data = fetch_barchart_options(ticker_str, expiry_offset)
    if data is not None:
        return data
    logger.info("Falling back to yfinance for %s", ticker_str)
    return fetch_yfinance_options(ticker_str, expiry_offset)


# ═══════════════════════════════════════
# Price Data (yfinance intraday)
# ═══════════════════════════════════════
def fetch_price_data(ticker_str, period="1d", interval="5m"):
    yf_s = _yf_sym(ticker_str)
    try:
        df = yf.Ticker(yf_s).history(period=period, interval=interval)
        if df.empty:
            df = yf.Ticker(yf_s).history(period="5d", interval="15m")
        return df
    except Exception:
        return pd.DataFrame()
