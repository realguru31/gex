"""
Shared utilities: Black-Scholes Greeks, IV fallback, data fetching.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
from config import RISK_FREE_RATE


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
# IV Estimation Fallbacks
# ═══════════════════════════════════════
def get_vix_as_iv():
    """Use VIX level as SPY IV proxy."""
    try:
        vix = yf.Ticker("^VIX")
        return float(vix.history(period="1d")["Close"].iloc[-1]) / 100.0
    except Exception:
        return None


def estimate_iv_historical(ticker_obj, days=30):
    """Historical realized volatility as IV fallback."""
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
    if row_iv > 0:
        return row_iv
    if fallback_iv is not None and fallback_iv > 0:
        return fallback_iv
    return None


# ═══════════════════════════════════════
# Data Fetching
# ═══════════════════════════════════════
def fetch_options_data(ticker_str, expiry_offset=0):
    """
    Fetch spot price, nearest options chain, compute time-to-expiry.
    Returns a dict consumed by all three chart generators, or None on failure.
    """
    MIN_T = 1 / (24 * 60)
    ticker = yf.Ticker(ticker_str)

    # Spot price
    try:
        spot = float(ticker.history(period="1d", interval="1m")["Close"].iloc[-1])
    except Exception:
        try:
            spot = float(ticker.history(period="5d")["Close"].iloc[-1])
        except Exception:
            return None

    # Options chain
    expirations = ticker.options
    if not expirations:
        return None
    idx = min(expiry_offset, len(expirations) - 1)
    expiry = expirations[idx]
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    chain = ticker.option_chain(expiry)
    calls, puts = chain.calls, chain.puts

    # Time to expiry (annualized)
    t_exp = max(MIN_T, (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
    is_0dte = expiry_dt.date() == datetime.now().date()
    label = "0DTE" if is_0dte else f"{t_exp * 365:.1f}d"

    # IV quality check + fallback
    n_valid = len(calls[calls["impliedVolatility"] > 0]) + len(puts[puts["impliedVolatility"] > 0])
    n_total = max(len(calls) + len(puts), 1)
    iv_pct = n_valid / n_total * 100.0

    fallback_iv = None
    if iv_pct < 50:
        fallback_iv = get_vix_as_iv() or estimate_iv_historical(ticker) or 0.15

    return {
        "ticker": ticker_str,
        "spot": spot,
        "calls": calls,
        "puts": puts,
        "expiry": expiry,
        "expiry_dt": expiry_dt,
        "t_expiry": t_exp,
        "expiry_label": label,
        "fallback_iv": fallback_iv,
        "iv_coverage": iv_pct,
    }


def fetch_price_data(ticker_str, period="1d", interval="5m"):
    """Fetch intraday OHLC for the price chart."""
    try:
        df = yf.Ticker(ticker_str).history(period=period, interval=interval)
        if df.empty:
            df = yf.Ticker(ticker_str).history(period="5d", interval="15m")
        return df
    except Exception:
        return pd.DataFrame()
