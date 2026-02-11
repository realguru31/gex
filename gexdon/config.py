"""
GEXdon Configuration — ZERO yfinance.
"""
import pytz
from datetime import time

NYSE_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 30)
SNAPSHOT_INTERVAL_MIN = 30
AUTO_REFRESH_SEC = 300

# Removed ^SPX (Barchart incompatible)
TICKERS = ["SPY", "QQQ"]
TICKER_DISPLAY = {"SPY": "SPY", "QQQ": "QQQ"}

# tvdatafeed mappings (price chart only)
TV_EXCHANGE = {"SPY": "AMEX", "QQQ": "NASDAQ"}
TV_SYMBOL   = {"SPY": "SPY",  "QQQ": "QQQ"}
TV_EXCHANGE_FALLBACKS = ["AMEX", "NYSE", "NASDAQ", "CBOE"]

# Barchart mappings
BC_BASE_SYM  = {"SPY": "SPY", "QQQ": "QQQ"}
BC_PAGE_TYPE = {"SPY": "etfs-funds", "QQQ": "etfs-funds"}

STRIKE_RANGES = [0.02, 0.03, 0.04, 0.05]
STRIKE_RANGE_LABELS = {0.02: "2%", 0.03: "3%", 0.04: "4%", 0.05: "5%"}

# DTE options: value = max DTE to aggregate (0 = 0DTE only, 3 = 0+1+2+3 DTE)
DTE_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7]
DTE_LABELS = {
    0: "0 DTE", 1: "0-1 DTE", 2: "0-2 DTE", 3: "0-3 DTE",
    4: "0-4 DTE", 5: "0-5 DTE", 6: "0-6 DTE", 7: "0-7 DTE",
}

RISK_FREE_RATE = 0.05
SNAPSHOT_DIR = "data/snapshots"

# Market profile period labels (30-min buckets starting 09:30)
PERIOD_LABELS = {
    "0930": "A", "1000": "B", "1030": "C", "1100": "D",
    "1130": "E", "1200": "F", "1230": "G", "1300": "H",
    "1330": "I", "1400": "J", "1430": "K", "1500": "L",
    "1530": "M", "1600": "N", "1630": "N+",
}

CS = {
    "bg":       "#050b1e",
    "plot_bg":  "#0d1f3c",
    "grid":     "#1a3a5c",
    "text":     "#c8d6e5",
    "cyan":     "#00d2ff",
    "blue":     "#1e90ff",
    "green":    "#00ff88",
    "red":      "#ff4757",
    "gold":     "#ffd700",
    "orange":   "#ff8c00",
    "purple":   "#a855f7",
}
