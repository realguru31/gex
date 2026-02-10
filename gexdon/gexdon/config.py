"""
GEXdon Configuration
NYSE hours, intervals, theme colors, supported tickers.
"""
import pytz
from datetime import time

# ── NYSE Trading Hours (Eastern Time) ──
NYSE_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 30)
SNAPSHOT_INTERVAL_MIN = 30
AUTO_REFRESH_SEC = 300  # 5 minutes

# ── Supported tickers ──
TICKERS = ["SPY", "QQQ", "^SPX"]
TICKER_DISPLAY = {"SPY": "SPY", "QQQ": "QQQ", "^SPX": "S&P 500 (^SPX)"}

# ── Strike range options ──
STRIKE_RANGES = [0.02, 0.03, 0.04, 0.05]
STRIKE_RANGE_LABELS = {0.02: "2%", 0.03: "3%", 0.04: "4%", 0.05: "5%"}

# ── Risk-free rate ──
RISK_FREE_RATE = 0.05

# ── Data paths (relative to app root) ──
SNAPSHOT_DIR = "data/snapshots"
PRICE_CACHE_DIR = "data/price_cache"

# ── Chart theme (dark blue) ──
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
