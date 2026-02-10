"""
Chart 2 — Charm Density by Strike
Charm is always computed via Black-Scholes (neither Barchart nor yfinance provide it).
Uses unified columns: strikePrice, openInterest, iv_decimal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_charm_chart(data, percent_range=0.03):
    spot    = data["spot"]
    calls   = data["calls"]
    puts    = data["puts"]
    t_exp   = data["t_expiry"]
    fb_iv   = data["fallback_iv"]
    exp_lbl = data["expiry_label"]
    source  = data.get("source", "yfinance")

    all_K = sorted(set(
        calls["strikePrice"].dropna().tolist() +
        puts["strikePrice"].dropna().tolist()
    ))
    rows = []
    for K in all_K:
        if pd.isna(K):
            continue
        cc, pc = 0.0, 0.0

        cd = calls[calls["strikePrice"] == K]
        if not cd.empty:
            row = cd.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "call")
                cc = ch * oi * 100

        pd_ = puts[puts["strikePrice"] == K]
        if not pd_.empty:
            row = pd_.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "put")
                pc = ch * oi * 100

        rows.append({"strike": K, "call_charm": cc, "put_charm": pc,
                      "net_charm": cc + pc})

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_charm"] != 0) | (df["put_charm"] != 0)]
    if len(df) < 2:
        return _empty("No Charm data"), {}

    levels = {
        "call_charm_exp":   float(df["call_charm"].sum()),
        "put_charm_exp":    float(df["put_charm"].sum()),
        "net_charm_exp":    float(df["call_charm"].sum() + df["put_charm"].sum()),
        "max_charm_strike": float(df.loc[df["net_charm"].abs().idxmax(), "strike"]),
    }

    fig = _plot(df, spot, exp_lbl, fb_iv, source)
    return fig, levels


def _plot(df, spot, exp_lbl, fb_iv, source):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])
    st = df["strike"].values
    cc = np.abs(df["call_charm"].values)
    pc = np.abs(df["put_charm"].values)
    nc = np.abs(df["net_charm"].values)
    tc = cc + pc
    s = 1.5 if len(st) > 3 else 0
    if s:
        cc = gaussian_filter1d(cc, s); pc = gaussian_filter1d(pc, s)
        nc = gaussian_filter1d(nc, s); tc = gaussian_filter1d(tc, s)
    ax.plot(st, cc, color=CS["green"],  lw=2.5, label="Call Charm", alpha=.8)
    ax.plot(st, pc, color=CS["red"],    lw=2.5, label="Put Charm",  alpha=.8)
    ax.plot(st, nc, color=CS["gold"],   lw=3,   label="Net Charm",  alpha=.9)
    ax.plot(st, tc, color=CS["orange"], lw=2,   label="Total |Charm|", alpha=.7)
    ax.axvline(spot, color=CS["cyan"], ls="--", lw=2,
               label=f"Spot ${spot:.2f}", alpha=.8)
    _style(ax, st, fig, exp_lbl, fb_iv, source)
    return fig


def _style(ax, strikes, fig, exp_lbl, fb_iv, source):
    ax.grid(True, alpha=.15, color=CS["grid"])
    ax.tick_params(colors=CS["text"], labelsize=7)
    for sp in ax.spines.values(): sp.set_color(CS["grid"])
    src_tag = f"[{source.upper()}]"
    iv_tag = "Live IV" if fb_iv is None else f"Fb IV {fb_iv:.3f}"
    ax.set_title(f"Charm Density ({exp_lbl}) — {iv_tag} {src_tag}",
                 color=CS["text"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Strike", color=CS["text"], fontsize=8)
    ax.set_ylabel("Charm (|val|)", color=CS["text"], fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5, facecolor=CS["plot_bg"],
              edgecolor=CS["grid"], labelcolor=CS["text"])
    step = 1 if len(strikes) <= 50 else 2
    t = strikes[::step]
    ax.set_xticks(t); ax.set_xticklabels([f"${x:.0f}" for x in t], rotation=45, fontsize=6)
    fig.tight_layout()


def _empty(msg):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])
    ax.text(.5, .5, msg, transform=ax.transAxes, ha="center", va="center",
            color=CS["text"], fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); return fig
