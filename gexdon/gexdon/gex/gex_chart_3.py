"""
Chart 3 — Charm Pressure & Acceleration (dual-axis)
Charm computed via BS, then OI-weighted into pressure + gradient acceleration.
Uses unified columns: strikePrice, openInterest, iv_decimal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_pressure_chart(data, percent_range=0.03):
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
        cc, pc, co, po = 0.0, 0.0, 0, 0

        cd = calls[calls["strikePrice"] == K]
        if not cd.empty:
            row = cd.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "call")
                cc = ch * oi * 100; co = oi

        pd_ = puts[puts["strikePrice"] == K]
        if not pd_.empty:
            row = pd_.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "put")
                pc = ch * oi * 100; po = oi

        rows.append({"strike": K, "net_charm": cc + pc,
                      "total_oi": co + po, "call_charm": cc, "put_charm": pc})

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_charm"] != 0) | (df["put_charm"] != 0)]
    if len(df) < 3:
        return _empty("Insufficient data for Pressure"), {}

    # ── pressure math ──
    ks       = df["strike"].values
    nc       = df["net_charm"].values
    toi      = df["total_oi"].values
    oi_sum   = toi.sum()
    w        = toi / oi_sum if oi_sum > 0 else np.ones_like(toi) / len(toi)
    pressure = nc * w * 1e3
    accel    = np.gradient(pressure)

    # ── key levels ──
    pi = int(np.argmax(np.abs(pressure)))
    ai = int(np.argmax(np.abs(accel)))
    levels = {
        "max_pressure_strike": float(ks[pi]),
        "max_pressure_val":    float(pressure[pi]),
        "max_accel_strike":    float(ks[ai]),
    }
    # equilibrium (zero-crossing nearest spot)
    sc = np.where(np.diff(np.sign(pressure)))[0]
    if len(sc):
        crosses = []
        for i in sc:
            p1, p2, s1, s2 = pressure[i], pressure[i+1], ks[i], ks[i+1]
            if p2 - p1 != 0:
                crosses.append(s1 + (s2 - s1) * (-p1) / (p2 - p1))
        if crosses:
            levels["pressure_eq"] = float(min(crosses, key=lambda x: abs(x - spot)))

    fig = _plot(ks, pressure, accel, spot, exp_lbl, source)
    return fig, levels


def _plot(ks, pressure, accel, spot, exp_lbl, source):
    fig, ax1 = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(CS["bg"]); ax1.set_facecolor(CS["plot_bg"])

    sig = 1.5 if len(ks) > 3 else 0
    ps = gaussian_filter1d(pressure, sig) if sig else pressure
    acs = gaussian_filter1d(accel, sig) if sig else accel

    ax1.plot(ks, ps, color=CS["purple"], lw=3, label="Charm Pressure", zorder=3)
    ax1.fill_between(ks, 0, ps, where=(ps > 0), color=CS["green"], alpha=.12)
    ax1.fill_between(ks, 0, ps, where=(ps < 0), color=CS["red"],   alpha=.12)
    ax1.axhline(0, color=CS["text"], alpha=.15, lw=.8)
    ax1.set_ylabel("Pressure", color=CS["purple"], fontsize=9)
    ax1.tick_params(axis="y", labelcolor=CS["purple"], labelsize=7)

    ax2 = ax1.twinx()
    ax2.plot(ks, acs, color="#ff4444", ls="--", lw=2, label="Acceleration", alpha=.8)
    ax2.set_ylabel("Acceleration", color="#ff4444", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#ff4444", labelsize=7)
    for sp in ax2.spines.values(): sp.set_color(CS["grid"])

    ax1.axvline(spot, color=CS["cyan"], ls="--", lw=2, alpha=.8, zorder=4)
    ax1.grid(True, alpha=.12, color=CS["grid"])
    ax1.tick_params(axis="x", colors=CS["text"], labelsize=7)
    for sp in ax1.spines.values(): sp.set_color(CS["grid"])

    src_tag = f"[{source.upper()}]"
    ax1.set_title(f"Charm Pressure & Acceleration ({exp_lbl}) {src_tag}",
                  color=CS["text"], fontsize=12, fontweight="bold")
    ax1.set_xlabel("Strike", color=CS["text"], fontsize=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5,
               facecolor=CS["plot_bg"], edgecolor=CS["grid"], labelcolor=CS["text"])

    step = 1 if len(ks) <= 50 else 2
    t = ks[::step]
    ax1.set_xticks(t); ax1.set_xticklabels([f"${x:.0f}" for x in t], rotation=45, fontsize=6)
    fig.tight_layout()
    return fig


def _empty(msg):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])
    ax.text(.5, .5, msg, transform=ax.transAxes, ha="center", va="center",
            color=CS["text"], fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); return fig
