"""
Chart 3 — Charm Pressure & Acceleration
OI-weighted charm pressure + gradient-based acceleration.
Uses strikePrice, openInterest, iv_decimal from Barchart data.
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
    source  = data.get("source", "barchart")

    all_K = sorted(set(
        calls["strikePrice"].dropna().tolist() +
        puts["strikePrice"].dropna().tolist()
    ))
    rows = []
    for K in all_K:
        if pd.isna(K):
            continue
        cc, pc, c_oi, p_oi = 0.0, 0.0, 0, 0

        cd = calls[calls["strikePrice"] == K]
        if not cd.empty:
            row = cd.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "call")
                cc = ch * oi * 100
                c_oi = oi

        pd_ = puts[puts["strikePrice"] == K]
        if not pd_.empty:
            row = pd_.iloc[0]
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "put")
                pc = ch * oi * 100
                p_oi = oi

        rows.append({
            "strike": K, "net_charm": cc + pc,
            "total_oi": c_oi + p_oi,
            "call_charm": cc, "put_charm": pc,
        })

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_charm"] != 0) | (df["put_charm"] != 0)]
    if len(df) < 2:
        return _empty("Insufficient data for Charm Pressure"), {}

    # Pressure metrics
    strikes   = df["strike"].values
    net_charm = df["net_charm"].values
    total_oi  = df["total_oi"].values

    oi_sum = total_oi.sum()
    oi_w = total_oi / oi_sum if oi_sum > 0 else np.ones_like(total_oi) / len(total_oi)
    charm_pressure = net_charm * oi_w * 1e3
    accel = np.gradient(charm_pressure)

    # Key levels
    max_p_idx = np.argmax(np.abs(charm_pressure))
    max_a_idx = np.argmax(np.abs(accel))
    levels = {
        "max_pressure_strike": float(strikes[max_p_idx]),
        "max_pressure_val":    float(charm_pressure[max_p_idx]),
        "max_accel_strike":    float(strikes[max_a_idx]),
    }
    # Pressure equilibrium (zero-crossing nearest spot)
    sc = np.where(np.diff(np.sign(charm_pressure)))[0]
    if len(sc):
        crosses = []
        for i in sc:
            s1, s2 = strikes[i], strikes[i+1]
            p1, p2 = charm_pressure[i], charm_pressure[i+1]
            if p2 - p1 != 0:
                crosses.append(s1 + (s2 - s1) * (-p1) / (p2 - p1))
        if crosses:
            levels["pressure_eq"] = float(min(crosses, key=lambda x: abs(x - spot)))

    # ── plot ──
    fig, ax1 = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(CS["bg"]); ax1.set_facecolor(CS["plot_bg"])

    s = 1.5 if len(strikes) > 3 else 0
    ps = gaussian_filter1d(charm_pressure, s) if s else charm_pressure
    ac = gaussian_filter1d(accel, s) if s else accel

    ax1.plot(strikes, ps, color=CS["purple"], lw=3, label="Charm Pressure", zorder=3)
    ax1.fill_between(strikes, 0, ps, where=(ps > 0), color=CS["green"], alpha=.15, zorder=2)
    ax1.fill_between(strikes, 0, ps, where=(ps < 0), color=CS["red"],   alpha=.15, zorder=2)
    ax1.set_ylabel("Pressure", color=CS["purple"], fontsize=10)
    ax1.tick_params(axis="y", labelcolor=CS["purple"], labelsize=8)
    ax1.axhline(y=0, color=CS["text"], alpha=.2, lw=.8)

    ax2 = ax1.twinx()
    ax2.plot(strikes, ac, color="#ff4444", ls="--", lw=2, label="Acceleration", alpha=.8)
    ax2.set_ylabel("Acceleration", color="#ff4444", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#ff4444", labelsize=8)

    ax1.axvline(spot, color=CS["cyan"], ls="--", lw=2, alpha=.8, zorder=4)

    # Styling
    ax1.grid(True, alpha=.15, color=CS["grid"])
    ax1.tick_params(axis="x", colors=CS["text"], labelsize=7)
    for sp in ax1.spines.values(): sp.set_color(CS["grid"])
    for sp in ax2.spines.values(): sp.set_color(CS["grid"])

    src_tag = source.upper()
    ax1.set_title(f"Charm Pressure & Acceleration ({exp_lbl}) [{src_tag}]",
                   color=CS["text"], fontsize=12, fontweight="bold")
    ax1.set_xlabel("Strike", color=CS["text"], fontsize=8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.5,
               facecolor=CS["plot_bg"], edgecolor=CS["grid"], labelcolor=CS["text"])

    step = 1 if len(strikes) <= 50 else 2
    t = strikes[::step]
    ax1.set_xticks(t); ax1.set_xticklabels([f"${x:.0f}" for x in t], rotation=45, fontsize=6)

    fig.tight_layout()
    return fig, levels


def _empty(msg):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])
    ax.text(.5, .5, msg, transform=ax.transAxes, ha="center", va="center",
            color=CS["text"], fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); return fig
