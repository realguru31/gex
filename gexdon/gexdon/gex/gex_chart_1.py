"""
Chart 1 — Gamma Exposure Density by Strike
Uses Barchart's pre-computed gamma when available (source='barchart').
Falls back to BS gamma computation when source='yfinance'.
Unified columns: strikePrice, openInterest, gamma, iv_decimal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_gex_chart(data, percent_range=0.03):
    spot      = data["spot"]
    calls     = data["calls"]
    puts      = data["puts"]
    t_exp     = data["t_expiry"]
    fb_iv     = data["fallback_iv"]
    exp_lbl   = data["expiry_label"]
    source    = data.get("source", "yfinance")
    use_bc_g  = source == "barchart"  # use Barchart gamma directly

    all_K = sorted(set(
        calls["strikePrice"].dropna().tolist() +
        puts["strikePrice"].dropna().tolist()
    ))
    rows = []
    for K in all_K:
        if pd.isna(K):
            continue
        cg, pg = 0.0, 0.0

        cd = calls[calls["strikePrice"] == K]
        if not cd.empty:
            row = cd.iloc[0]
            oi = int(row["openInterest"])
            if oi > 0:
                if use_bc_g and row["gamma"] > 0:
                    cg = row["gamma"] * oi * 100
                else:
                    iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
                    if iv:
                        _, g, _ = bs_greeks(spot, K, t_exp, R, iv, "call")
                        cg = g * oi * 100

        pd_ = puts[puts["strikePrice"] == K]
        if not pd_.empty:
            row = pd_.iloc[0]
            oi = int(row["openInterest"])
            if oi > 0:
                if use_bc_g and row["gamma"] > 0:
                    pg = row["gamma"] * oi * 100
                else:
                    iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
                    if iv:
                        _, g, _ = bs_greeks(spot, K, t_exp, R, iv, "put")
                        pg = g * oi * 100

        rows.append({"strike": K, "call_gex": cg, "put_gex": pg,
                      "net_gex": cg - pg, "sell_gamma": cg, "buy_gamma": pg})

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_gex"] != 0) | (df["put_gex"] != 0)]
    if len(df) < 2:
        return _empty("No Gamma data"), {}

    levels = _extract_levels(df, spot)
    fig = _plot(df, spot, exp_lbl, fb_iv, source)
    return fig, levels


def _extract_levels(df, spot):
    lv = {}
    lv["max_gamma"] = float(df.loc[df["net_gex"].abs().idxmax(), "strike"])
    net = df["net_gex"].values; ks = df["strike"].values
    sc = np.where(np.diff(np.sign(net)))[0]
    if len(sc):
        crosses = []
        for i in sc:
            n1, n2, s1, s2 = net[i], net[i+1], ks[i], ks[i+1]
            if n2 - n1 != 0:
                crosses.append(s1 + (s2 - s1) * (-n1) / (n2 - n1))
        if crosses:
            lv["zero_gamma"] = float(min(crosses, key=lambda x: abs(x - spot)))
    lv["call_wall"] = float(df.loc[df["call_gex"].idxmax(), "strike"])
    lv["put_wall"]  = float(df.loc[df["put_gex"].idxmax(),  "strike"])
    return lv


def _plot(df, spot, exp_lbl, fb_iv, source):
    fig, ax = plt.subplots(figsize=(14, 4.2))
    fig.patch.set_facecolor(CS["bg"]); ax.set_facecolor(CS["plot_bg"])
    st = df["strike"].values
    sg = np.abs(df["sell_gamma"].values)
    bg = np.abs(df["buy_gamma"].values)
    ng = np.abs(df["net_gex"].values)
    tg = sg + bg
    s = 1.5 if len(st) > 3 else 0
    if s:
        sg = gaussian_filter1d(sg, s); bg = gaussian_filter1d(bg, s)
        ng = gaussian_filter1d(ng, s); tg = gaussian_filter1d(tg, s)
    ax.plot(st, sg, color=CS["green"],  lw=2.5, label="Sell Γ",   alpha=.8)
    ax.plot(st, bg, color=CS["red"],    lw=2.5, label="Buy Γ",    alpha=.8)
    ax.plot(st, ng, color=CS["gold"],   lw=3,   label="Net GEX",  alpha=.9)
    ax.plot(st, tg, color=CS["orange"], lw=2,   label="Total |Γ|",alpha=.7)
    ax.axvline(spot, color=CS["cyan"], ls="--", lw=2,
               label=f"Spot ${spot:.2f}", alpha=.8)
    _style(ax, st, fig, exp_lbl, fb_iv, source)
    ax.set_ylim(bottom=0)
    return fig


def _style(ax, strikes, fig, exp_lbl, fb_iv, source):
    ax.grid(True, alpha=.15, color=CS["grid"])
    ax.tick_params(colors=CS["text"], labelsize=7)
    for sp in ax.spines.values(): sp.set_color(CS["grid"])
    src_tag = f"[{source.upper()}]"
    iv_tag = "Live IV" if fb_iv is None else f"Fb IV {fb_iv:.3f}"
    ax.set_title(f"Gamma Density ({exp_lbl}) — {iv_tag} {src_tag}",
                 color=CS["text"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Strike", color=CS["text"], fontsize=8)
    ax.set_ylabel("Gamma (|val|)", color=CS["text"], fontsize=8)
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
