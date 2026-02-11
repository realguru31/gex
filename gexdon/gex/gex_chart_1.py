"""
Chart 1 — Gamma Exposure Density by Strike (Plotly)
Multi-DTE aware: each option row has its own t_expiry.
Uses Barchart pre-computed gamma when available, BS fallback.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_gex_chart(data, percent_range=0.03):
    spot    = data["spot"]
    calls   = data["calls"]
    puts    = data["puts"]
    fb_iv   = data["fallback_iv"]
    exp_lbl = data["expiry_label"]
    source  = data.get("source", "barchart")
    multi   = data.get("multi_dte", False)

    # Build per-strike aggregated GEX
    all_K = sorted(set(
        calls["strikePrice"].dropna().tolist() +
        puts["strikePrice"].dropna().tolist()
    ))
    rows = []
    for K in all_K:
        if pd.isna(K):
            continue
        cg, pg = 0.0, 0.0

        # Sum across all expiries for this strike
        cd = calls[calls["strikePrice"] == K]
        for _, row in cd.iterrows():
            OI = int(row["openInterest"])
            if OI <= 0:
                continue
            t_exp = row.get("t_expiry", data["t_expiry"])
            bc_gamma = float(row.get("gamma", 0))
            if bc_gamma > 0 and source == "barchart":
                cg += bc_gamma * OI * 100
            else:
                iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
                if iv:
                    _, g, _ = bs_greeks(spot, K, t_exp, R, iv, "call")
                    cg += g * OI * 100

        pd_ = puts[puts["strikePrice"] == K]
        for _, row in pd_.iterrows():
            OI = int(row["openInterest"])
            if OI <= 0:
                continue
            t_exp = row.get("t_expiry", data["t_expiry"])
            bc_gamma = float(row.get("gamma", 0))
            if bc_gamma > 0 and source == "barchart":
                pg += bc_gamma * OI * 100
            else:
                iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
                if iv:
                    _, g, _ = bs_greeks(spot, K, t_exp, R, iv, "put")
                    pg += g * OI * 100

        rows.append({"strike": K, "call_gex": cg, "put_gex": pg,
                      "net_gex": cg - pg, "sell_gamma": cg, "buy_gamma": pg})

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_gex"] != 0) | (df["put_gex"] != 0)]
    if len(df) < 2:
        return _empty("No Gamma data"), {}

    levels = _extract_levels(df, spot)

    # Smoothing
    st = df["strike"].values
    sg = np.abs(df["sell_gamma"].values)
    bg = np.abs(df["buy_gamma"].values)
    ng = np.abs(df["net_gex"].values)
    tg = sg + bg
    if len(st) > 3:
        sg = gaussian_filter1d(sg, 1.5)
        bg = gaussian_filter1d(bg, 1.5)
        ng = gaussian_filter1d(ng, 1.5)
        tg = gaussian_filter1d(tg, 1.5)

    iv_tag = "Barchart IV" if fb_iv is None else f"Est IV {fb_iv:.3f}"
    title = f"Gamma Density ({exp_lbl}) — {iv_tag} [{source.upper()}]"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st, y=sg, name="Sell Γ", line=dict(color=CS["green"], width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=st, y=bg, name="Buy Γ", line=dict(color=CS["red"], width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=st, y=ng, name="Net GEX", line=dict(color=CS["gold"], width=3), opacity=0.9))
    fig.add_trace(go.Scatter(x=st, y=tg, name="Total |Γ|", line=dict(color=CS["orange"], width=2), opacity=0.7))
    fig.add_vline(x=spot, line=dict(color=CS["cyan"], width=2, dash="dash"),
                  annotation_text=f"Spot ${spot:.2f}", annotation_font_color=CS["cyan"])

    fig.update_layout(
        title=dict(text=title, font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=10),
        xaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat="$,.0f"),
        yaxis=dict(gridcolor=CS["grid"], title="Gamma (|val|)", rangemode="tozero"),
        legend=dict(bgcolor=CS["plot_bg"], bordercolor=CS["grid"], font=dict(size=9, color=CS["text"])),
        margin=dict(l=50, r=20, t=40, b=40),
        height=280,
    )
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


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=CS["text"], size=16))
    fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                      height=280, margin=dict(l=20, r=20, t=20, b=20))
    return fig
