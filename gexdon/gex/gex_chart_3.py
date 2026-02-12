"""
Chart 3 — Charm Pressure & Acceleration (Plotly)
Multi-DTE: uses per-row t_expiry. Dual y-axis.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_pressure_chart(data, percent_range=0.03):
    spot    = data["spot"]
    calls   = data["calls"]
    puts    = data["puts"]
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
        for _, row in cd.iterrows():
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            t_exp = row.get("t_expiry", data["t_expiry"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "call")
                cc += ch * oi * 100
                c_oi += oi

        pd_ = puts[puts["strikePrice"] == K]
        for _, row in pd_.iterrows():
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            oi = int(row["openInterest"])
            t_exp = row.get("t_expiry", data["t_expiry"])
            if iv and oi > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "put")
                pc += ch * oi * 100
                p_oi += oi

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

    strikes   = df["strike"].values
    net_charm = df["net_charm"].values
    total_oi  = df["total_oi"].values

    oi_sum = total_oi.sum()
    oi_w = total_oi / oi_sum if oi_sum > 0 else np.ones_like(total_oi) / len(total_oi)
    charm_pressure = net_charm * oi_w * 1e3
    accel = np.gradient(charm_pressure)

    # Smoothing
    if len(strikes) > 3:
        ps = gaussian_filter1d(charm_pressure, 1.5)
        ac = gaussian_filter1d(accel, 1.5)
    else:
        ps, ac = charm_pressure, accel

    # Key levels
    max_p_idx = np.argmax(np.abs(ps))
    max_a_idx = np.argmax(np.abs(ac))
    levels = {
        "max_pressure_strike": float(strikes[max_p_idx]),
        "max_pressure_val":    float(ps[max_p_idx]),
        "max_accel_strike":    float(strikes[max_a_idx]),
    }
    sc = np.where(np.diff(np.sign(ps)))[0]
    if len(sc):
        crosses = []
        for i in sc:
            s1, s2 = strikes[i], strikes[i+1]
            p1, p2 = ps[i], ps[i+1]
            if p2 - p1 != 0:
                crosses.append(s1 + (s2 - s1) * (-p1) / (p2 - p1))
        if crosses:
            levels["pressure_eq"] = float(min(crosses, key=lambda x: abs(x - spot)))

    title = f"Charm Pressure & Acceleration ({exp_lbl}) [{source.upper()}]"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Pressure fill regions
    pos_y = np.where(ps > 0, ps, 0)
    neg_y = np.where(ps < 0, ps, 0)
    fig.add_trace(go.Scatter(x=strikes, y=pos_y, fill="tozeroy", fillcolor="rgba(0,255,136,0.12)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), secondary_y=False)
    fig.add_trace(go.Scatter(x=strikes, y=neg_y, fill="tozeroy", fillcolor="rgba(255,71,87,0.12)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), secondary_y=False)

    # Pressure line
    fig.add_trace(go.Scatter(x=strikes, y=ps, name="Charm Pressure",
                             line=dict(color=CS["purple"], width=3)), secondary_y=False)
    # Acceleration line
    fig.add_trace(go.Scatter(x=strikes, y=ac, name="Acceleration",
                             line=dict(color="#ff4444", width=2, dash="dash"), opacity=0.8), secondary_y=True)

    fig.add_vline(x=spot, line=dict(color=CS["cyan"], width=2, dash="dash"))
    fig.add_hline(y=0, line=dict(color=CS["text"], width=0.8), opacity=0.2, secondary_y=False)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=10),
        xaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat="$,.0f"),
        legend=dict(bgcolor="rgba(13,31,60,0.9)", bordercolor=CS["grid"],
                    font=dict(size=10, color="#ffffff")),
        margin=dict(l=50, r=50, t=40, b=40),
        height=440,
    )
    fig.update_yaxes(title=dict(text="Pressure", font=dict(color=CS["purple"])),
                     tickfont=dict(color=CS["purple"]), gridcolor=CS["grid"], secondary_y=False)
    fig.update_yaxes(title=dict(text="Acceleration", font=dict(color="#ff4444")),
                     tickfont=dict(color="#ff4444"), gridcolor=CS["grid"], secondary_y=True)

    return fig, levels


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=CS["text"], size=16))
    fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                      height=440, margin=dict(l=20, r=20, t=20, b=20))
    return fig
