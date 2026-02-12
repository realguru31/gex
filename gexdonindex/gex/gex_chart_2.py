"""
Chart 2 — Charm Density by Strike (Plotly)
Multi-DTE: uses per-row t_expiry for BS charm computation.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


def generate_charm_chart(data, percent_range=0.03):
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
        cc, pc = 0.0, 0.0

        cd = calls[calls["strikePrice"] == K]
        for _, row in cd.iterrows():
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            OI = int(row["openInterest"])
            t_exp = row.get("t_expiry", data["t_expiry"])
            if iv and OI > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "call")
                cc += ch * OI * 100

        pd_ = puts[puts["strikePrice"] == K]
        for _, row in pd_.iterrows():
            iv = resolve_iv(row.get("iv_decimal", 0), fb_iv)
            OI = int(row["openInterest"])
            t_exp = row.get("t_expiry", data["t_expiry"])
            if iv and OI > 0:
                _, _, ch = bs_greeks(spot, K, t_exp, R, iv, "put")
                pc += ch * OI * 100

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

    st = df["strike"].values
    cc = np.abs(df["call_charm"].values)
    pc = np.abs(df["put_charm"].values)
    nc = np.abs(df["net_charm"].values)
    tc = cc + pc
    if len(st) > 3:
        cc = gaussian_filter1d(cc, 1.5)
        pc = gaussian_filter1d(pc, 1.5)
        nc = gaussian_filter1d(nc, 1.5)
        tc = gaussian_filter1d(tc, 1.5)

    iv_tag = "Barchart IV" if fb_iv is None else f"Est IV {fb_iv:.3f}"
    title = f"Charm Density ({exp_lbl}) — {iv_tag} [{source.upper()}]"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st, y=cc, name="Call Charm", line=dict(color=CS["green"], width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=st, y=pc, name="Put Charm", line=dict(color=CS["red"], width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=st, y=nc, name="Net Charm", line=dict(color=CS["gold"], width=3), opacity=0.9))
    fig.add_trace(go.Scatter(x=st, y=tc, name="Total |Charm|", line=dict(color=CS["orange"], width=2), opacity=0.7))
    fig.add_vline(x=spot, line=dict(color=CS["cyan"], width=2, dash="dash"),
                  annotation_text=f"Spot ${spot:.2f}", annotation_font_color=CS["cyan"])

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=10),
        xaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat="$,.0f"),
        yaxis=dict(gridcolor=CS["grid"], title="Charm (|val|)"),
        legend=dict(bgcolor="rgba(13,31,60,0.9)", bordercolor=CS["grid"],
                    font=dict(size=10, color="#ffffff")),
        margin=dict(l=50, r=20, t=40, b=40),
        height=420,
    )
    return fig, levels


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=CS["text"], size=16))
    fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                      height=420, margin=dict(l=20, r=20, t=20, b=20))
    return fig
