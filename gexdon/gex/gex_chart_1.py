"""
Chart 1 — Gamma Exposure Density by Strike (Plotly, Dual Y-Axis)
Multi-DTE aware: each option row has its own t_expiry.
Uses Barchart pre-computed gamma when available, BS fallback.

Implements:
  - Dual y-axis: Left = Sell/Buy/Total Gamma (positive). Right = Net GEX (signed + shading)
  - K* Optimization via Put-Call Parity + Gamma Balance
  - Gamma Flip zero-crossing detection with interpolation
  - Buy/Sell Gamma peak markers (top 3)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from gex.gex_utils import bs_greeks, resolve_iv
from config import RISK_FREE_RATE as R, CS


# ─────────────────────────────────────
# K* Optimization (Put-Call Parity + Gamma Balance)
# ─────────────────────────────────────
def _compute_forward_price(call_price, put_price, strike):
    """F = C + K - P  (Put-Call Parity)."""
    return call_price + strike - put_price


def _find_optimal_strike(gex_df, calls_df, puts_df):
    """
    For each strike:
      1. Read call/put lastPrice → forward price F via Put-Call Parity
      2. f(x) = |put_gex - call_gex|  (gamma balance objective)
      3. Bounds: Fmin = |put_gex - call_gex|, Fmax = put_gex + call_gex
      4. Contradiction check: F must lie within [Fmin, Fmax]
      5. K* = strike minimising f(x) with no contradiction

    Returns K_star, forward_price_at_K*, results DataFrame
    """
    results = []
    for _, row in gex_df.iterrows():
        strike = row["strike"]
        call_gex = abs(row["call_gex"])
        put_gex = abs(row["put_gex"])

        # Get lastPrice from chain
        c_row = calls_df[calls_df["strikePrice"] == strike]
        p_row = puts_df[puts_df["strikePrice"] == strike]

        call_price = float(c_row.iloc[0]["lastPrice"]) if not c_row.empty else 0.0
        put_price = float(p_row.iloc[0]["lastPrice"]) if not p_row.empty else 0.0

        # Guard NaN
        if np.isnan(call_price):
            call_price = 0.0
        if np.isnan(put_price):
            put_price = 0.0

        forward_price = _compute_forward_price(call_price, put_price, strike)

        # Objective: f(x) = |put_gex - call_gex|
        fx = abs(put_gex - call_gex)

        # Bounds
        f_min = abs(put_gex - call_gex)
        f_max = put_gex + call_gex

        # Contradiction check
        has_prices = (call_price != 0 or put_price != 0)
        contradiction = has_prices and not (f_min <= forward_price <= f_max)

        results.append({
            "strike": strike,
            "call_gex": call_gex, "put_gex": put_gex,
            "call_price": call_price, "put_price": put_price,
            "forward_price": forward_price,
            "fx": fx, "f_min": f_min, "f_max": f_max,
            "contradiction": contradiction,
        })

    results_df = pd.DataFrame(results)

    # Select K*: minimize f(x) among non-contradicting strikes
    valid = results_df[~results_df["contradiction"]]
    if valid.empty:
        valid = results_df  # fallback

    best_idx = valid["fx"].idxmin()
    K_star = float(valid.loc[best_idx, "strike"])
    F_at_Kstar = float(valid.loc[best_idx, "forward_price"])

    return K_star, F_at_Kstar, results_df


# ─────────────────────────────────────
# Main Chart Generator
# ─────────────────────────────────────
def generate_gex_chart(data, percent_range=0.03):
    spot = data["spot"]
    calls = data["calls"]
    puts = data["puts"]
    fb_iv = data["fallback_iv"]
    exp_lbl = data["expiry_label"]
    source = data.get("source", "barchart")

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

        # Calls at this strike (sum across expiries)
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

        # Puts at this strike
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

        rows.append({
            "strike": K, "call_gex": cg, "put_gex": pg,
            "net_gex": cg - pg, "sell_gamma": cg, "buy_gamma": pg,
        })

    df = pd.DataFrame(rows)
    rng = spot * percent_range
    df = df[(df["strike"] >= spot - rng) & (df["strike"] <= spot + rng)]
    df = df[(df["call_gex"] != 0) | (df["put_gex"] != 0)]
    if len(df) < 2:
        return _empty("No Gamma data"), {}

    # ── K* Optimization ──
    K_star, F_at_Kstar, opt_results = _find_optimal_strike(df, calls, puts)

    # Get K* row details for annotation
    kstar_row = opt_results[opt_results["strike"] == K_star]
    if not kstar_row.empty:
        fx_k = float(kstar_row.iloc[0]["fx"])
        fmin_k = float(kstar_row.iloc[0]["f_min"])
        fmax_k = float(kstar_row.iloc[0]["f_max"])
        contr_k = bool(kstar_row.iloc[0]["contradiction"])
    else:
        fx_k = fmin_k = fmax_k = 0.0
        contr_k = False

    # ── Prepare plot data ──
    strikes = df["strike"].values
    sell_gamma = np.abs(df["sell_gamma"].values)
    buy_gamma = np.abs(df["buy_gamma"].values)
    net_gex_raw = df["net_gex"].values  # SIGNED — preserves flip points
    total_gamma = sell_gamma + buy_gamma

    # Smoothing
    if len(strikes) > 3:
        sigma = 1.5
        sell_smooth = gaussian_filter1d(sell_gamma, sigma)
        buy_smooth = gaussian_filter1d(buy_gamma, sigma)
        net_smooth = gaussian_filter1d(net_gex_raw, sigma)
        total_smooth = gaussian_filter1d(total_gamma, sigma)
    else:
        sell_smooth = sell_gamma
        buy_smooth = buy_gamma
        net_smooth = net_gex_raw
        total_smooth = total_gamma

    # ── Gamma Flip Detection (zero-crossings on signed net GEX) ──
    gamma_flips = []
    for i in range(len(net_smooth) - 1):
        if net_smooth[i] * net_smooth[i + 1] < 0:
            x1, x2 = strikes[i], strikes[i + 1]
            y1, y2 = net_smooth[i], net_smooth[i + 1]
            if y2 - y1 != 0:
                flip = x1 - y1 * (x2 - x1) / (y2 - y1)
                gamma_flips.append(flip)

    # ── Peak Detection (top 3) ──
    def find_peaks(data_arr, n=3):
        peaks = []
        for i in range(1, len(data_arr) - 1):
            if data_arr[i] > data_arr[i - 1] and data_arr[i] > data_arr[i + 1]:
                peaks.append((strikes[i], data_arr[i]))
        return sorted(peaks, key=lambda x: x[1], reverse=True)[:n]

    sell_peaks = find_peaks(sell_smooth)
    buy_peaks = find_peaks(buy_smooth)

    # ── Build Dual Y-Axis Chart ──
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # LEFT AXIS: Sell Gamma, Buy Gamma, Total Gamma (always positive)
    fig.add_trace(go.Scatter(
        x=strikes, y=sell_smooth, name="Sell Gamma",
        line=dict(color=CS["red"], width=3), opacity=0.9,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=strikes, y=buy_smooth, name="Buy Gamma",
        line=dict(color=CS["green"], width=3), opacity=0.9,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=strikes, y=total_smooth, name="Total Gamma",
        line=dict(color=CS["orange"], width=2.5), opacity=0.85,
    ), secondary_y=False)

    # RIGHT AXIS: Net GEX signed + shading
    fig.add_trace(go.Scatter(
        x=strikes, y=net_smooth, name="Net GEX",
        line=dict(color=CS["gold"], width=3.5), opacity=0.95,
    ), secondary_y=True)

    # Positive GEX fill (green)
    net_pos = np.where(net_smooth > 0, net_smooth, 0)
    fig.add_trace(go.Scatter(
        x=strikes, y=net_pos,
        fill="tozeroy", fillcolor="rgba(0,200,0,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), secondary_y=True)

    # Negative GEX fill (red)
    net_neg = np.where(net_smooth < 0, net_smooth, 0)
    fig.add_trace(go.Scatter(
        x=strikes, y=net_neg,
        fill="tozeroy", fillcolor="rgba(255,0,0,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), secondary_y=True)

    # Zero line on right axis
    fig.add_trace(go.Scatter(
        x=[strikes[0], strikes[-1]], y=[0, 0],
        line=dict(color=CS["gold"], width=0.8, dash="solid"),
        opacity=0.3, showlegend=False, hoverinfo="skip",
    ), secondary_y=True)

    # ── Vertical reference lines ──
    # Spot
    fig.add_vline(x=spot, line=dict(color=CS["cyan"], width=2.5, dash="dash"),
                  annotation_text=f"Spot ${spot:.2f}",
                  annotation_font_color=CS["cyan"],
                  annotation_font_size=9)

    # K* (optimal strike)
    fig.add_vline(x=K_star, line=dict(color="magenta", width=3, dash="dot"),
                  annotation_text=f"K* ${K_star:.0f}",
                  annotation_font_color="magenta",
                  annotation_font_size=9,
                  annotation_position="bottom right")

    # Forward Price at K*
    fig.add_vline(x=F_at_Kstar, line=dict(color="#ffffff", width=2, dash="dashdot"),
                  annotation_text=f"Fwd ${F_at_Kstar:.2f}",
                  annotation_font_color="#ffffff",
                  annotation_font_size=9,
                  annotation_position="top left")

    # Gamma Flip vertical lines
    for i, flip in enumerate(gamma_flips):
        fig.add_vline(x=flip, line=dict(color="darkgoldenrod", width=2, dash="dash"),
                      annotation_text=f"Flip ${flip:.0f}" if i == 0 else f"${flip:.0f}",
                      annotation_font_color="darkgoldenrod",
                      annotation_font_size=8,
                      annotation_position="top right")

    # ── Peak Markers ──
    if sell_peaks:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in sell_peaks],
            y=[p[1] for p in sell_peaks],
            mode="markers+text",
            marker=dict(color=CS["red"], size=10, symbol="triangle-down"),
            text=[f"${p[0]:.0f}" for p in sell_peaks],
            textposition="top center",
            textfont=dict(color=CS["red"], size=8),
            name="Sell Peak", showlegend=False,
        ), secondary_y=False)

    if buy_peaks:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in buy_peaks],
            y=[p[1] for p in buy_peaks],
            mode="markers+text",
            marker=dict(color=CS["green"], size=10, symbol="triangle-up"),
            text=[f"${p[0]:.0f}" for p in buy_peaks],
            textposition="top center",
            textfont=dict(color=CS["green"], size=8),
            name="Buy Peak", showlegend=False,
        ), secondary_y=False)

    # ── Annotation Box (K* details) ──
    ann_text = (
        f"Spot: ${spot:.2f}<br>"
        f"K* (Optimal): ${K_star:.2f}<br>"
        f"Forward F: ${F_at_Kstar:.2f}<br>"
        f"f(x) at K*: {fx_k:.4f}<br>"
        f"Bounds: [{fmin_k:.2f}, {fmax_k:.2f}]<br>"
        f"{'⚠ Contradiction' if contr_k else '✓ No contradiction'}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=ann_text, showarrow=False,
        font=dict(size=8, color=CS["text"], family="monospace"),
        bgcolor="rgba(30,50,80,0.85)", bordercolor=CS["grid"],
        borderwidth=1, borderpad=4,
        align="left", valign="top",
    )

    # ── Layout ──
    iv_tag = "Barchart IV" if fb_iv is None else f"Est IV {fb_iv:.3f}"
    title = f"Gamma Density ({exp_lbl}) — {iv_tag} [{source.upper()}]"

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(color=CS["text"], size=13)),
        paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
        font=dict(color=CS["text"], size=10),
        xaxis=dict(gridcolor=CS["grid"], title="Strike", tickformat="$,.0f"),
        legend=dict(bgcolor="rgba(13,31,60,0.9)", bordercolor=CS["grid"],
                    font=dict(size=10, color="#ffffff"),
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        margin=dict(l=55, r=55, t=55, b=40),
        height=350,
    )

    # Left y-axis: always positive gamma
    fig.update_yaxes(
        title=dict(text="Gamma (Sell / Buy / Total)", font=dict(color="#ffffff")),
        tickfont=dict(color="#ffffff"),
        gridcolor=CS["grid"], rangemode="tozero",
        secondary_y=False,
    )

    # Right y-axis: signed net GEX
    net_max = float(np.max(np.abs(net_smooth))) if len(net_smooth) > 0 else 1
    fig.update_yaxes(
        title=dict(text="Net GEX (signed)", font=dict(color=CS["gold"])),
        tickfont=dict(color=CS["gold"]),
        gridcolor=CS["grid"],
        range=[-net_max * 1.4, net_max * 1.4],
        secondary_y=True,
    )

    # ── Extract levels ──
    levels = _extract_levels(df, spot, gamma_flips, K_star, F_at_Kstar,
                             sell_peaks, buy_peaks)

    return fig, levels


# ─────────────────────────────────────
# Level Extraction
# ─────────────────────────────────────
def _extract_levels(df, spot, gamma_flips, K_star, F_at_Kstar,
                    sell_peaks, buy_peaks):
    lv = {}

    # K* and Forward Price
    lv["k_star"] = K_star
    lv["forward_price"] = F_at_Kstar

    # Max absolute gamma strike
    lv["max_gamma"] = float(df.loc[df["net_gex"].abs().idxmax(), "strike"])

    # Gamma Flip (closest to spot)
    if gamma_flips:
        lv["zero_gamma"] = float(min(gamma_flips, key=lambda x: abs(x - spot)))
        lv["all_flips"] = [float(f) for f in gamma_flips]

    # Call Wall / Put Wall (max sell/buy gamma)
    lv["call_wall"] = float(df.loc[df["call_gex"].idxmax(), "strike"])
    lv["put_wall"] = float(df.loc[df["put_gex"].idxmax(), "strike"])

    # Max/Min Net GEX strikes
    lv["max_net_gex_strike"] = float(df.loc[df["net_gex"].idxmax(), "strike"])
    lv["min_net_gex_strike"] = float(df.loc[df["net_gex"].idxmin(), "strike"])

    # Sell Gamma peaks
    if sell_peaks:
        lv["sell_peak_1"] = sell_peaks[0][0]
        if len(sell_peaks) > 1:
            lv["sell_peak_2"] = sell_peaks[1][0]

    # Buy Gamma peaks
    if buy_peaks:
        lv["buy_peak_1"] = buy_peaks[0][0]
        if len(buy_peaks) > 1:
            lv["buy_peak_2"] = buy_peaks[1][0]

    return lv


# ─────────────────────────────────────
# Empty Chart
# ─────────────────────────────────────
def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color=CS["text"], size=16))
    fig.update_layout(paper_bgcolor=CS["bg"], plot_bgcolor=CS["plot_bg"],
                      height=350, margin=dict(l=20, r=20, t=20, b=20))
    return fig
