"""
Model page — logistic regression diagram + feature table.
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

_DATAHACKS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_DATAHACKS))

dash.register_page(__name__, path="/model", name="Model", order=1)

# ── Palette ────────────────────────────────────────────────────────────────────
_BG     = "#faf9f5"
_CARD   = "#ffffff"
_BORDER = "#e8e6dc"
_TEXT   = "#141413"
_MUTED  = "#6b6b63"
_ORANGE = "#d97757"
_GREEN  = "#788c5d"
_RED    = "#c0392b"
_BLUE   = "#6a9bcc"

FEATURES = [
    ("btc_return_30s",          "BTC Momentum",  "Δ BTC / BTC  [−30s]"),
    ("btc_return_60s",          "BTC Momentum",  "Δ BTC / BTC  [−60s]"),
    ("btc_return_300s",         "BTC Momentum",  "Δ BTC / BTC  [−300s]"),
    ("btc_ema_cross_20_120",    "BTC Momentum",  "(EMA₂₀ − EMA₁₂₀) / EMA₁₂₀"),
    ("btc_realized_vol_60s",    "BTC Momentum",  "σ(1s returns, 60s window)"),
    ("btc_trend_consistency",   "BTC Momentum",  "sign-consistency of 1s returns [60s]"),
    ("btc_return_accel",        "BTC Momentum",  "return_30s − return_60s"),
    ("btc_spread_norm",         "BTC Spread",    "spread / mid  [Binance]"),
    ("btc_spread_delta_30s",    "BTC Spread",    "Δspread / mid  [−30s]"),
    ("yes_price",               "Poly Prob",     "YES mid-price"),
    ("yes_price_mom_30s",       "Poly Prob",     "Δ yes_price  [−30s]"),
    ("yes_price_mom_60s",       "Poly Prob",     "Δ yes_price  [−60s]"),
    ("yes_obi_level1",          "OBI",           "(bid_sz − ask_sz) / total  [L1]"),
    ("yes_obi_multilevel",      "OBI",           "exp-decayed OBI  [all levels]"),
    ("yes_microprice_dev",      "OBI",           "(microprice − mid) / mid"),
    ("yes_depth_ratio",         "OBI",           "log(total_bid / total_ask)"),
    ("yes_near_book_imbalance", "OBI",           "OBI  |p − mid| ≤ 0.05"),
    ("yes_spread_norm",         "Poly Spread",   "(ask − bid) / mid  [YES]"),
    ("yes_spread_delta_30s",    "Poly Spread",   "Δ spread / mid  [−30s]"),
    ("time_remaining_frac",     "Timing",        "time_left / duration"),
    ("log_time_remaining",      "Timing",        "ln(seconds remaining)"),
    ("btc_vs_poly_divergence",  "Divergence",    "poly_signal − btc_signal"),
    ("chainlink_vs_binance",    "Divergence",    "(CL − Binance) / Binance"),
]

_GROUP_COLORS = {
    "BTC Momentum": _BLUE,
    "BTC Spread":   "#14b8a6",
    "Poly Prob":    _GREEN,
    "OBI":          _ORANGE,
    "Poly Spread":  "#b58a6e",
    "Timing":       "#9b8ec4",
    "Divergence":   _RED,
}

_WEIGHTS_FILE = _DATAHACKS / "models" / "weights" / "btc_5m_current.npz"
_BAKED_COEF = [
     0.07487681, -0.28664297, -0.16029155,  0.14720130, -0.07334121,
     0.00000000,  0.00000000,
     0.18334326, -0.10993217,  2.14291915, -0.05242122, -0.16313906,
     0.08220650,  0.10749001, -0.14586830,  0.31657409, -0.10730961,
    -1.31421537, -0.00592992,  0.09930648, -0.21187793, -0.56199010,
    -0.12081161,
]


def _load_weights():
    feat_names = [f[0] for f in FEATURES]
    if not _WEIGHTS_FILE.exists():
        return _BAKED_COEF[:len(feat_names)]
    wf = np.load(_WEIGHTS_FILE)
    raw_coef = wf["coef"].tolist()
    if "feature_names" in wf:
        file_names = [str(n).replace("feat_", "") for n in wf["feature_names"]]
        name_to_coef = dict(zip(file_names, raw_coef))
        return [name_to_coef.get(n, 0.0) for n in feat_names]
    return raw_coef[:len(feat_names)]


def _cubic_bezier(p0, p1, p2, p3, n=60):
    """Sample n points along a cubic bezier curve."""
    t = np.linspace(0, 1, n)
    x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
    y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
    return x.tolist(), y.tolist()


def _build_lr_diagram(coef):
    n       = len(coef)
    names   = [f[0] for f in FEATURES[:n]]
    groups  = [f[1] for f in FEATURES[:n]]
    max_abs = max(abs(c) for c in coef) or 1.0

    # Draw heaviest lines last (on top)
    order    = sorted(range(n), key=lambda i: abs(coef[i]))
    s_names  = [names[i]  for i in order]
    s_coef   = [coef[i]   for i in order]
    s_groups = [groups[i] for i in order]

    y_feat             = [i / (n - 1) for i in range(n)]
    sigma_x, sigma_y   = 0.68, 0.50
    feat_x             = 0.06

    # Sample output probability (filler for illustration)
    sample_prob = 0.643

    fig = go.Figure()

    # ── Bezier connection lines ───────────────────────────────────────────────
    for i, (name, w, g) in enumerate(zip(s_names, s_coef, s_groups)):
        lw    = 0.4 + 13.0 * abs(w) / max_abs
        alpha = 0.18 + 0.58 * abs(w) / max_abs
        if w >= 0:
            r, g_c, b = 120, 140, 93
        else:
            r, g_c, b = 192, 57, 43
        color = f"rgba({r},{g_c},{b},{alpha:.2f})"

        yf = y_feat[i]
        # Control points: leave horizontally, arrive horizontally at sigma
        bx, by = _cubic_bezier(
            (feat_x, yf),
            (feat_x + 0.28, yf),       # pull right along feature level
            (sigma_x - 0.22, sigma_y), # approach from the left at sigma height
            (sigma_x, sigma_y),
        )
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color=color, width=lw),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Group: {g}<br>Weight: {w:+.4f}<extra></extra>",
        ))

    # ── Feature nodes (clickable) ─────────────────────────────────────────────
    node_colors = [_GROUP_COLORS.get(g, "#888") for g in s_groups]
    fig.add_trace(go.Scatter(
        x=[feat_x] * n, y=y_feat,
        mode="markers+text",
        marker=dict(size=10, color=node_colors,
                    line=dict(width=1.5, color="white"),
                    symbol="circle"),
        text=s_names,
        textposition="middle left",
        textfont=dict(family="monospace", size=9.5, color=_ORANGE),
        customdata=s_names,   # used by click callback
        showlegend=False,
        hovertemplate="<b>%{text}</b>  — click to open wiki<extra></extra>",
        name="features",
    ))

    # ── Sigma node ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[sigma_x], y=[sigma_y],
        mode="markers+text",
        marker=dict(size=92, color="#1e3a8a",
                    line=dict(width=3, color="#3b5fc0"), symbol="circle"),
        text=["σ"],
        textfont=dict(size=28, color="white", family="Georgia, serif"),
        textposition="middle center",
        showlegend=False, hoverinfo="skip",
    ))

    # ── Arrow sigma → output ──────────────────────────────────────────────────
    # sigma circle radius in data coords ≈ 0.06 at typical viewport width.
    # Tail starts just outside sigma's right edge; head stops before output text.
    prob_color = _GREEN if sample_prob >= 0.5 else _RED
    out_x      = 0.96    # output label centre
    arrow_tail = sigma_x + 0.07   # clear of sigma circle edge
    arrow_head = out_x   - 0.11   # clear of output text

    # Draw arrow as a Scatter line + arrowhead annotation so we control positions precisely
    fig.add_trace(go.Scatter(
        x=[arrow_tail, arrow_head],
        y=[sigma_y, sigma_y],
        mode="lines",
        line=dict(color=_MUTED, width=2),
        showlegend=False, hoverinfo="skip",
    ))
    # Arrowhead: a filled right-pointing triangle marker at the tip
    fig.add_trace(go.Scatter(
        x=[arrow_head], y=[sigma_y],
        mode="markers",
        marker=dict(symbol="triangle-right", size=10, color=_MUTED),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Sample output text (no bounding box) ─────────────────────────────────
    fig.add_annotation(
        x=out_x, y=sigma_y + 0.10,
        text="sample output",
        showarrow=False,
        font=dict(size=10, color=_MUTED, family="Poppins,sans-serif"),
        xanchor="center",
    )
    fig.add_annotation(
        x=out_x, y=sigma_y + 0.01,
        text=f"<b>{sample_prob:.1%}</b>",
        showarrow=False,
        font=dict(size=30, color=prob_color, family="Poppins,sans-serif"),
        xanchor="center",
    )
    fig.add_annotation(
        x=out_x, y=sigma_y - 0.085,
        text="P(BTC ↑)",
        showarrow=False,
        font=dict(size=11, color=_MUTED, family="Poppins,sans-serif"),
        xanchor="center",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    for label, color in [("Positive weight", "rgba(120,140,93,0.85)"),
                          ("Negative weight", "rgba(192,57,43,0.85)")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                  line=dict(color=color, width=4),
                                  name=label, showlegend=True))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=_CARD,
        plot_bgcolor="#faf9f5",
        height=max(640, n * 28),
        xaxis=dict(range=[-0.38, 1.12], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-0.06, 1.06], showgrid=False,
                   showticklabels=False, zeroline=False, fixedrange=True),
        margin=dict(l=215, r=20, t=30, b=30),
        font=dict(color=_TEXT, family="Lora, Georgia, serif"),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=11, color=_TEXT, family="Poppins,sans-serif"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=_BORDER, borderwidth=1,
        ),
    )
    return fig


def _feature_table_rows(coef):
    rows = []
    ranked = sorted(enumerate(coef), key=lambda x: abs(x[1]), reverse=True)
    for rank, (i, w) in enumerate(ranked, 1):
        name, group, expr = FEATURES[i]
        sign_color = _GREEN if w >= 0 else _RED
        gc = _GROUP_COLORS.get(group, "#888")
        rows.append(html.Tr([
            html.Td(html.Span(str(rank), style={"color": _MUTED, "fontSize": "0.78rem",
                                                 "fontFamily": "Poppins,sans-serif"}),
                    className="text-center"),
            html.Td(dcc.Link(
                name, href=f"/feature-wiki?feature={name}",
                style={"color": _ORANGE, "fontFamily": "monospace",
                       "fontSize": "0.84rem", "textDecoration": "none"},
            )),
            html.Td(dbc.Badge(group, style={
                "backgroundColor": f"{gc}18", "color": gc,
                "border": f"1px solid {gc}55", "fontSize": "0.72rem",
                "fontFamily": "Poppins,sans-serif",
            })),
            html.Td(html.Code(expr, style={"fontSize": "0.79rem", "color": _TEXT,
                                            "backgroundColor": "transparent"})),
            html.Td(html.Span(f"{w:+.4f}", style={
                "color": sign_color, "fontFamily": "monospace",
                "fontSize": "0.84rem", "fontWeight": "700",
            }), className="text-end"),
        ], style={"borderBottom": f"1px solid {_BORDER}"}))
    return rows


# ── Build ──────────────────────────────────────────────────────────────────────
coef = _load_weights()
_lr_fig = _build_lr_diagram(coef)

# ── Layout ─────────────────────────────────────────────────────────────────────

def _stat_card(label, value, color=None):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label, style={"color": _MUTED, "fontSize": "0.75rem", "fontWeight": "500",
                              "fontFamily": "Poppins,sans-serif", "letterSpacing": "0.04em",
                              "textTransform": "uppercase", "marginBottom": "4px"}),
        html.H5(value, style={"color": color or _TEXT, "fontFamily": "monospace",
                               "fontWeight": "700", "margin": 0, "fontSize": "0.95rem"}),
    ]), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
               "borderRadius": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}),
    width=3)


layout = dbc.Container(
    fluid=True, className="px-4 py-3",
    style={"backgroundColor": _BG, "minHeight": "100vh"},
    children=[
        # Used by click callback to navigate client-side
        html.Div(id="model-nav-target", style={"display": "none"}),
        dbc.Row([
            dbc.Col([
                html.H4("Logistic Regression Model",
                        style={"color": _TEXT, "fontFamily": "Poppins,sans-serif",
                               "fontWeight": "700", "marginBottom": "4px"}),
                html.P([
                    "Binary classifier — predicts P(BTC UP) for each 5-minute contract. "
                    "Features are ", html.Strong("StandardScaler-normalised"),
                    " before fitting. Hover any line to see exact weight.",
                ], style={"color": _MUTED, "fontSize": "0.88rem",
                           "fontFamily": "Lora,Georgia,serif", "margin": 0}),
            ], width=8),
            dbc.Col([
                dcc.Link(dbc.Button("← Backtest", color="outline-secondary", size="sm",
                                    style={"borderRadius": "8px",
                                           "fontFamily": "Poppins,sans-serif"}), href="/"),
                dcc.Link(dbc.Button("Factor Library →", size="sm", className="ms-2",
                                    style={"borderRadius": "8px", "border": "none",
                                           "backgroundColor": _ORANGE, "color": "white",
                                           "fontFamily": "Poppins,sans-serif"}),
                          href="/factor-library"),
            ], width=4, className="text-end d-flex align-items-center justify-content-end"),
        ], className="mb-4 align-items-center"),

        dbc.Row([
            _stat_card("Bias term",       f"{-0.02062640:+.6f}"),
            _stat_card("Entry threshold", "p − ask > 0.04"),
            _stat_card("Kelly fraction",  "0.5",  _ORANGE),
            _stat_card("Max shares",      "500",  _BLUE),
        ], className="mb-4 g-3"),

        dbc.Card([
            dbc.CardBody([
                html.P([
                    html.Span("Feature → σ(·) → P(BTC ↑)",
                              style={"fontFamily": "Poppins,sans-serif", "fontWeight": "600",
                                     "color": _TEXT, "fontSize": "0.88rem"}),
                    html.Span("  ·  Line thickness ∝ |weight|  ·  ",
                              style={"color": _MUTED, "fontSize": "0.82rem"}),
                    html.Span("Green = positive", style={"color": _GREEN, "fontWeight": "600",
                                                          "fontSize": "0.82rem"}),
                    html.Span("  ·  ", style={"color": _MUTED, "fontSize": "0.82rem"}),
                    html.Span("Red = negative", style={"color": _RED, "fontWeight": "600",
                                                        "fontSize": "0.82rem"}),
                ], style={"marginBottom": "8px"}),
                html.P("Click any feature node to open its wiki page.",
                       style={"color": _MUTED, "fontSize": "0.78rem",
                              "fontFamily": "Poppins,sans-serif", "marginBottom": "4px"}),
                dcc.Graph(id="lr-diagram", figure=_lr_fig,
                          config={"displayModeBar": False}),
            ])
        ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
                   "marginBottom": "24px"}),

        html.Span("Feature Details — ranked by |weight|",
                  style={"fontFamily": "Poppins,sans-serif", "fontWeight": "600",
                         "fontSize": "0.75rem", "letterSpacing": "0.08em",
                         "textTransform": "uppercase", "color": _MUTED,
                         "display": "block", "marginBottom": "12px"}),
        dbc.Card(dbc.CardBody(
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("#",          style={"width": "4%",  "color": _MUTED,
                                                  "fontFamily": "Poppins,sans-serif",
                                                  "fontSize": "0.75rem", "fontWeight": "600",
                                                  "borderBottom": f"2px solid {_BORDER}"}),
                    html.Th("Feature",    style={"width": "22%", "color": _MUTED,
                                                  "fontFamily": "Poppins,sans-serif",
                                                  "fontSize": "0.75rem", "fontWeight": "600",
                                                  "borderBottom": f"2px solid {_BORDER}"}),
                    html.Th("Group",      style={"width": "16%", "color": _MUTED,
                                                  "fontFamily": "Poppins,sans-serif",
                                                  "fontSize": "0.75rem", "fontWeight": "600",
                                                  "borderBottom": f"2px solid {_BORDER}"}),
                    html.Th("Expression", style={"color": _MUTED,
                                                  "fontFamily": "Poppins,sans-serif",
                                                  "fontSize": "0.75rem", "fontWeight": "600",
                                                  "borderBottom": f"2px solid {_BORDER}"}),
                    html.Th("Weight",     className="text-end",
                            style={"width": "10%", "color": _MUTED,
                                   "fontFamily": "Poppins,sans-serif",
                                   "fontSize": "0.75rem", "fontWeight": "600",
                                   "borderBottom": f"2px solid {_BORDER}"}),
                ])),
                html.Tbody(_feature_table_rows(coef)),
            ], borderless=True, size="sm", className="mb-0"),
            style={"padding": "4px"},
        ), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"}),
    ],
)


# ── Click-to-wiki: write URL into hidden div, client-side JS does navigation ──

@callback(
    Output("model-nav-target", "children"),
    Input("lr-diagram", "clickData"),
    prevent_initial_call=True,
)
def navigate_to_wiki(click_data):
    if not click_data:
        return dash.no_update
    pts = click_data.get("points", [])
    if not pts:
        return dash.no_update
    cd = pts[0].get("customdata")
    feat_names = [f[0] for f in FEATURES]
    if cd and cd in feat_names:
        url = f"/feature-wiki?feature={cd}"
        # Inject a tiny script that fires once and navigates
        return html.Script(f"window.location.href='{url}';")
    return dash.no_update
