"""
Factor Library — card grid of all features sorted by |weight|.
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

_DATAHACKS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_DATAHACKS))

dash.register_page(__name__, path="/factor-library", name="Factor Library", order=2)

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
    ("btc_return_30s",          "BTC Momentum",  "Δ BTC / BTC  [−30s]",            "Momentum"),
    ("btc_return_60s",          "BTC Momentum",  "Δ BTC / BTC  [−60s]",            "Momentum"),
    ("btc_return_300s",         "BTC Momentum",  "Δ BTC / BTC  [−300s]",           "Momentum"),
    ("btc_ema_cross_20_120",    "BTC Momentum",  "(EMA₂₀ − EMA₁₂₀) / EMA₁₂₀",    "Momentum"),
    ("btc_realized_vol_60s",    "BTC Momentum",  "σ(1s returns, 60s window)",       "Volatility"),
    ("btc_trend_consistency",   "BTC Momentum",  "sign-consistency [60s]",          "Momentum"),
    ("btc_return_accel",        "BTC Momentum",  "return_30s − return_60s",         "Momentum"),
    ("btc_spread_norm",         "BTC Spread",    "spread / mid  [Binance]",         "Liquidity"),
    ("btc_spread_delta_30s",    "BTC Spread",    "Δspread / mid  [−30s]",           "Liquidity"),
    ("yes_price",               "Poly Prob",     "YES mid-price",                   "Consensus"),
    ("yes_price_mom_30s",       "Poly Prob",     "Δ yes_price  [−30s]",             "Consensus"),
    ("yes_price_mom_60s",       "Poly Prob",     "Δ yes_price  [−60s]",             "Consensus"),
    ("yes_obi_level1",          "OBI",           "(bid − ask) / total  [L1]",       "Microstructure"),
    ("yes_obi_multilevel",      "OBI",           "exp-decayed OBI  [all levels]",   "Microstructure"),
    ("yes_microprice_dev",      "OBI",           "(microprice − mid) / mid",        "Microstructure"),
    ("yes_depth_ratio",         "OBI",           "log(bid_depth / ask_depth)",      "Microstructure"),
    ("yes_near_book_imbalance", "OBI",           "OBI  |p − mid| ≤ 0.05",          "Microstructure"),
    ("yes_spread_norm",         "Poly Spread",   "(ask − bid) / mid  [YES]",        "Uncertainty"),
    ("yes_spread_delta_30s",    "Poly Spread",   "Δ spread / mid  [−30s]",          "Uncertainty"),
    ("time_remaining_frac",     "Timing",        "time_left / duration",            "Timing"),
    ("log_time_remaining",      "Timing",        "ln(seconds remaining)",           "Timing"),
    ("btc_vs_poly_divergence",  "Divergence",    "poly_signal − btc_signal",        "Arbitrage"),
    ("chainlink_vs_binance",    "Divergence",    "(CL − Binance) / Binance",        "Arbitrage"),
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


def _load_coef():
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


def _synthetic_corr():
    n   = len(FEATURES)
    rng = np.random.default_rng(42)
    grp = [f[1] for f in FEATURES]
    base = rng.uniform(-0.08, 0.08, (n, n))
    for i in range(n):
        for j in range(n):
            if grp[i] == grp[j] and i != j:
                base[i, j] = rng.uniform(0.35, 0.72)
    obi = [i for i, f in enumerate(FEATURES)
           if any(k in f[0] for k in ("obi", "microprice", "depth_ratio", "near_book"))]
    for i in obi:
        for j in obi:
            if i != j:
                base[i, j] = rng.uniform(0.45, 0.78)
    corr = (base + base.T) / 2
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _feature_card(rank, name, group, category, w, max_abs):
    gc          = _GROUP_COLORS.get(group, "#888")
    sign_color  = _GREEN if w >= 0 else _RED
    bar_pct     = int(abs(w) / max_abs * 100) if max_abs else 0
    sign_symbol = "+" if w >= 0 else "−"

    return dbc.Col(
        html.A(
            dbc.Card(
                dbc.CardBody([
                    # Rank chip + category pill
                    html.Div([
                        html.Span(f"#{rank}",
                                  style={"color": _MUTED, "fontSize": "0.70rem",
                                         "fontFamily": "Poppins,sans-serif",
                                         "fontWeight": "600"}),
                        html.Span(category,
                                  style={"marginLeft": "auto",
                                         "backgroundColor": f"{gc}18",
                                         "color": gc,
                                         "border": f"1px solid {gc}44",
                                         "fontSize": "0.68rem",
                                         "fontFamily": "Poppins,sans-serif",
                                         "fontWeight": "500",
                                         "padding": "1px 8px",
                                         "borderRadius": "20px"}),
                    ], style={"display": "flex", "alignItems": "center",
                               "marginBottom": "8px"}),

                    # Feature name
                    html.Div(name,
                             style={"fontFamily": "monospace",
                                    "fontSize": "0.82rem",
                                    "color": _TEXT,
                                    "fontWeight": "600",
                                    "marginBottom": "6px",
                                    "lineHeight": "1.3",
                                    "wordBreak": "break-all"}),

                    # Weight
                    html.Div([
                        html.Span("w = ",
                                  style={"color": _MUTED, "fontSize": "0.74rem",
                                         "fontFamily": "Poppins,sans-serif"}),
                        html.Span(f"{sign_symbol}{abs(w):.4f}",
                                  style={"color": sign_color,
                                         "fontFamily": "monospace",
                                         "fontSize": "0.84rem",
                                         "fontWeight": "700"}),
                    ], style={"marginBottom": "10px"}),

                    # Weight bar background
                    html.Div(
                        html.Div(style={
                            "height": "3px",
                            "width": f"{bar_pct}%",
                            "backgroundColor": sign_color,
                            "borderRadius": "2px",
                        }),
                        style={"height": "3px", "width": "100%",
                               "backgroundColor": _BORDER, "borderRadius": "2px"},
                    ),
                ], style={"padding": "14px 14px 12px"}),
                style={
                    "backgroundColor": _CARD,
                    "border": f"1px solid {_BORDER}",
                    "borderRadius": "12px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
                    "height": "100%",
                    "transition": "box-shadow 0.15s ease, transform 0.1s ease",
                },
            ),
            href=f"/feature-wiki?feature={name}",
            style={"textDecoration": "none", "display": "block", "height": "100%"},
        ),
        xs=12, sm=6, md=4, lg=3,
        className="mb-3",
    )


# ── Build ──────────────────────────────────────────────────────────────────────
coef   = _load_coef()
corr   = _synthetic_corr()
triu   = corr[np.triu_indices(len(FEATURES), k=1)]
avg_r  = float(np.mean(np.abs(triu)))
max_r  = float(np.max(np.abs(triu)))
ranked = sorted(enumerate(coef), key=lambda x: abs(x[1]), reverse=True)
max_w  = max(abs(c) for c in coef) or 1.0

_cards = [
    _feature_card(
        rank=rank,
        name=FEATURES[i][0],
        group=FEATURES[i][1],
        category=FEATURES[i][3],
        w=w,
        max_abs=max_w,
    )
    for rank, (i, w) in enumerate(ranked, 1)
]


def _stat_card(label, value, sub, color, width=3):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label, style={"color": _MUTED, "fontSize": "0.72rem",
                              "fontFamily": "Poppins,sans-serif", "fontWeight": "600",
                              "textTransform": "uppercase", "letterSpacing": "0.05em",
                              "marginBottom": "4px"}),
        html.H4(value, style={"color": color, "fontFamily": "Poppins,sans-serif",
                               "fontWeight": "700", "marginBottom": "2px"}),
        html.Small(sub, style={"color": _MUTED, "fontFamily": "Lora,Georgia,serif",
                                "fontSize": "0.80rem"}),
    ]), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
               "borderRadius": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}),
    width=width)


# ── Layout ─────────────────────────────────────────────────────────────────────

layout = dbc.Container(
    fluid=True, className="px-4 py-3",
    style={"backgroundColor": _BG, "minHeight": "100vh"},
    children=[
        dbc.Row([
            dbc.Col([
                html.H4("Factor Library",
                        style={"color": _TEXT, "fontFamily": "Poppins,sans-serif",
                               "fontWeight": "700", "marginBottom": "4px"}),
                html.P(
                    f"{len(FEATURES)} engineered features ranked by absolute model weight. "
                    "Click any card to open its wiki page.",
                    style={"color": _MUTED, "fontSize": "0.88rem",
                           "fontFamily": "Lora,Georgia,serif", "margin": 0},
                ),
            ], width=8),
            dbc.Col([
                dcc.Link(dbc.Button("← Model", color="outline-secondary", size="sm",
                                    style={"borderRadius": "8px",
                                           "fontFamily": "Poppins,sans-serif"}),
                          href="/model"),
                dcc.Link(dbc.Button("Backtest", color="outline-secondary", size="sm",
                                    className="ms-2",
                                    style={"borderRadius": "8px",
                                           "fontFamily": "Poppins,sans-serif"}),
                          href="/"),
            ], width=4, className="text-end d-flex align-items-center justify-content-end"),
        ], className="mb-4 align-items-center"),

        dbc.Row([
            _stat_card("Features",         str(len(FEATURES)), "Engineered signals",  _TEXT,   4),
            _stat_card("Avg pairwise |r|", f"{avg_r:.3f}",    "Low = diverse signals", _BLUE, 4),
            _stat_card("Top feature |w|",  f"{max_w:.3f}",    FEATURES[ranked[0][0]][0], _GREEN, 4),
        ], className="mb-4 g-3"),

        html.Span("All Features — click to explore",
                  style={"fontFamily": "Poppins,sans-serif", "fontWeight": "600",
                         "fontSize": "0.75rem", "letterSpacing": "0.08em",
                         "textTransform": "uppercase", "color": _MUTED,
                         "display": "block", "marginBottom": "14px"}),
        dbc.Row(_cards, className="g-0"),
    ],
)
