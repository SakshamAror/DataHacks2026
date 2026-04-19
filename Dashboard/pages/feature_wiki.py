"""
Feature Wiki — per-feature expression + motivation.
URL: /feature-wiki?feature=<name>
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

dash.register_page(__name__, path="/feature-wiki", name="Feature Wiki", order=3)

_CONTENT_DIR = Path(__file__).resolve().parent.parent / "content" / "features"

_BG     = "#faf9f5"
_CARD   = "#ffffff"
_BORDER = "#e8e6dc"
_TEXT   = "#141413"
_MUTED  = "#6b6b63"
_ORANGE = "#d97757"
_GREEN  = "#788c5d"
_RED    = "#c0392b"
_BLUE   = "#6a9bcc"

_GROUP_COLORS = {
    "BTC Momentum": _BLUE,
    "BTC Spread":   "#14b8a6",
    "Poly Prob":    _GREEN,
    "OBI":          _ORANGE,
    "Poly Spread":  "#b58a6e",
    "Timing":       "#9b8ec4",
    "Divergence":   _RED,
    # Legacy keys
    "A · BTC Momentum": _BLUE,
    "B · BTC Spread":   "#14b8a6",
    "C · Poly Prob":    _GREEN,
    "D · OBI":          _ORANGE,
    "E · Poly Spread":  "#b58a6e",
    "F · Timing":       "#9b8ec4",
    "G · Divergence":   _RED,
}

_FEATURE_META = {
    "btc_return_30s":          ("BTC Momentum",  "Δ BTC / BTC  [−30s]"),
    "btc_return_60s":          ("BTC Momentum",  "Δ BTC / BTC  [−60s]"),
    "btc_return_300s":         ("BTC Momentum",  "Δ BTC / BTC  [−300s]"),
    "btc_ema_cross_20_120":    ("BTC Momentum",  "(EMA₂₀ − EMA₁₂₀) / EMA₁₂₀"),
    "btc_realized_vol_60s":    ("BTC Momentum",  "σ(1s returns, 60s window)"),
    "btc_trend_consistency":   ("BTC Momentum",  "mean(sign(Δ BTC), 60s)"),
    "btc_return_accel":        ("BTC Momentum",  "return_30s − return_60s"),
    "btc_spread_norm":         ("BTC Spread",    "spread / mid  [Binance]"),
    "btc_spread_delta_30s":    ("BTC Spread",    "Δspread / mid  [−30s]"),
    "yes_price":               ("Poly Prob",     "YES mid-price"),
    "yes_price_mom_30s":       ("Poly Prob",     "Δ yes_price  [−30s]"),
    "yes_price_mom_60s":       ("Poly Prob",     "Δ yes_price  [−60s]"),
    "yes_obi_level1":          ("OBI",           "(bid_sz − ask_sz) / total  [L1]"),
    "yes_obi_multilevel":      ("OBI",           "exp-decayed OBI  [all levels]"),
    "yes_microprice_dev":      ("OBI",           "(microprice − mid) / mid"),
    "yes_depth_ratio":         ("OBI",           "log(total_bid / total_ask)"),
    "yes_near_book_imbalance": ("OBI",           "OBI  |p − mid| ≤ 0.05"),
    "yes_spread_norm":         ("Poly Spread",   "(ask − bid) / mid  [YES]"),
    "yes_spread_delta_30s":    ("Poly Spread",   "Δ spread / mid  [−30s]"),
    "time_remaining_frac":     ("Timing",        "time_left / duration"),
    "log_time_remaining":      ("Timing",        "ln(seconds remaining)"),
    "btc_vs_poly_divergence":  ("Divergence",    "poly_signal − btc_signal"),
    "chainlink_vs_binance":    ("Divergence",    "(CL − Binance) / Binance"),
}


def _load_wiki(name: str) -> str | None:
    path = _CONTENT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _render_content(feature_name: str):
    if not feature_name or feature_name not in _FEATURE_META:
        return dbc.Card(dbc.CardBody([
            html.H5("Feature not found",
                    style={"color": _TEXT, "fontFamily": "Poppins,sans-serif",
                           "fontWeight": "600"}),
            html.P(["Go back to the ",
                    dcc.Link("Factor Library", href="/factor-library",
                             style={"color": _ORANGE}),
                    " and click a feature card."],
                   style={"color": _MUTED, "fontFamily": "Lora,Georgia,serif"}),
        ]), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px"})

    group, expr = _FEATURE_META[feature_name]
    gc = _GROUP_COLORS.get(group, "#888")
    wiki_md = _load_wiki(feature_name)

    header = dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(group,
                          style={"backgroundColor": f"{gc}18", "color": gc,
                                 "border": f"1px solid {gc}44",
                                 "fontSize": "0.72rem", "fontFamily": "Poppins,sans-serif",
                                 "fontWeight": "500", "padding": "2px 10px",
                                 "borderRadius": "20px", "marginBottom": "10px",
                                 "display": "inline-block"}),
            ]),
            html.H4(feature_name,
                    style={"color": _TEXT, "fontFamily": "monospace",
                           "fontWeight": "700", "marginBottom": "6px",
                           "fontSize": "1.3rem"}),
            html.Code(expr,
                      style={"color": _MUTED, "fontSize": "0.88rem",
                             "backgroundColor": f"{_BORDER}80",
                             "padding": "2px 8px", "borderRadius": "4px"}),
        ], width=8),
        dbc.Col([
            dcc.Link(dbc.Button("← Factor Library", color="outline-secondary", size="sm",
                                style={"borderRadius": "8px",
                                       "fontFamily": "Poppins,sans-serif"}),
                     href="/factor-library"),
            dcc.Link(dbc.Button("Model →", size="sm", className="ms-2",
                                style={"borderRadius": "8px", "border": "none",
                                       "backgroundColor": _ORANGE, "color": "white",
                                       "fontFamily": "Poppins,sans-serif"}),
                     href="/model"),
        ], width=4, className="text-end d-flex align-items-center justify-content-end"),
    ], className="mb-4 align-items-start")

    if wiki_md:
        body = dbc.Card(dbc.CardBody(
            dcc.Markdown(wiki_md, style={"lineHeight": "1.85", "fontSize": "0.93rem",
                                         "color": _TEXT,
                                         "fontFamily": "Lora,Georgia,serif"}),
        ), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"})
    else:
        body = dbc.Card(dbc.CardBody(
            html.P("Wiki content not yet written for this feature.",
                   style={"color": _MUTED, "fontFamily": "Lora,Georgia,serif",
                          "margin": 0}),
        ), style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px"})

    return html.Div([header, body])


layout = dbc.Container(
    fluid=True, className="px-4 py-3",
    style={"backgroundColor": _BG, "minHeight": "100vh"},
    children=[
        dcc.Location(id="wiki-url", refresh=False),
        html.Div(id="wiki-content"),
    ],
)


@callback(
    Output("wiki-content", "children"),
    Input("wiki-url", "search"),
)
def render_wiki(search: str):
    feature = ""
    if search:
        for part in search.lstrip("?").split("&"):
            if part.startswith("feature="):
                feature = part[len("feature="):]
                break
    return _render_content(feature)
