"""Buy page — sustainability marketplace (demo)."""

import sys
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash import html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_state import state

dash.register_page(__name__, path="/buy", name="Buy", order=4)

_CARD   = "rgba(255,255,255,0.08)"
_BORDER = "rgba(255,255,255,0.14)"
_TEXT   = "#ffffff"
_MUTED  = "rgba(255,255,255,0.5)"
_GREEN  = "#4ade80"

_PRODUCTS = [
    ("Solar Panel Kit",        "$299", "200W home solar kit — DIY install, 25-year lifespan.",       "https://picsum.photos/seed/solar1/300/140"),
    ("Bamboo Water Bottle",    "$28",  "750ml, fully biodegradable. Replaces ~1,000 plastic bottles.","https://picsum.photos/seed/bamboo2/300/140"),
    ("Compostable Phone Case", "$42",  "Plant-based casing. Fully compostable within 12 months.",     "https://picsum.photos/seed/phone3/300/140"),
    ("LED Smart Bulb Pack",    "$65",  "Set of 4 — cuts lighting energy use by up to 85%.",           "https://picsum.photos/seed/bulb4/300/140"),
    ("Organic Cotton Tote",    "$18",  "Fair-trade certified. One tote replaces 500 plastic bags.",   "https://picsum.photos/seed/tote5/300/140"),
    ("Beeswax Wraps 6-pack",   "$22",  "Natural cling-wrap alternative. Reusable for up to 1 year.", "https://picsum.photos/seed/bees6/300/140"),
    ("Recycled Sneakers",      "$120", "Upper crafted from reclaimed ocean plastic bottles.",          "https://picsum.photos/seed/shoe7/300/140"),
]

_FUNDRAISERS = [
    ("Amazon Reforestation",  "$500K goal", "$312K raised", "Plant 1 million trees in the Brazilian Amazon by 2026.",            "https://picsum.photos/seed/amazon8/300/140"),
    ("Ocean Plastic Cleanup", "$200K goal", "$148K raised", "Remove 50 tonnes of plastic from the North Pacific Gyre.",          "https://picsum.photos/seed/ocean9/300/140"),
    ("Solar Schools Africa",  "$75K goal",  "$61K raised",  "Install off-grid solar in 50 rural Kenyan schools.",                "https://picsum.photos/seed/school10/300/140"),
    ("Clean Water Initiative","$150K goal", "$89K raised",  "Provide clean drinking water to 10,000 families in Uganda.",        "https://picsum.photos/seed/water11/300/140"),
    ("Coral Reef Restoration","$300K goal", "$201K raised", "Restore 5 hectares of bleached reef in the Great Barrier Reef.",    "https://picsum.photos/seed/coral12/300/140"),
    ("Community Wind Farm",   "$1M goal",   "$730K raised", "Community-owned wind energy cooperative in rural Wales.",           "https://picsum.photos/seed/wind13/300/140"),
]


def _card(body):
    return html.Div(body, style={
        "minWidth": "240px", "maxWidth": "240px", "flexShrink": "0",
        "background": _CARD, "border": f"1px solid {_BORDER}",
        "borderRadius": "16px", "padding": "16px",
        "display": "flex", "flexDirection": "column",
    })


def _btn(label):
    return html.Button(label, disabled=True, style={
        "background": _GREEN, "border": "none", "borderRadius": "50px",
        "padding": "6px 18px", "fontSize": "0.80rem", "fontWeight": "600",
        "color": "#14532d", "cursor": "not-allowed", "opacity": "0.65",
        "fontFamily": "Inter,sans-serif", "marginTop": "auto",
    })


def _row(cards):
    return html.Div(cards, style={
        "display": "flex", "overflowX": "auto", "gap": "14px",
        "paddingBottom": "8px", "flexWrap": "nowrap", "scrollbarWidth": "thin",
    })


def _label(txt):
    return html.Div(txt, style={"color": _MUTED, "fontSize": "0.72rem", "fontWeight": "600",
                                 "textTransform": "uppercase", "letterSpacing": "0.06em",
                                 "fontFamily": "Inter,sans-serif", "marginBottom": "4px"})


def layout():
    pnl    = state.summary.get("total_pnl", 0)
    budget = max(pnl, 0)

    _img = lambda src: html.Img(src=src, style={
        "width": "100%", "height": "130px", "objectFit": "cover",
        "borderRadius": "10px", "marginBottom": "12px", "display": "block",
    })

    product_cards = [_card([
        _img(img),
        html.Div(n, style={"fontWeight": "700", "fontSize": "0.93rem", "color": _TEXT,
                            "fontFamily": "Inter,sans-serif", "marginBottom": "4px"}),
        html.Div(d, style={"color": _MUTED, "fontSize": "0.81rem", "lineHeight": "1.5",
                            "fontFamily": "Inter,sans-serif", "marginBottom": "10px", "flex": "1"}),
        html.Div(p, style={"color": _GREEN, "fontWeight": "700", "fontFamily": "monospace",
                            "fontSize": "1rem", "marginBottom": "10px"}),
        _btn("Buy"),
    ]) for n, p, d, img in _PRODUCTS]

    fundraiser_cards = [_card([
        _img(img),
        html.Div(n, style={"fontWeight": "700", "fontSize": "0.93rem", "color": _TEXT,
                            "fontFamily": "Inter,sans-serif", "marginBottom": "4px"}),
        html.Div(d, style={"color": _MUTED, "fontSize": "0.81rem", "lineHeight": "1.5",
                            "fontFamily": "Inter,sans-serif", "marginBottom": "8px", "flex": "1"}),
        html.Div([
            html.Span(r, style={"color": _GREEN, "fontWeight": "600",
                                 "fontSize": "0.78rem", "fontFamily": "monospace"}),
            html.Span(f" / {g}", style={"color": _MUTED, "fontSize": "0.76rem",
                                         "fontFamily": "monospace"}),
        ], style={"marginBottom": "10px"}),
        _btn("Donate"),
    ]) for n, g, r, d, img in _FUNDRAISERS]

    return html.Div([
        # ── Stats pane ───────────────────────────────────────────────────────
        dbc.Card(dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    _label("Profit Available"),
                    html.Div(
                        f"${budget:,.2f}" if state.done else "Run a backtest first",
                        style={"color": _GREEN if budget > 0 else _MUTED,
                               "fontWeight": "700", "fontSize": "1.5rem",
                               "fontFamily": "monospace"},
                    ),
                ], width="auto"),
                dbc.Col([
                    _label("Status"),
                    html.Div(
                        "Ready to deploy" if budget > 0 else "No funds yet",
                        style={"color": _MUTED, "fontWeight": "500",
                               "fontSize": "0.88rem", "fontFamily": "Inter,sans-serif"},
                    ),
                ], width="auto"),
                dbc.Col([
                    _label("Buying Power"),
                    html.Div("Locked — demo mode", style={"color": _MUTED, "fontWeight": "500",
                                                           "fontSize": "0.88rem",
                                                           "fontFamily": "Inter,sans-serif"}),
                ], width="auto"),
            ], className="g-4 align-items-center"),
        ), style={"background": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "16px", "marginBottom": "28px"}),

        # ── Products ─────────────────────────────────────────────────────────
        html.Div("🌱  Sustainability Products",
                 style={"color": _TEXT, "fontWeight": "700", "fontFamily": "Raleway,sans-serif",
                         "fontSize": "1rem", "marginBottom": "12px"}),
        _row(product_cards),

        html.Div(style={"height": "28px"}),

        # ── Fundraisers ───────────────────────────────────────────────────────
        html.Div("💚  Sustainability Fundraisers",
                 style={"color": _TEXT, "fontWeight": "700", "fontFamily": "Raleway,sans-serif",
                         "fontSize": "1rem", "marginBottom": "12px"}),
        _row(fundraiser_cards),
    ])
