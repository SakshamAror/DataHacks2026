"""
Home page — Run Backtest + live market view (left) + PNL chart (right).
"""

import sys
from pathlib import Path

import pandas as _pd  # noqa: F401 — force early init, prevents plotly circular import

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_state import state
from backtest_adapter import run_backtest_async

_DATAHACKS = Path(__file__).resolve().parent.parent.parent
_DATA_DIR   = str(_DATAHACKS / "data" / "validation")
if not Path(_DATA_DIR).exists():
    _DATA_DIR = str(_DATAHACKS / "datasets" / "validation")

dash.register_page(__name__, path="/", name="Backtest", order=0)

# ── Glass palette ─────────────────────────────────────────────────────────────
_BG     = "transparent"
_CARD   = "rgba(255,255,255,0.08)"
_BORDER = "rgba(255,255,255,0.14)"
_TEXT   = "#ffffff"
_MUTED  = "rgba(255,255,255,0.5)"
_ORANGE = "#4ade80"
_GREEN  = "#4ade80"
_RED    = "#f87171"
_BLUE   = "#60a5fa"


# ── Slug parser ────────────────────────────────────────────────────────────────

def _parse_slug(slug: str) -> tuple[str, str]:
    from datetime import datetime, timezone
    parts    = slug.lower().split("-")
    asset    = parts[0].upper() if parts else "?"
    interval = next((p for p in parts if p.endswith("m") or p.endswith("h")), "?")
    ts_str   = next((p for p in reversed(parts) if p.isdigit()), None)
    if ts_str:
        try:
            dt      = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
            exp_str = dt.strftime("%H:%M UTC")
        except (ValueError, OSError):
            exp_str = "?"
    else:
        exp_str = "?"
    direction = "Up / Down" if "updown" in slug else "Prediction"
    return f"{asset}  {direction}  ·  {interval}", f"Expires {exp_str}"


# ── Position helpers ───────────────────────────────────────────────────────────

def _net_position(fills):
    yes, no = 0.0, 0.0
    for f in fills:
        sign = 1 if f["side"] == "BUY" else -1
        if f["token"] == "YES":
            yes += sign * f["size"]
        else:
            no  += sign * f["size"]
    return yes, no


def _avg_entry(fills, token, side="BUY"):
    """Weighted average entry price for a given token/side."""
    total_size, total_cost = 0.0, 0.0
    for f in fills:
        if f["token"] == token and f["side"] == side:
            total_size += f["size"]
            total_cost += f["size"] * f["avg_price"]
    return total_cost / total_size if total_size > 0 else None


# ── UI components ──────────────────────────────────────────────────────────────

def _fill_blocks(fills):
    if not fills:
        return html.Div()
    blocks = []
    for f in fills:
        is_buy = f["side"] == "BUY"
        bg     = _GREEN if is_buy else _RED
        arrow  = "▲" if is_buy else "▼"
        blocks.append(html.Span(
            f"{arrow} {f['side']} {f.get('token','?')}  "
            f"{int(f.get('size',0))} @ {f.get('avg_price',0):.3f}",
            style={"backgroundColor": bg, "color": "#fff",
                   "fontSize": "0.70rem", "fontFamily": "monospace",
                   "fontWeight": "600", "padding": "2px 9px",
                   "borderRadius": "4px", "display": "inline-block"},
        ))
    return html.Div(blocks, style={"display": "flex", "flexWrap": "wrap",
                                    "gap": "5px", "marginTop": "7px"})


def _settlement_card(slug, outcome, btc_open, btc_close, fills):
    """Card shown for a just-settled market."""
    title, subtitle = _parse_slug(slug)
    yes_pos, no_pos = _net_position(fills)

    # Did we win?
    our_token  = "YES" if abs(yes_pos) >= abs(no_pos) else "NO"
    our_size   = abs(yes_pos) if our_token == "YES" else abs(no_pos)
    won        = (our_token == outcome) and our_size > 0.01
    had_pos    = our_size > 0.01

    # Estimated P&L from this market
    pnl_est = None
    if had_pos:
        avg_buy = _avg_entry(fills, our_token, "BUY")
        avg_sell = _avg_entry(fills, our_token, "SELL")
        if avg_buy:
            settle_val = 1.0 if won else 0.0
            # open P&L on unsettled shares
            open_shares = our_size
            if avg_sell:
                closed_shares = sum(f["size"] for f in fills
                                    if f["token"] == our_token and f["side"] == "SELL")
                open_shares = our_size  # net already accounts for sells
            pnl_est = open_shares * (settle_val - avg_buy)

    btc_dir    = "↑" if btc_close > btc_open else "↓"
    outcome_color = _GREEN if outcome == "YES" else _RED
    result_color  = _GREEN if won else (_RED if had_pos else _MUTED)
    result_label  = ("✓ WON" if won else ("✗ LOST" if had_pos else "No position"))

    return dbc.Card(
        dbc.CardBody([
            # Settlement banner
            html.Div([
                html.Span("SETTLED",
                          style={"backgroundColor": "rgba(255,255,255,0.12)",
                                 "color": _MUTED, "fontSize": "0.65rem",
                                 "fontFamily": "Inter,sans-serif",
                                 "fontWeight": "700", "padding": "1px 8px",
                                 "borderRadius": "20px", "letterSpacing": "0.06em",
                                 "marginRight": "8px"}),
                html.Span(title, style={"fontFamily": "Inter,sans-serif",
                                         "fontWeight": "600", "fontSize": "0.84rem",
                                         "color": _TEXT}),
            ], style={"marginBottom": "8px"}),

            dbc.Row([
                # Outcome column
                dbc.Col([
                    html.Div("Outcome", style={"color": _MUTED, "fontSize": "0.70rem",
                                               "fontFamily": "Inter,sans-serif",
                                               "textTransform": "uppercase",
                                               "letterSpacing": "0.05em",
                                               "marginBottom": "2px"}),
                    html.Div([
                        html.Span(f"{'BTC rose' if outcome=='YES' else 'BTC fell'}  "
                                  f"({btc_dir} {btc_open:,.0f}→{btc_close:,.0f})",
                                  style={"color": outcome_color, "fontFamily": "monospace",
                                         "fontSize": "0.78rem", "fontWeight": "600"}),
                    ]),
                ], width=7),
                # Result column
                dbc.Col([
                    html.Div("Our result", style={"color": _MUTED, "fontSize": "0.70rem",
                                                   "fontFamily": "Inter,sans-serif",
                                                   "textTransform": "uppercase",
                                                   "letterSpacing": "0.05em",
                                                   "marginBottom": "2px"}),
                    html.Div([
                        html.Span(result_label,
                                  style={"color": result_color,
                                         "fontFamily": "Inter,sans-serif",
                                         "fontSize": "0.82rem", "fontWeight": "700"}),
                        html.Span(
                            f"  {our_token} {our_size:.0f} shares" if had_pos else "",
                            style={"color": _MUTED, "fontFamily": "monospace",
                                   "fontSize": "0.74rem"}
                        ),
                    ]),
                    html.Div(
                        f"Est. P&L: {pnl_est:+.2f}" if pnl_est is not None else "",
                        style={"color": result_color, "fontFamily": "monospace",
                               "fontSize": "0.74rem", "marginTop": "2px"},
                    ),
                ], width=5),
            ], className="g-0"),

            _fill_blocks(fills),
        ], style={"padding": "10px 14px 10px"}),
        style={"background": _CARD,
               "border": f"2px solid {_BORDER}",
               "borderLeft": f"4px solid {result_color}",
               "borderRadius": "10px",
               "marginBottom": "8px",
               "boxShadow": "none"},
    )


def _market_card(slug, ts_list, yes_list, p_list, fills):
    title, subtitle = _parse_slug(slug)
    yes_pos, no_pos = _net_position(fills)
    has_position     = abs(yes_pos) > 0.01 or abs(no_pos) > 0.01

    if has_position:
        parts = []
        if abs(yes_pos) > 0.01: parts.append(f"YES {yes_pos:+.0f}")
        if abs(no_pos) > 0.01:  parts.append(f"NO {no_pos:+.0f}")
        pos_badge = dbc.Badge("  |  ".join(parts),
                               style={"backgroundColor": f"{_ORANGE}22", "color": _ORANGE,
                                      "border": f"1px solid {_ORANGE}55",
                                      "fontSize": "0.70rem", "fontFamily": "Inter,sans-serif"})
    else:
        pos_badge = dbc.Badge("No open position",
                               style={"backgroundColor": f"{_MUTED}15", "color": _MUTED,
                                      "border": f"1px solid {_BORDER}",
                                      "fontSize": "0.70rem"})

    fig = go.Figure()
    if yes_list:
        xs = list(range(len(yes_list)))
        fig.add_trace(go.Scatter(x=xs, y=yes_list, mode="lines", name="Market",
                                  line=dict(color=_BLUE, width=1.6)))
        fig.add_trace(go.Scatter(x=xs, y=p_list, mode="lines", name="Model",
                                  line=dict(color=_ORANGE, width=1.6, dash="dot")))
        for f in fills:
            idx  = f.get("chart_idx", len(xs) - 1)
            yv   = f.get("avg_price", 0.5)
            side = f.get("side", "BUY")
            fig.add_trace(go.Scatter(
                x=[idx], y=[yv], mode="markers",
                marker=dict(symbol="triangle-up" if side == "BUY" else "triangle-down",
                            size=11,
                            color=_GREEN if side == "BUY" else _RED,
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertext=f"{side} {f.get('token','?')} "
                          f"{int(f.get('size',0))} @ {f.get('avg_price',0):.3f}",
                hoverinfo="text",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=120,
        margin=dict(l=32, r=8, t=6, b=18),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", range=[0, 1],
                   tickformat=".0%", tickfont=dict(size=9, color=_MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9, color=_MUTED)),
        hovermode="x unified",
    )

    last_yes = yes_list[-1] if yes_list else None
    last_p   = p_list[-1]   if p_list   else None
    edge_el  = html.Div()
    if last_yes is not None and last_p is not None:
        edge = last_p - last_yes
        ec   = _GREEN if edge > 0.04 else (_ORANGE if edge > 0 else _RED)
        edge_el = html.Div(f"Edge {edge:+.3f}",
                           style={"color": ec, "fontFamily": "monospace",
                                  "fontSize": "0.75rem", "fontWeight": "600",
                                  "textAlign": "right"})

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(title, style={"fontFamily": "Inter,sans-serif",
                                           "fontWeight": "600", "fontSize": "0.85rem",
                                           "color": _TEXT, "letterSpacing": "-0.01em"}),
                    html.Div(subtitle, style={"color": _MUTED, "fontSize": "0.73rem",
                                              "fontFamily": "monospace"}),
                ], width=7),
                dbc.Col([
                    html.Div(pos_badge, className="text-end"),
                    edge_el,
                ], width=5),
            ], className="mb-1 align-items-start"),
            dcc.Graph(figure=fig, config={"displayModeBar": False},
                      style={"height": "120px"}),
            _fill_blocks(fills),
        ], style={"padding": "12px 14px 10px"}),
        style={"background": _CARD, "border": f"1px solid {_BORDER}",
               "borderRadius": "12px", "marginBottom": "10px",
               "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
    )


def _safe_figure(build_fn):
    try:
        return build_fn()
    except Exception:
        return go.Figure()


def _empty_pnl_figure():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, title="", color=_MUTED,
                   linecolor=_BORDER),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
                   title="Portfolio Value ($)", color=_MUTED,
                   linecolor="rgba(255,255,255,0.08)"),
        margin=dict(l=70, r=20, t=10, b=40),
        font=dict(color=_TEXT, family="Inter, system-ui, sans-serif"),
    )
    return fig


def _pnl_figure(all_pnl):
    from datetime import datetime, timezone
    fig = _empty_pnl_figure()
    if not all_pnl:
        return fig
    xs  = [datetime.fromtimestamp(r["ts"], tz=timezone.utc) for r in all_pnl]
    ys  = [r["total_value"] for r in all_pnl]
    up  = ys[-1] >= 10_000
    lc  = _GREEN if up else _RED
    fc  = "rgba(74,222,128,0.12)" if up else "rgba(248,113,113,0.12)"
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.08, 5)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy",
                              fillcolor=fc, line=dict(color=lc, width=2),
                              name="Portfolio Value"))
    fig.add_hline(y=10_000, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(yaxis=dict(range=[y_min - pad, y_max + pad],
                                  showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                                  zeroline=False, title="Portfolio Value ($)",
                                  color=_MUTED))
    return fig


def _summary_cards(summary: dict):
    pnl     = summary.get("total_pnl", None)
    trades  = summary.get("total_trades", "—")
    settled = summary.get("total_settlements", "—")
    elapsed = summary.get("elapsed_s", None)
    pnl_str   = f"${pnl:+,.2f}" if pnl is not None else "—"
    pnl_color = _GREEN if (pnl or 0) >= 0 else _RED
    elapsed_s = f"{elapsed:.1f}s" if elapsed is not None else "—"

    def _c(label, value, color):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.P(label, style={"color": _MUTED, "fontSize": "0.78rem",
                                  "fontFamily": "Inter,sans-serif", "fontWeight": "500",
                                  "marginBottom": "4px", "letterSpacing": "0.04em",
                                  "textTransform": "uppercase"}),
            html.H5(value, style={"color": color, "fontFamily": "Inter,sans-serif",
                                   "fontWeight": "700", "margin": 0}),
        ]), style={"background": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px"}), xs=6, md=3)

    return [
        _c("Total P&L", pnl_str, pnl_color),
        _c("Trades",    str(trades),  _BLUE),
        _c("Settled",   str(settled), _MUTED),
        _c("Runtime",   elapsed_s,    _MUTED),
    ]


# ── Layout ─────────────────────────────────────────────────────────────────────

layout = dbc.Container(
    fluid=True, className="px-4 py-3",
    style={"backgroundColor": _BG, "minHeight": "100vh"},
    children=[
        dbc.Row(className="align-items-center mb-4 g-2", children=[
            dbc.Col(html.H4("Strategy Backtest",
                            style={"color": _TEXT, "fontFamily": "Inter,sans-serif",
                                   "fontWeight": "700", "margin": 0}), width="auto"),
            dbc.Col(dbc.Button("▶  Run Backtest", id="run-btn", color="success", size="md",
                               style={"backgroundColor": _ORANGE, "border": "none",
                                      "fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                      "borderRadius": "8px", "padding": "8px 20px"}),
                    width="auto"),
            dbc.Col(dbc.Badge("Ready", id="status-badge", color="secondary",
                               className="fs-6 px-3 py-2",
                               style={"fontFamily": "Inter,sans-serif"}), width="auto"),
            dbc.Col(dbc.Progress(id="progress-bar", value=0, striped=True, animated=True,
                                  style={"height": "8px", "minWidth": "200px",
                                         "borderRadius": "4px"},
                                  color="success", className="d-none"), width=3),
        ]),

        dbc.Row(className="g-3", children=[
            # Left: markets
            dbc.Col(width=5, children=[
                html.Div(style={"display": "flex", "alignItems": "center",
                                "marginBottom": "12px"}, children=[
                    html.Span("Active Markets",
                              style={"fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                     "fontSize": "0.75rem", "letterSpacing": "0.08em",
                                     "textTransform": "uppercase", "color": _MUTED}),
                    html.Span(id="market-count-badge",
                              style={"marginLeft": "8px",
                                     "backgroundColor": f"{_ORANGE}20", "color": _ORANGE,
                                     "fontSize": "0.70rem", "fontFamily": "Inter,sans-serif",
                                     "fontWeight": "600", "padding": "1px 8px",
                                     "borderRadius": "20px"}),
                ]),
                html.Div(id="markets-panel",
                         style={"maxHeight": "76vh", "overflowY": "auto",
                                "paddingRight": "4px"},
                         children=[dbc.Card(dbc.CardBody(
                             html.P("Click Run Backtest to begin.",
                                    style={"color": _MUTED, "margin": 0,
                                           "fontFamily": "Inter,sans-serif"})),
                             style={"background": _CARD,
                                    "border": f"1px solid {_BORDER}",
                                    "borderRadius": "12px"})]),
            ]),
            # Right: PNL
            dbc.Col(width=7, children=[
                html.Span("Portfolio P&L",
                          style={"fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                 "fontSize": "0.75rem", "letterSpacing": "0.08em",
                                 "textTransform": "uppercase", "color": _MUTED,
                                 "display": "block", "marginBottom": "12px"}),
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="pnl-chart", style={"height": "44vh"},
                              config={"displayModeBar": False},
                              figure=_empty_pnl_figure()),
                    style={"padding": "8px"},
                ), style={"background": _CARD, "border": f"1px solid {_BORDER}",
                           "borderRadius": "12px",
                           "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
                           "marginBottom": "14px"}),
                dbc.Row(id="summary-row", className="g-2",
                        children=_summary_cards({})),
            ]),
        ]),

        dcc.Interval(id="poll-interval", interval=250, disabled=True),
        dcc.Store(id="fc-cursor",      data=0),
        dcc.Store(id="fill-cursor",    data=0),
        dcc.Store(id="pnl-cursor",     data=0),
        dcc.Store(id="settle-cursor",  data=0),
        dcc.Store(id="market-data",    data={}),
        dcc.Store(id="settled-data",   data={}),
    ],
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("poll-interval",  "disabled"),
    Output("run-btn",        "disabled"),
    Output("status-badge",   "children"),
    Output("status-badge",   "color"),
    Output("progress-bar",   "className"),
    # Reset all cursors and stored data on new run
    Output("fc-cursor",      "data"),
    Output("fill-cursor",    "data"),
    Output("pnl-cursor",     "data"),
    Output("settle-cursor",  "data"),
    Output("market-data",    "data"),
    Output("settled-data",   "data"),
    Input("run-btn",         "n_clicks"),
    prevent_initial_call=True,
)
def on_run_click(n_clicks):
    run_backtest_async(_DATA_DIR, state, hours=None)
    return False, True, "Loading data…", "warning", "d-block", 0, 0, 0, 0, {}, {}


@callback(
    Output("markets-panel",       "children"),
    Output("market-count-badge",  "children"),
    Output("pnl-chart",           "figure"),
    Output("summary-row",         "children"),
    Output("status-badge",        "children",  allow_duplicate=True),
    Output("status-badge",        "color",     allow_duplicate=True),
    Output("run-btn",             "disabled",  allow_duplicate=True),
    Output("poll-interval",       "disabled",  allow_duplicate=True),
    Output("progress-bar",        "value"),
    Output("progress-bar",        "className", allow_duplicate=True),
    Output("fc-cursor",           "data",      allow_duplicate=True),
    Output("fill-cursor",         "data",      allow_duplicate=True),
    Output("pnl-cursor",          "data",      allow_duplicate=True),
    Output("settle-cursor",       "data",      allow_duplicate=True),
    Output("market-data",         "data",      allow_duplicate=True),
    Output("settled-data",        "data",      allow_duplicate=True),
    Input("poll-interval",        "n_intervals"),
    State("fc-cursor",            "data"),
    State("fill-cursor",          "data"),
    State("pnl-cursor",           "data"),
    State("settle-cursor",        "data"),
    State("market-data",          "data"),
    State("settled-data",         "data"),
    prevent_initial_call=True,
)
def poll_state(n_intervals, fc_cursor, fill_cursor, pnl_cursor,
               settle_cursor, market_data, settled_data):

    # Ingest forecast rows
    new_fc = state.forecast_rows[fc_cursor:]
    for row in new_fc:
        slug = row["slug"]
        if slug not in market_data:
            market_data[slug] = {"ts": [], "yes": [], "p": [], "fills": []}
        market_data[slug]["ts"].append(row["ts"])
        market_data[slug]["yes"].append(row["yes_price"])
        market_data[slug]["p"].append(row["p_model"])
    fc_cursor += len(new_fc)

    # Ingest fills
    new_fills = state.fill_rows[fill_cursor:]
    for fill in new_fills:
        slug = fill["slug"]
        if slug in market_data:
            idx = len(market_data[slug]["ts"]) - 1
            market_data[slug]["fills"].append({**fill, "chart_idx": idx})
    fill_cursor += len(new_fills)

    # Ingest settlements
    new_settlements = state.settlement_rows[settle_cursor:]
    for s in new_settlements:
        slug = s["slug"]
        settled_data[slug] = {
            "outcome":   s["outcome"],
            "btc_open":  s["btc_open"],
            "btc_close": s["btc_close"],
            "ts":        s["ts"],
            "fills":     market_data.get(slug, {}).get("fills", []),
        }
    settle_cursor += len(new_settlements)

    # Ingest PNL
    new_pnl    = state.pnl_rows[pnl_cursor:]
    pnl_cursor += len(new_pnl)
    all_pnl    = state.pnl_rows[:pnl_cursor]

    # ── Build market panel ────────────────────────────────────────────────────
    latest_ts = max(
        (d["ts"][-1] for d in market_data.values() if d["ts"]), default=0
    )

    # Active: ticks within 350s AND model traded on it
    active_slugs = [
        s for s, d in market_data.items()
        if d["ts"]
        and (latest_ts - d["ts"][-1]) < 350
        and len(d["fills"]) > 0
        and s not in settled_data
    ]

    def _sort_key(s):
        d = market_data[s]
        yp, np_ = _net_position(d["fills"])
        return (0 if abs(yp) > 0.01 or abs(np_) > 0.01 else 1,
                -(d["ts"][-1] if d["ts"] else 0))

    active_slugs = sorted(active_slugs, key=_sort_key)[:6]

    # Recent settlements (last 5, sorted newest first)
    recent_settled = sorted(settled_data.items(),
                             key=lambda kv: kv[1].get("ts", 0),
                             reverse=True)[:5]

    cards = []

    # Settlement cards first
    for slug, sd in recent_settled:
        cards.append(_settlement_card(
            slug,
            sd["outcome"],
            sd["btc_open"],
            sd["btc_close"],
            sd["fills"],
        ))

    # Active market cards
    for slug in active_slugs:
        d = market_data[slug]
        cards.append(_market_card(slug, d["ts"], d["yes"], d["p"], d["fills"]))

    if not cards:
        placeholder = ("Waiting for first trades…" if state.running
                       else "Click Run Backtest to begin.")
        cards = [dbc.Card(dbc.CardBody(
            html.P(placeholder, style={"color": _MUTED, "margin": 0,
                                        "fontFamily": "Inter,sans-serif"})),
            style={"background": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "12px"})]

    count_label = f"{len(active_slugs)} active · {len(recent_settled)} settled"

    # PNL figure
    fig = _safe_figure(lambda: _pnl_figure(all_pnl))

    # Status
    total     = max(state.total_ticks, 1)
    processed = state.processed_ticks
    progress  = min(int(processed / total * 100), 100)

    if state.error:
        bt, bc  = f"Error: {state.error[:60]}", "danger"
        bd, ivd = False, True;  pcls = "d-none"
    elif state.done:
        bt, bc  = "Done ✓", "success"
        bd, ivd = False, True;  pcls = "d-none"
    elif state.running and state.total_ticks == 0:
        bt, bc  = "Loading data…", "warning"
        bd, ivd = True,  False;  pcls = "d-block"
    elif state.running:
        bt, bc  = f"Tick {processed:,} / {total:,}", "warning"
        bd, ivd = True,  False;  pcls = "d-block"
    else:
        bt, bc  = "Ready", "secondary"
        bd, ivd = False, True;  pcls = "d-none"

    return (
        cards, count_label, fig, _summary_cards(state.summary),
        bt, bc, bd, ivd, progress, pcls,
        fc_cursor, fill_cursor, pnl_cursor, settle_cursor,
        market_data, settled_data,
    )
