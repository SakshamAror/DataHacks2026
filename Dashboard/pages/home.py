"""
Home page — Run Backtest + live market view (left) + PNL chart (right).
"""

import sys
from pathlib import Path

import pandas as _pd  # noqa: F401

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_state import state
from backtest_adapter import run_backtest_async

_DATAHACKS = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = str(_DATAHACKS / "data" / "train")
if not Path(_DATA_DIR).exists():
    _DATA_DIR = str(_DATAHACKS / "datasets" / "train")

dash.register_page(__name__, path="/", name="Backtest", order=0)

# ── Glass palette ──────────────────────────────────────────────────────────────
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
    parts = slug.lower().split("-")
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_figure(build_fn):
    try:
        return build_fn()
    except Exception:
        return go.Figure()


def _glass_layout(**extra):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.65)", family="Inter, system-ui, sans-serif"),
        margin=dict(l=60, r=20, t=10, b=40),
    )
    base.update(extra)
    return base


def _empty_pnl_figure():
    fig = go.Figure()
    fig.update_layout(
        **_glass_layout(margin=dict(l=70, r=20, t=10, b=40)),
        xaxis=dict(showgrid=False, zeroline=False, title="",
                   color=_MUTED, linecolor=_BORDER),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
                   title="Portfolio Value ($)", color=_MUTED,
                   linecolor="rgba(255,255,255,0.08)"),
    )
    return fig


def _pnl_figure(all_pnl):
    from datetime import datetime, timezone
    fig = _empty_pnl_figure()
    if not all_pnl:
        return fig

    xs = [datetime.fromtimestamp(r["ts"], tz=timezone.utc) for r in all_pnl]
    ys = [r["total_value"] for r in all_pnl]
    up = ys[-1] >= 10_000
    line_color = _GREEN if up else _RED
    fill_color = "rgba(74,222,128,0.12)" if up else "rgba(248,113,113,0.12)"

    y_min = min(ys)
    y_max = max(ys)
    pad   = max((y_max - y_min) * 0.08, 5)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        name="Portfolio Value",
    ))
    fig.add_hline(y=10_000, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(
        yaxis=dict(
            range=[y_min - pad, y_max + pad],
            showgrid=True, gridcolor="rgba(255,255,255,0.07)", zeroline=False,
            title="Portfolio Value ($)", color=_MUTED,
        )
    )
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
            html.P(label, style={"color": _MUTED, "fontSize": "0.72rem",
                                  "fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                  "marginBottom": "4px", "letterSpacing": "0.05em",
                                  "textTransform": "uppercase"}),
            html.H5(value, style={"color": color, "fontFamily": "Raleway,sans-serif",
                                   "fontWeight": "700", "margin": 0}),
        ]), style={"background": _CARD, "border": f"1px solid {_BORDER}",
                   "borderRadius": "14px", "backdropFilter": "blur(20px)"}), xs=6, md=3)

    return [
        _c("Total P&L", pnl_str, pnl_color),
        _c("Trades",    str(trades),  _BLUE),
        _c("Settled",   str(settled), _MUTED),
        _c("Runtime",   elapsed_s,    _MUTED),
    ]


def _net_position(fills):
    yes, no = 0.0, 0.0
    for f in fills:
        sign = 1 if f["side"] == "BUY" else -1
        if f["token"] == "YES":
            yes += sign * f["size"]
        else:
            no  += sign * f["size"]
    return yes, no


def _fill_blocks(fills):
    if not fills:
        return html.Div()
    blocks = []
    for f in fills:
        is_buy = f["side"] == "BUY"
        bg     = "rgba(74,222,128,0.25)"  if is_buy else "rgba(248,113,113,0.25)"
        border = "rgba(74,222,128,0.5)"   if is_buy else "rgba(248,113,113,0.5)"
        color  = "#4ade80" if is_buy else "#f87171"
        arrow  = "▲" if is_buy else "▼"
        token  = f.get("token", "YES")
        size   = int(f.get("size", 0))
        price  = f.get("avg_price", 0)
        blocks.append(html.Span(
            f"{arrow} {f['side']} {token}  {size} @ {price:.3f}",
            style={
                "backgroundColor": bg,
                "color": color,
                "border": f"1px solid {border}",
                "fontSize": "0.70rem",
                "fontFamily": "monospace",
                "fontWeight": "600",
                "padding": "2px 10px",
                "borderRadius": "20px",
                "display": "inline-block",
                "letterSpacing": "0.01em",
            },
        ))
    return html.Div(blocks, style={"display": "flex", "flexWrap": "wrap", "gap": "5px",
                                    "marginTop": "8px"})


def _market_card(slug, ts_list, yes_list, p_list, fills):
    title, subtitle = _parse_slug(slug)
    yes_pos, no_pos = _net_position(fills)
    has_position    = abs(yes_pos) > 0.01 or abs(no_pos) > 0.01

    if has_position:
        pos_parts = []
        if abs(yes_pos) > 0.01: pos_parts.append(f"YES {yes_pos:+.0f}")
        if abs(no_pos)  > 0.01: pos_parts.append(f"NO {no_pos:+.0f}")
        pos_badge = dbc.Badge(
            "  |  ".join(pos_parts),
            style={"backgroundColor": "rgba(74,222,128,0.15)", "color": _GREEN,
                   "border": "1px solid rgba(74,222,128,0.4)",
                   "fontSize": "0.70rem", "fontFamily": "Inter,sans-serif",
                   "borderRadius": "20px"},
        )
    else:
        pos_badge = dbc.Badge("No open position",
                               style={"backgroundColor": "rgba(255,255,255,0.08)",
                                      "color": _MUTED, "border": f"1px solid {_BORDER}",
                                      "fontSize": "0.70rem", "borderRadius": "20px"})

    fig = go.Figure()
    if yes_list:
        xs = list(range(len(yes_list)))
        fig.add_trace(go.Scatter(x=xs, y=yes_list, mode="lines", name="Market",
                                  line=dict(color=_BLUE, width=1.6)))
        fig.add_trace(go.Scatter(x=xs, y=p_list, mode="lines", name="Model",
                                  line=dict(color=_ORANGE, width=1.6, dash="dot")))
        for f in fills:
            idx       = f.get("chart_idx", len(xs) - 1)
            yv        = f.get("avg_price", 0.5)
            side      = f.get("side", "BUY")
            pin_color = _GREEN if side == "BUY" else _RED
            sym       = "triangle-up" if side == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[idx], y=[yv], mode="markers",
                marker=dict(symbol=sym, size=11, color=pin_color,
                            line=dict(width=1.5, color="rgba(255,255,255,0.8)")),
                showlegend=False,
                hovertext=f"{side} {f.get('token','?')} {int(f.get('size',0))} @ {f.get('avg_price',0):.3f}",
                hoverinfo="text",
            ))

    fig.update_layout(
        **_glass_layout(margin=dict(l=32, r=8, t=6, b=18)),
        height=120,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", range=[0, 1],
                   tickformat=".0%", tickfont=dict(size=9, color=_MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=9, color=_MUTED),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        hovermode="x unified",
    )

    last_yes = yes_list[-1] if yes_list else None
    last_p   = p_list[-1]   if p_list   else None
    edge_str = ""
    if last_yes is not None and last_p is not None:
        edge       = last_p - last_yes
        edge_color = _GREEN if edge > 0.04 else (_ORANGE if edge > 0 else _RED)
        edge_str   = html.Span(f"Edge {edge:+.3f}",
                               style={"color": edge_color, "fontFamily": "monospace",
                                      "fontSize": "0.75rem", "fontWeight": "600"})

    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(title, style={"fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                           "fontSize": "0.85rem", "color": _TEXT,
                                           "letterSpacing": "-0.01em"}),
                    html.Div(subtitle, style={"color": _MUTED, "fontSize": "0.73rem",
                                              "fontFamily": "monospace"}),
                ], width=7),
                dbc.Col([
                    html.Div([pos_badge], className="text-end"),
                    html.Div(edge_str, className="text-end mt-1") if edge_str else html.Div(),
                ], width=5),
            ], className="mb-1 align-items-start"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "120px"}),
            _fill_blocks(fills),
        ], style={"padding": "12px 14px 10px"}),
        style={"background": _CARD, "border": f"1px solid {_BORDER}",
               "borderRadius": "16px", "marginBottom": "10px",
               "backdropFilter": "blur(20px)"},
    )


# ── Layout ─────────────────────────────────────────────────────────────────────

layout = html.Div([
        # ── Stores / interval ─────────────────────────────────────────────────
        dcc.Interval(id="poll-interval", interval=250, disabled=True),
        dcc.Store(id="fc-cursor",   data=0),
        dcc.Store(id="fill-cursor", data=0),
        dcc.Store(id="pnl-cursor",  data=0),
        dcc.Store(id="market-data", data={}),

        # ── Header ───────────────────────────────────────────────────────────
        dbc.Row(className="align-items-center mb-4 g-2", children=[
            dbc.Col(html.H4("Strategy Backtest",
                            style={"color": _TEXT, "fontFamily": "Raleway,sans-serif",
                                   "fontWeight": "700", "margin": 0}), width="auto"),
            dbc.Col([
                dbc.Button("▶  Run Backtest", id="run-btn", color="success", size="md",
                           style={"background": _GREEN, "border": "none",
                                  "fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                  "borderRadius": "50px", "padding": "8px 22px",
                                  "color": "#14532d"}),
                dbc.Badge("Ready", id="status-badge", color="secondary",
                          className="fs-6 px-3 py-2 ms-2",
                          style={"fontFamily": "Inter,sans-serif", "borderRadius": "20px",
                                 "background": "rgba(255,255,255,0.12)", "color": _TEXT}),
                dbc.Progress(id="progress-bar", value=0, striped=True, animated=True,
                             style={"height": "8px", "minWidth": "160px", "borderRadius": "20px",
                                    "marginTop": "6px"},
                             color="success", className="d-none"),
            ], width="auto", className="d-flex align-items-center flex-wrap ms-auto"),
        ]),

        # ── Content ──────────────────────────────────────────────────────────
        dbc.Row(className="g-3", children=[
            dbc.Col(width=5, children=[
                html.Div(style={"display": "flex", "alignItems": "center",
                                "marginBottom": "12px"}, children=[
                    html.Span("Active Markets",
                              style={"fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                     "fontSize": "0.72rem", "letterSpacing": "0.08em",
                                     "textTransform": "uppercase", "color": _MUTED}),
                    html.Span(id="market-count-badge",
                              style={"marginLeft": "8px",
                                     "backgroundColor": "rgba(74,222,128,0.15)",
                                     "color": _GREEN, "fontSize": "0.70rem",
                                     "fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                     "padding": "1px 8px", "borderRadius": "20px",
                                     "border": "1px solid rgba(74,222,128,0.35)"}),
                ]),
                html.Div(id="markets-panel",
                         style={"maxHeight": "68vh", "overflowY": "auto", "paddingRight": "4px"},
                         children=[dbc.Card(dbc.CardBody(
                             html.P("Click Run Backtest to begin.",
                                    style={"color": _MUTED, "margin": 0,
                                           "fontFamily": "Inter,sans-serif"})),
                             style={"background": "rgba(255,255,255,0.06)",
                                    "border": f"1px solid {_BORDER}",
                                    "borderRadius": "16px"})]),
            ]),

            dbc.Col(width=7, children=[
                html.Span("Portfolio P&L",
                          style={"fontFamily": "Inter,sans-serif", "fontWeight": "600",
                                 "fontSize": "0.72rem", "letterSpacing": "0.08em",
                                 "textTransform": "uppercase", "color": _MUTED,
                                 "display": "block", "marginBottom": "12px"}),
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="pnl-chart", style={"height": "42vh"},
                              config={"displayModeBar": False},
                              figure=_empty_pnl_figure()),
                    style={"padding": "8px"},
                ), style={"background": "rgba(255,255,255,0.06)",
                           "border": f"1px solid {_BORDER}",
                           "borderRadius": "16px", "marginBottom": "14px"}),
                dbc.Row(id="summary-row", className="g-2", children=_summary_cards({})),
            ]),
        ]),
])


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("poll-interval", "disabled"),
    Output("run-btn", "disabled"),
    Output("status-badge", "children"),
    Output("status-badge", "color"),
    Output("progress-bar", "className"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_run_click(n_clicks):
    run_backtest_async(_DATA_DIR, state, hours=None)
    return False, True, "Loading data…", "warning", "d-block"


@callback(
    Output("markets-panel",   "children"),
    Output("market-count-badge", "children"),
    Output("pnl-chart",       "figure"),
    Output("summary-row",     "children"),
    Output("status-badge",    "children",  allow_duplicate=True),
    Output("status-badge",    "color",     allow_duplicate=True),
    Output("run-btn",         "disabled",  allow_duplicate=True),
    Output("poll-interval",   "disabled",  allow_duplicate=True),
    Output("progress-bar",    "value"),
    Output("progress-bar",    "className", allow_duplicate=True),
    Output("fc-cursor",       "data"),
    Output("fill-cursor",     "data"),
    Output("pnl-cursor",      "data"),
    Output("market-data",     "data"),
    Input("poll-interval",    "n_intervals"),
    State("fc-cursor",        "data"),
    State("fill-cursor",      "data"),
    State("pnl-cursor",       "data"),
    State("market-data",      "data"),
    prevent_initial_call=True,
)
def poll_state(n_intervals, fc_cursor, fill_cursor, pnl_cursor, market_data):
    new_fc = state.forecast_rows[fc_cursor:]
    for row in new_fc:
        slug = row["slug"]
        if slug not in market_data:
            market_data[slug] = {"ts": [], "yes": [], "p": [], "fills": []}
        market_data[slug]["ts"].append(row["ts"])
        market_data[slug]["yes"].append(row["yes_price"])
        market_data[slug]["p"].append(row["p_model"])
    fc_cursor += len(new_fc)

    new_fills = state.fill_rows[fill_cursor:]
    for fill in new_fills:
        slug = fill["slug"]
        if slug in market_data:
            idx = len(market_data[slug]["ts"]) - 1
            market_data[slug]["fills"].append({**fill, "chart_idx": idx})
    fill_cursor += len(new_fills)

    new_pnl = state.pnl_rows[pnl_cursor:]
    pnl_cursor += len(new_pnl)
    all_pnl = state.pnl_rows[:pnl_cursor]

    latest_ts = max(
        (d["ts"][-1] for d in market_data.values() if d["ts"]), default=0,
    )
    active_slugs = [
        s for s, d in market_data.items()
        if d["ts"] and (latest_ts - d["ts"][-1]) < 350 and len(d["fills"]) > 0
    ]

    def _sort_key(s):
        d = market_data[s]
        yp, np_ = _net_position(d["fills"])
        has_open = abs(yp) > 0.01 or abs(np_) > 0.01
        return (0 if has_open else 1, -(d["ts"][-1] if d["ts"] else 0))

    active_slugs = sorted(active_slugs, key=_sort_key)[:8]

    if active_slugs:
        market_cards = [
            _market_card(s, market_data[s]["ts"], market_data[s]["yes"],
                         market_data[s]["p"], market_data[s]["fills"])
            for s in active_slugs
        ]
        count_label = str(len(active_slugs))
    else:
        placeholder_msg = "Waiting for first trades…" if state.running else "Click Run Backtest to begin."
        market_cards = [dbc.Card(dbc.CardBody(
            html.P(placeholder_msg,
                   style={"color": _MUTED, "margin": 0, "fontFamily": "Inter,sans-serif"})),
            style={"background": _CARD, "border": f"1px solid {_BORDER}", "borderRadius": "16px"})]
        count_label = "0"

    fig = _safe_figure(lambda: _pnl_figure(all_pnl))

    total     = max(state.total_ticks, 1)
    processed = state.processed_ticks
    progress  = min(int(processed / total * 100), 100)

    if state.error:
        bt, bc, bd, ivd, pcls = f"Error: {state.error[:60]}", "danger",  False, True,  "d-none"
    elif state.done:
        bt, bc, bd, ivd, pcls = "Done ✓",                    "success", False, True,  "d-none"
    elif state.running and state.total_ticks == 0:
        bt, bc, bd, ivd, pcls = "Loading data…",             "warning", True,  False, "d-block"
    elif state.running:
        bt, bc, bd, ivd, pcls = f"Tick {processed:,} / {total:,}", "warning", True, False, "d-block"
    else:
        bt, bc, bd, ivd, pcls = "Ready",                     "secondary", False, True, "d-none"

    return (
        market_cards, count_label, fig, _summary_cards(state.summary),
        bt, bc, bd, ivd, progress, pcls,
        fc_cursor, fill_cursor, pnl_cursor, market_data,
    )
