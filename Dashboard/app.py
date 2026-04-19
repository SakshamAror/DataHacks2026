"""
DataHacks 2026 — Strategy Dashboard
Run: cd Dashboard && python app.py  →  http://localhost:8050
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DASH_JUPYTER_MODE", "false")

try:
    import comm as _comm
    class _NoOpComm:
        def on_close(self, *a, **kw): pass
        def on_msg(self, *a, **kw): pass
        def send(self, *a, **kw): pass
        def close(self, *a, **kw): pass
    _comm.create_comm = lambda *a, **kw: _NoOpComm()
except ImportError:
    pass

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent))

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Raleway:wght@600;700;800&"
    "family=Inter:wght@300;400;500;600;700&"
    "display=swap"
)

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, _FONTS],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "DataHacks 2026"

_CSS = """
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  height: 100%;
  font-family: 'Inter', system-ui, sans-serif;
  color: #ffffff !important;
  overflow-x: hidden;
  background: transparent !important;
  background-color: transparent !important;
}
#react-entry-point, #_dash-app-content,
._dash-loading, .dash-loading {
  background: transparent !important;
}

/* ── Bokeh background ── */
.bg-bokeh {
  position: fixed;
  inset: 0;
  z-index: -2;
  background: linear-gradient(168deg, #6DB3D4 0%, #2d6a4f 46%, #6B5B3E 100%);
  overflow: hidden;
}
.bg-bokeh::before {
  content: '';
  position: absolute;
  inset: -120px;
  background:
    radial-gradient(circle at 12% 22%, rgba(120,200,120,0.78) 0%, transparent 22%),
    radial-gradient(circle at 78% 10%, rgba(140,200,240,0.82) 0%, transparent 20%),
    radial-gradient(circle at 20% 74%, rgba(50,140,60,0.88)   0%, transparent 18%),
    radial-gradient(circle at 84% 78%, rgba(220,180,55,0.78)  0%, transparent 22%),
    radial-gradient(circle at 50% 50%, rgba(80,170,90,0.42)   0%, transparent 35%),
    radial-gradient(circle at 66% 36%, rgba(180,220,100,0.52) 0%, transparent 16%),
    radial-gradient(circle at 36% 28%, rgba(160,210,200,0.42) 0%, transparent 18%),
    radial-gradient(circle at 92% 42%, rgba(100,180,80,0.55)  0%, transparent 14%);
  filter: blur(72px);
}

/* ── Glass pane ── */
.glass-pane {
  background: rgba(255,255,255,0.09) !important;
  backdrop-filter: blur(28px) !important;
  -webkit-backdrop-filter: blur(28px) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 24px !important;
  box-shadow: 0 8px 40px rgba(0,0,0,0.20) !important;
}
.glass-pane > .card-body {
  background: transparent !important;
  padding: 20px 24px 24px !important;
}

/* ── Pane header divider ── */
.pane-header-divider {
  border: none;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  margin: 12px 0 16px;
}

/* ── Pane sub-nav ── */
.pane-sub-nav .nav-link {
  color: rgba(255,255,255,0.75) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  font-family: 'Inter', sans-serif !important;
  padding: 6px 16px !important;
  border-radius: 8px !important;
  transition: background 0.15s, color 0.15s;
}
.pane-sub-nav .nav-link:hover {
  color: #ffffff !important;
  background: rgba(255,255,255,0.08) !important;
}
.pane-sub-nav .nav-link.active {
  color: #4ade80 !important;
  font-weight: 600 !important;
  background: rgba(74,222,128,0.10) !important;
}

/* ── Cards (inner sections) ── */
.card {
  background: rgba(255,255,255,0.07) !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 16px !important;
  color: #ffffff !important;
  box-shadow: 0 2px 16px rgba(0,0,0,0.12) !important;
}
.card-body { background: transparent !important; color: #ffffff !important; }

/* ── Badges ── */
.badge { border-radius: 20px !important; font-family: 'Inter', sans-serif; }

/* ── Buttons ── */
.btn { font-family: 'Inter', sans-serif; border-radius: 50px !important; }
.btn-outline-secondary {
  color: rgba(255,255,255,0.7) !important;
  border-color: rgba(255,255,255,0.25) !important;
  background: transparent !important;
}
.btn-outline-secondary:hover {
  background: rgba(255,255,255,0.1) !important;
  color: #ffffff !important;
  border-color: rgba(255,255,255,0.4) !important;
}

/* ── Tables ── */
.table { color: #ffffff !important; }
.table > :not(caption) > * > * {
  color: #ffffff !important;
  background: transparent !important;
  border-color: rgba(255,255,255,0.1) !important;
}
.table-hover > tbody > tr:hover > * { background: rgba(255,255,255,0.05) !important; }

/* ── Progress ── */
.progress { background: rgba(255,255,255,0.1) !important; border-radius: 20px !important; }
.progress-bar { border-radius: 20px !important; }

/* ── Headings ── */
h1,h2,h3,h4,h5,h6 { font-family: 'Raleway', sans-serif !important; color: #ffffff !important; }

/* ── Code ── */
code { color: #4ade80 !important; background: rgba(74,222,128,0.1) !important;
       padding: 1px 6px !important; border-radius: 4px !important; }

/* ── Links ── */
a { color: #4ade80 !important; }
a:hover { color: #86efac !important; }
/* Pill nav — active pill must show dark text on white background */
#pill-nav a.pill-active         { color: #141413 !important; }
#pill-nav a.pill-active:hover   { color: #141413 !important; }
#pill-nav a.pill-inactive       { color: rgba(255,255,255,0.5) !important; }
#pill-nav a.pill-inactive:hover { color: #ffffff !important; }
/* Pill nav inherits inline color */
#pill-nav a { color: inherit !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.35); }

/* ── Markdown ── */
.dash-markdown p, .dash-markdown li, .dash-markdown blockquote {
  color: rgba(255,255,255,0.85) !important;
  font-family: 'Inter', sans-serif !important;
}
.dash-markdown h1,.dash-markdown h2,.dash-markdown h3 {
  font-family: 'Raleway', sans-serif !important;
  color: #ffffff !important;
}
.dash-markdown code { color: #4ade80 !important; background: rgba(74,222,128,0.12) !important; }
.dash-markdown pre { background: rgba(0,0,0,0.25) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; }
"""

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>""" + _CSS + """</style>
</head>
<body>
<div class="bg-bokeh"></div>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# ── Pane header components ──────────────────────────────────────────────────────

_LOGO = html.Span([
    html.Span("DataHacks", style={"color": "#4ade80", "fontWeight": "800",
                                   "fontFamily": "Raleway, sans-serif"}),
    html.Span(" 2026", style={"color": "#ffffff", "fontWeight": "600",
                               "fontFamily": "Raleway, sans-serif"}),
], style={"fontSize": "1.1rem", "letterSpacing": "-0.02em"})

_PILL_CONTAINER_STYLE = {
    "display": "inline-flex", "alignItems": "center", "gap": "2px",
    "background": "rgba(255,255,255,0.10)", "borderRadius": "50px",
    "padding": "3px", "border": "1px solid rgba(255,255,255,0.18)",
}
_PILL_ACTIVE = {
    "padding": "6px 22px", "borderRadius": "50px",
    "background": "#ffffff", "color": "#141413",
    "fontFamily": "Inter, sans-serif", "fontWeight": "600",
    "fontSize": "0.88rem", "boxShadow": "0 1px 6px rgba(0,0,0,0.18)",
    "display": "inline-block", "textDecoration": "none",
}
_PILL_INACTIVE = {
    "padding": "6px 22px", "borderRadius": "50px",
    "color": "rgba(255,255,255,0.72)",
    "fontFamily": "Inter, sans-serif", "fontWeight": "500",
    "fontSize": "0.88rem", "display": "inline-block",
    "textDecoration": "none", "cursor": "pointer",
}

_PILL_NAV = html.Div(id="pill-nav", style=_PILL_CONTAINER_STYLE)

_SUB_NAV = dbc.Nav([
    dbc.NavItem(dbc.NavLink("Backtest",       href="/",               active="exact")),
    dbc.NavItem(dbc.NavLink("Model",          href="/model",          active="exact")),
    dbc.NavItem(dbc.NavLink("Factor Library", href="/factor-library", active="exact")),
], className="pane-sub-nav")

# ── Root layout ────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "transparent",
           "padding": "16px 20px 20px"},
    children=[
        dcc.Location(id="_root-url", refresh=False),

        dbc.Card(className="glass-pane", children=[
            dbc.CardBody([
                # Brand row
                dbc.Row([
                    dbc.Col(_LOGO, width="auto"),
                    dbc.Col(_PILL_NAV,
                            style={"flex": "1", "display": "flex",
                                   "justifyContent": "center", "alignItems": "center"},
                            width="auto"),
                    dbc.Col(width="auto", style={"minWidth": "130px"}),
                ], align="center", className="mb-0"),

                html.Hr(className="pane-header-divider"),

                # Sub-nav
                _SUB_NAV,

                html.Div(style={"height": "12px"}),

                # Page content
                dash.page_container,
            ]),
        ]),
    ],
)

@app.callback(Output("pill-nav", "children"), Input("_root-url", "pathname"))
def _pill_nav(pathname):
    on_buy = pathname == "/buy"
    return [
        dcc.Link("Portfolio", href="/",    className="pill-active" if not on_buy else "pill-inactive",
                 style=_PILL_ACTIVE   if not on_buy else _PILL_INACTIVE),
        dcc.Link("Buy",       href="/buy", className="pill-active" if on_buy     else "pill-inactive",
                 style=_PILL_ACTIVE   if on_buy     else _PILL_INACTIVE),
    ]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
