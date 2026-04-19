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
from dash import dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent))

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Poppins:wght@400;500;600;700&"
    "family=Lora:ital,wght@0,400;0,500;1,400&"
    "display=swap"
)

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY, _FONTS],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "DataHacks 2026"

# ── Global style injection ─────────────────────────────────────────────────────

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  body { background: #faf9f5; font-family: 'Lora', Georgia, serif; }
  h1,h2,h3,h4,h5,h6,.fw-bold,.fw-semibold { font-family: 'Poppins', system-ui, sans-serif; }
  .navbar-brand { font-family: 'Poppins', sans-serif; }
  a { color: #d97757; }
  a:hover { color: #b85e3a; }
  .card { border-radius: 12px !important; }
  .badge { border-radius: 20px !important; font-family: 'Poppins', sans-serif; }
  .btn { font-family: 'Poppins', sans-serif; border-radius: 8px !important; }
  .nav-link { font-family: 'Poppins', sans-serif; font-weight: 500; }
  .nav-link.active { color: #d97757 !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #f0ede6; }
  ::-webkit-scrollbar-thumb { background: #c8c4bb; border-radius: 3px; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# ── Navbar ─────────────────────────────────────────────────────────────────────

_LOGO = html.Span([
    html.Span("DataHacks", style={"color": "#d97757", "fontWeight": "700"}),
    html.Span(" 2026", style={"color": "#141413", "fontWeight": "400"}),
], style={"fontSize": "1.1rem", "letterSpacing": "-0.01em", "fontFamily": "Poppins, sans-serif"})

navbar = dbc.Navbar(
    dbc.Container(fluid=True, children=[
        dbc.NavbarBrand(_LOGO, href="/"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Backtest",       href="/",               active="exact")),
            dbc.NavItem(dbc.NavLink("Model",          href="/model",          active="exact")),
            dbc.NavItem(dbc.NavLink("Factor Library", href="/factor-library", active="exact")),
        ], navbar=True, className="ms-auto"),
    ]),
    color="white",
    dark=False,
    className="mb-0",
    style={"borderBottom": "1px solid #e8e6dc", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
)

# ── Root layout ────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [dcc.Location(id="_root-url", refresh=False), navbar, dash.page_container],
    style={"minHeight": "100vh", "backgroundColor": "#faf9f5"},
)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
