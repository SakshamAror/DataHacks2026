#!/usr/bin/env python3
"""
OGQuants dashboard server.
Serves the frontend and exposes /api/run-backtest.
"""
import json
import logging
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent
PROJECT_DIR  = FRONTEND_DIR.parent

sys.path.insert(0, str(PROJECT_DIR))

from backtester.data_loader import build_timeline
from backtester.engine import BacktestEngine
from backtester.runner import load_strategy_from_file
from backtester.scoring import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def snapshots_to_series(snapshots):
    """Return {time, value} points sampled at ~1-minute intervals."""
    return [
        {"time": snap.timestamp, "value": round(snap.total_value, 2)}
        for snap in snapshots
    ]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/run-backtest":
            self._handle_backtest()
        else:
            self.send_error(404)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _handle_backtest(self):
        try:
            log.info("Backtest started: my_strategy.py --hours 4 --assets BTC --intervals 5m")

            strategy_path = PROJECT_DIR / "my_strategy.py"
            data_dir      = PROJECT_DIR / "data" / "train"

            strategy = load_strategy_from_file(strategy_path)
            data = build_timeline(
                data_dir=data_dir,
                intervals=["5m"],
                hours=4,
                assets=["BTC"],
            )

            engine = BacktestEngine(
                data=data,
                strategy=strategy,
                starting_cash=10_000.0,
                snapshot_interval=60,
            )
            result = engine.run()
            score  = compute_score(result)

            from datetime import datetime, timezone
            def fmt(ts):
                if ts <= 0: return "N/A"
                return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d, %Y %H:%M UTC")

            self._send_json(200, {
                "metrics": {
                    "total_pnl":            round(score.total_pnl, 2),
                    "return_pct":           round(score.return_pct, 2),
                    "sharpe_ratio":         round(score.sharpe_ratio, 2),
                    "max_drawdown":         round(score.max_drawdown, 2),
                    "max_drawdown_pct":     round(score.max_drawdown_pct, 2),
                    "win_rate":             round(score.win_rate * 100, 1),
                    "total_trades":         score.total_trades,
                    "total_settlements":    score.total_settlements,
                    "avg_trade_pnl":        round(score.avg_trade_pnl, 4),
                    "final_portfolio_value": round(score.final_portfolio_value, 2),
                    "starting_cash":        score.starting_cash,
                    "competition_score":    round(score.competition_score, 2),
                },
                "series": snapshots_to_series(result.portfolio_snapshots),
                "period": {
                    "start":    fmt(result.start_ts),
                    "end":      fmt(result.end_ts),
                    "start_ts": result.start_ts,
                    "end_ts":   result.end_ts,
                },
            })
            log.info("Backtest complete — P&L: $%.2f  Sharpe: %.2f", score.total_pnl, score.sharpe_ratio)

        except Exception as exc:
            log.exception("Backtest failed")
            self._send_json(500, {"error": str(exc)})


if __name__ == "__main__":
    port   = 3456
    server = ThreadingHTTPServer(("localhost", port), Handler)
    log.info("OGQuants dashboard → http://localhost:%d", port)
    server.serve_forever()
