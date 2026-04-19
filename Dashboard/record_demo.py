"""
Run the backtest once using my_strategy and save all streaming data to
demo_results.json for offline/hosted replay.

Usage:
    cd Dashboard
    python record_demo.py
"""

import json
import sys
from pathlib import Path

_PROJ  = Path(__file__).resolve().parent.parent
_STRAT = _PROJ / "strategies"
for p in (_PROJ, _STRAT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtester.data_loader import build_timeline
from backtester.engine import BacktestEngine
from my_strategy import MyStrategy

_DATA_DIR = _PROJ / "data" / "validation"
if not _DATA_DIR.exists():
    _DATA_DIR = _PROJ / "datasets" / "validation"

OUT = Path(__file__).resolve().parent / "demo_results.json"


class RecordingEngine(BacktestEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._fc_cursor     = 0
        self._fill_cursor   = 0
        self._settle_cursor = 0
        self.rec_forecast   = []
        self.rec_fills      = []
        self.rec_pnl        = []
        self.rec_settlements = []

    def _process_tick(self, tick):
        super()._process_tick(tick)

        # Forecast rows
        n = len(self.forecast_records)
        for rec in self.forecast_records[self._fc_cursor:]:
            self.rec_forecast.append({
                "ts": rec.timestamp,
                "slug": rec.market_slug,
                "yes_price": rec.market_yes_price,
                "p_model": rec.model_forecast,
            })
        self._fc_cursor = n

        # Fill rows
        nf = len(self.all_fills)
        for fill in self.all_fills[self._fill_cursor:]:
            self.rec_fills.append({
                "ts": fill.timestamp,
                "slug": fill.market_slug,
                "token": fill.token.value,
                "side": fill.side.value,
                "size": fill.size,
                "avg_price": fill.avg_price,
            })
        self._fill_cursor = nf

        # PnL snapshot
        if self.snapshots:
            snap = self.snapshots[-1]
            self.rec_pnl.append({
                "ts": snap.timestamp,
                "total_value": snap.total_value,
                "realized_pnl": snap.realized_pnl,
                "unrealized_pnl": snap.unrealized_pnl,
            })

        # Settlement rows
        ns = len(self.all_settlements)
        for s in self.all_settlements[self._settle_cursor:]:
            self.rec_settlements.append({
                "ts": s.end_ts,
                "slug": s.market_slug,
                "outcome": s.outcome.value,
                "btc_open": s.chainlink_open,
                "btc_close": s.chainlink_close,
            })
        self._settle_cursor = ns


def main():
    print(f"Loading data from {_DATA_DIR} …")
    data = build_timeline(
        data_dir=_DATA_DIR,
        intervals=["5m", "15m", "hourly"],
        hours=None,
        assets=["BTC"],
    )
    print(f"  {len(data.timeline):,} ticks loaded")

    engine = RecordingEngine(
        data=data,
        strategy=MyStrategy(),
        starting_cash=10_000.0,
        snapshot_interval=1,
    )
    print("Running backtest …")
    result = engine.run()
    print(f"  Done — {result.total_trades} trades, P&L ${result.total_pnl:+,.2f}")

    demo = {
        "summary": {
            "total_pnl": result.total_pnl,
            "final_value": result.final_portfolio_value,
            "total_trades": result.total_trades,
            "total_settlements": result.total_settlements,
            "total_rejected": result.total_rejected,
            "elapsed_s": result.elapsed_seconds,
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
        },
        "total_ticks": len(data.timeline),
        "forecast_rows": engine.rec_forecast,
        "fill_rows": engine.rec_fills,
        "pnl_rows": engine.rec_pnl,
        "settlement_rows": engine.rec_settlements,
    }

    print(f"Saving {len(engine.rec_pnl):,} PnL points, "
          f"{len(engine.rec_fills):,} fills, "
          f"{len(engine.rec_settlements):,} settlements …")
    with open(OUT, "w") as f:
        json.dump(demo, f, separators=(",", ":"))
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
