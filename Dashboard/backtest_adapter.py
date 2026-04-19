"""
Runs the backtest in a background daemon thread, streaming tick data into
shared_state so the Dash UI can poll it every 250ms.

We subclass BacktestEngine and override _process_tick to emit live data
without touching the engine's core logic.
"""

import sys
import threading
from pathlib import Path  # noqa: F401 — used in _worker

# Dashboard lives inside DATAHACKS2026/, so parent.parent IS the project root.
_PROJ = Path(__file__).resolve().parent.parent
_STRAT = _PROJ / "strategies"
for p in (_PROJ, _STRAT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_state import BacktestState


# ── Streaming engine subclass ─────────────────────────────────────────────────

def _make_streaming_engine(base_engine_cls, _state: BacktestState):
    """Factory: returns a BacktestEngine subclass that streams into _state."""

    class StreamingEngine(base_engine_cls):
        def _process_tick(self, tick):
            super()._process_tick(tick)
            _state.processed_ticks += 1

            # Emit forecast records added this tick (they were appended by super)
            # We compare against the cursor we maintain externally.
            # forecast_records grows by ≤ N_active_markets per tick.
            n = len(self.forecast_records)
            cursor = getattr(self, "_fc_cursor", 0)
            for rec in self.forecast_records[cursor:]:
                _state.forecast_rows.append({
                    "ts": rec.timestamp,
                    "slug": rec.market_slug,
                    "yes_price": rec.market_yes_price,
                    "p_model": rec.model_forecast,
                })
            self._fc_cursor = n

            # Emit fills added this tick.
            nf = len(self.all_fills)
            fc = getattr(self, "_fill_cursor", 0)
            for fill in self.all_fills[fc:]:
                _state.fill_rows.append({
                    "ts": fill.timestamp,
                    "slug": fill.market_slug,
                    "token": fill.token.value,
                    "side": fill.side.value,
                    "size": fill.size,
                    "avg_price": fill.avg_price,
                })
            self._fill_cursor = nf

            # Emit the latest portfolio snapshot every tick.
            if self.snapshots:
                snap = self.snapshots[-1]
                _state.pnl_rows.append({
                    "ts": snap.timestamp,
                    "total_value": snap.total_value,
                    "realized_pnl": snap.realized_pnl,
                    "unrealized_pnl": snap.unrealized_pnl,
                })

    return StreamingEngine


# ── Background worker ─────────────────────────────────────────────────────────

def _worker(data_dir: str, hours: float | None, _state: BacktestState):
    try:
        from backtester.data_loader import build_timeline
        from backtester.engine import BacktestEngine
        from trader import MyStrategy

        data = build_timeline(
            data_dir=Path(data_dir),
            intervals=["5m", "15m", "hourly"],
            hours=hours,
            assets=["BTC"],  # strategy targets BTC only; speeds up load
        )

        if not data.timeline:
            _state.error = "No data found in the specified directory."
            _state.done = True
            _state.running = False
            return

        _state.total_ticks = len(data.timeline)

        StreamingEngine = _make_streaming_engine(BacktestEngine, _state)
        strategy = MyStrategy()
        engine = StreamingEngine(
            data=data,
            strategy=strategy,
            starting_cash=10_000.0,
            snapshot_interval=1,
        )
        result = engine.run()

        _state.summary = {
            "total_pnl": result.total_pnl,
            "final_value": result.final_portfolio_value,
            "total_trades": result.total_trades,
            "total_settlements": result.total_settlements,
            "total_rejected": result.total_rejected,
            "elapsed_s": result.elapsed_seconds,
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
        }

    except Exception as exc:
        import traceback
        _state.error = traceback.format_exc()

    finally:
        _state.done = True
        _state.running = False


def run_backtest_async(data_dir: str, _state: BacktestState, hours: float | None = None):
    """Launch backtest in background thread. Resets state first."""
    _state.reset()
    _state.running = True
    t = threading.Thread(target=_worker, args=(data_dir, hours, _state), daemon=True)
    t.start()
