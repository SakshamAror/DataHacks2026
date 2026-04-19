"""
Global in-process state shared between the backtest thread and Dash callbacks.
All lists are append-only during a run; reads are safe without locking because
Python's GIL guarantees list.append and list reads are atomic at the C level.
"""

from dataclasses import dataclass, field


@dataclass
class BacktestState:
    running: bool = False
    done: bool = False
    error: str = ""

    # Live tick data emitted while backtest runs
    forecast_rows: list = field(default_factory=list)

    # Portfolio value snapshots: {ts, total_value, realized_pnl, unrealized_pnl}
    pnl_rows: list = field(default_factory=list)

    # Executed fills: {ts, slug, token, side, size, avg_price}
    fill_rows: list = field(default_factory=list)

    # Market settlements: {ts, slug, outcome, btc_open, btc_close}
    settlement_rows: list = field(default_factory=list)

    # Final summary populated when done=True
    summary: dict = field(default_factory=dict)

    total_ticks: int = 0
    processed_ticks: int = 0

    def reset(self):
        self.running = False
        self.done = False
        self.error = ""
        self.forecast_rows.clear()
        self.pnl_rows.clear()
        self.fill_rows.clear()
        self.settlement_rows.clear()
        self.summary.clear()
        self.total_ticks = 0
        self.processed_ticks = 0


state = BacktestState()
