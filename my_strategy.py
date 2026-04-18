"""
Logistic regression strategy for BTC Polymarket prediction markets.

Deploy:
    cp /Users/devangpant/logreg-strategy/my_strategy.py /Users/devangpant/DATAHACKS2026/
    cp /Users/devangpant/logreg-strategy/model_weights.json /Users/devangpant/DATAHACKS2026/
    cd /Users/devangpant/DATAHACKS2026
    python run_backtest.py my_strategy.py --assets BTC

Or set WEIGHTS_PATH env var to point to the weights file from any location.
"""

import json
import math
import os
from collections import deque
from pathlib import Path

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)

_WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH",
    str(Path(__file__).parent / "model_weights.json"),
)

THRESHOLD    = float(os.environ.get("THRESHOLD", "0.03"))
TRADE_SIZE   = float(os.environ.get("TRADE_SIZE", "10"))
MAX_POSITION = float(os.environ.get("MAX_POSITION", "500"))  # shares per side per market
LIMIT_PAD    = float(os.environ.get("LIMIT_PAD", "0.01"))    # pad limit price for 1s execution lag
MIN_TIME_FRAC = 0.20  # skip markets in their final 20%


def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _predict(features: list[float], weights: list[float], bias: float) -> float:
    z = sum(w * f for w, f in zip(weights, features)) + bias
    return _sigmoid(z)


class LogRegStrategy(BaseStrategy):
    def __init__(self) -> None:
        with open(_WEIGHTS_PATH) as f:
            d = json.load(f)
        self._weights: list[float] = d["weights"]
        self._bias: float = d["bias"]
        self._feature_names: list[str] = d["features"]
        self._means: list[float] = d["means"]
        self._stds: list[float] = [s if s > 1e-9 else 1.0 for s in d["stds"]]

        # Extract N from saved feature name "btc_momentum_N"
        self._N = 60
        for name in self._feature_names:
            if name.startswith("btc_momentum_"):
                self._N = int(name.split("_")[-1])
                break

        # Rolling buffers (N+1 values = window of N ticks back)
        self._btc_buf = deque(maxlen=self._N + 1)
        self._cl_buf  = deque(maxlen=self._N + 1)

        # BTC and Chainlink price at first tick each market was seen
        self._btc_open: dict[str, float] = {}
        self._cl_open:  dict[str, float] = {}

        # Cumulative shares held per (market, side) — capped at MAX_POSITION
        self._positions: dict[tuple[str, Token], float] = {}

        # Latest forecasts (returned via get_forecasts for Brier scoring)
        self._forecasts: dict[str, float] = {}

    def _normalize(self, raw: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(raw, self._means, self._stds)]

    def _build_features(self, state: MarketState, slug: str) -> list[float]:
        market = state.markets[slug]

        # Global features
        chainlink_lag = state.btc_mid - state.chainlink_btc
        btc_spread    = state.btc_spread

        # Momentum: difference between current and N-ticks-ago value in buffer
        btc_momentum = (
            self._btc_buf[-1] - self._btc_buf[0]
            if len(self._btc_buf) == self._N + 1 else 0.0
        )
        cl_momentum = (
            self._cl_buf[-1] - self._cl_buf[0]
            if len(self._cl_buf) == self._N + 1 else 0.0
        )

        # Per-market features
        yes_ask   = market.yes_ask
        total_bid = market.yes_book.total_bid_size
        total_ask        = market.yes_book.total_ask_size
        book_imbalance   = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)
        time_remaining   = market.time_remaining_frac

        # Build in the exact order stored in weights JSON
        # Drift from market open — record on first tick, compute delta every tick
        if slug not in self._btc_open and state.btc_mid > 0:
            self._btc_open[slug] = state.btc_mid
        if slug not in self._cl_open and state.chainlink_btc > 0:
            self._cl_open[slug] = state.chainlink_btc
        btc_drift = state.btc_mid      - self._btc_open.get(slug, state.btc_mid)
        cl_drift  = state.chainlink_btc - self._cl_open.get(slug, state.chainlink_btc)

        total_bid = market.yes_book.total_bid_size
        total_ask = market.yes_book.total_ask_size
        book_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

        feat_lookup = {
            "chainlink_lag":                 chainlink_lag,
            "btc_spread":                    btc_spread,
            f"btc_momentum_{self._N}":       btc_momentum,
            f"chainlink_momentum_{self._N}": cl_momentum,
            "btc_drift_from_open":           btc_drift,
            "chainlink_drift_from_open":     cl_drift,
            "time_remaining_frac":           time_remaining,
            "yes_ask":                       yes_ask,
            "book_imbalance":                book_imbalance,
        }
        return [feat_lookup.get(name, 0.0) for name in self._feature_names]

    def on_tick(self, state: MarketState) -> list[Order]:
        # Advance rolling buffers (only when feed is live)
        if state.btc_mid > 0:
            self._btc_buf.append(state.btc_mid)
        if state.chainlink_btc > 0:
            self._cl_buf.append(state.chainlink_btc)

        orders: list[Order] = []
        self._forecasts = {}

        for slug, market in state.markets.items():
            if not (slug.startswith("btc-") or slug.startswith("bitcoin-")):
                continue
            if market.time_remaining_frac < MIN_TIME_FRAC:
                continue
            if market.yes_ask <= 0 and market.no_ask <= 0:
                continue

            raw  = self._build_features(state, slug)
            norm = self._normalize(raw)
            p_yes = _predict(norm, self._weights, self._bias)
            self._forecasts[slug] = p_yes

            yes_ask = market.yes_ask
            yes_bid = market.yes_bid
            no_ask  = market.no_ask

            if yes_ask > 0 and p_yes > yes_ask + THRESHOLD:
                key = (slug, Token.YES)
                held = self._positions.get(key, 0.0)
                room = MAX_POSITION - held
                size = min(TRADE_SIZE, room)
                if size > 0:
                    cost = size * yes_ask
                    if state.cash >= cost:
                        orders.append(Order(
                            market_slug=slug,
                            token=Token.YES,
                            side=Side.BUY,
                            size=size,
                            limit_price=min(yes_ask + LIMIT_PAD, 0.99),
                        ))
                        self._positions[key] = held + size

            elif yes_bid > 0 and p_yes < yes_bid - THRESHOLD:
                key = (slug, Token.NO)
                held = self._positions.get(key, 0.0)
                room = MAX_POSITION - held
                size = min(TRADE_SIZE, room)
                if size > 0 and no_ask > 0:
                    cost = size * no_ask
                    if state.cash >= cost:
                        orders.append(Order(
                            market_slug=slug,
                            token=Token.NO,
                            side=Side.BUY,
                            size=size,
                            limit_price=min(no_ask + LIMIT_PAD, 0.99),
                        ))
                        self._positions[key] = held + size

        return orders

    def get_forecasts(self, state: MarketState) -> dict[str, float]:
        return dict(self._forecasts)

    def on_settlement(self, settlement: Settlement) -> None:
        slug = settlement.market_slug
        self._positions.pop((slug, Token.YES), None)
        self._positions.pop((slug, Token.NO), None)
        self._forecasts.pop(slug, None)
        self._btc_open.pop(slug, None)
        self._cl_open.pop(slug, None)
