"""
Hybrid strategy: OUR model (trader.py 22-feature logistic regression)
                 + BALL (devang) execution logic (fixed size, limit orders, cash check).

Purpose: isolate whether devang's higher backtest PnL is due to its execution
logic or its model, by controlling for execution while swapping in our model.
"""

import os
from collections import deque
from pathlib import Path

import numpy as np

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    MarketView,
    Order,
    Settlement,
    Side,
    Token,
)

# ── Execution hyperparameters (devang/ball execution logic) ──────────────────
THRESHOLD    = float(os.environ.get("THRESHOLD", "0.03"))
TRADE_SIZE   = float(os.environ.get("TRADE_SIZE", "10"))
MAX_POSITION = float(os.environ.get("MAX_POSITION", "500"))
LIMIT_PAD    = float(os.environ.get("LIMIT_PAD", "0.01"))
MIN_TIME_FRAC = 0.20

# ── Model constants (from trader.py) ─────────────────────────────────────────
_GLOBAL_HISTORY = 3600
_INTERVAL_TICKS: dict[str, int] = {"5m": 300, "15m": 900, "hourly": 3600}
_MIN_WARMUP = 30

from dataclasses import dataclass

@dataclass
class FeatureContext:
    market_hist:   list
    btc_mids:      np.ndarray
    btc_spreads:   np.ndarray
    chainlink_btc: np.ndarray
    timestamps:    np.ndarray


# ── Feature helpers (verbatim from trader.py) ─────────────────────────────────

def _ema(series: np.ndarray, span: int) -> float:
    if len(series) == 0:
        return 0.0
    alpha = 2.0 / (span + 1)
    val = float(series[0])
    for x in series[1:]:
        val = alpha * float(x) + (1.0 - alpha) * val
    return val


def _btc_return(mids: np.ndarray, lookback: int) -> float:
    n = len(mids)
    if n < 2:
        return 0.0
    actual = min(lookback, n - 1)
    prev = mids[-1 - actual]
    if prev == 0.0:
        return 0.0
    return float((mids[-1] - prev) / prev)


def _yes_spread(m: MarketView) -> float:
    return float(m.yes_ask - m.yes_bid)


def feat_btc_return_30s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.btc_mids, 30)

def feat_btc_return_60s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.btc_mids, 60)

def feat_btc_return_300s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.btc_mids, 300)

def feat_btc_ema_cross_20_120(ctx: FeatureContext) -> float:
    mids = ctx.btc_mids
    if len(mids) == 0:
        return 0.0
    ema20  = _ema(mids, 20)
    ema120 = _ema(mids, 120)
    if ema120 == 0.0:
        return 0.0
    return float((ema20 - ema120) / ema120)

def feat_btc_realized_vol_60s(ctx: FeatureContext) -> float:
    mids = ctx.btc_mids
    n = min(len(mids), 61)
    if n < 2:
        return 0.0
    window = mids[-n:]
    rets = np.diff(window) / np.where(window[:-1] != 0, window[:-1], 1.0)
    return float(np.std(rets))

def feat_btc_spread_norm(ctx: FeatureContext) -> float:
    mids = ctx.btc_mids
    spreads = ctx.btc_spreads
    if len(mids) == 0 or len(spreads) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    return float(spreads[-1] / mid)

def feat_btc_spread_delta_30s(ctx: FeatureContext) -> float:
    spreads = ctx.btc_spreads
    mids = ctx.btc_mids
    if len(spreads) < 2 or len(mids) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    lookback = min(30, len(spreads) - 1)
    return float((spreads[-1] - spreads[-1 - lookback]) / mid)

def feat_yes_price(ctx: FeatureContext) -> float:
    return float(ctx.market_hist[-1].yes_price)

def feat_yes_price_mom_30s(ctx: FeatureContext) -> float:
    h = ctx.market_hist
    lookback = min(30, len(h) - 1)
    if lookback < 1:
        return 0.0
    return float(h[-1].yes_price - h[-1 - lookback].yes_price)

def feat_yes_price_mom_60s(ctx: FeatureContext) -> float:
    h = ctx.market_hist
    lookback = min(60, len(h) - 1)
    if lookback < 1:
        return 0.0
    return float(h[-1].yes_price - h[-1 - lookback].yes_price)

def feat_yes_obi_level1(ctx: FeatureContext) -> float:
    book = ctx.market_hist[-1].yes_book
    if not book.bids or not book.asks:
        return 0.0
    b_sz = book.bids[0].size
    a_sz = book.asks[0].size
    denom = b_sz + a_sz
    if denom < 1e-9:
        return 0.0
    return float((b_sz - a_sz) / denom)

def feat_yes_obi_multilevel(ctx: FeatureContext) -> float:
    book = ctx.market_hist[-1].yes_book
    mid = book.mid
    if mid <= 0.0:
        return 0.0
    lam = 10.0
    bid_w = 0.0
    ask_w = 0.0
    for lvl in book.bids:
        w = float(np.exp(-lam * abs(lvl.price - mid)))
        bid_w += w * lvl.size
    for lvl in book.asks:
        w = float(np.exp(-lam * abs(lvl.price - mid)))
        ask_w += w * lvl.size
    denom = bid_w + ask_w
    if denom < 1e-9:
        return 0.0
    return float((bid_w - ask_w) / denom)

def feat_yes_microprice_dev(ctx: FeatureContext) -> float:
    book = ctx.market_hist[-1].yes_book
    mid = book.mid
    if mid <= 0.0 or not book.bids or not book.asks:
        return 0.0
    b_sz = book.bids[0].size
    a_sz = book.asks[0].size
    denom = b_sz + a_sz
    if denom < 1e-9:
        return 0.0
    microprice = (book.best_ask * b_sz + book.best_bid * a_sz) / denom
    return float((microprice - mid) / mid)

def feat_yes_depth_ratio(ctx: FeatureContext) -> float:
    book = ctx.market_hist[-1].yes_book
    total_bid = book.total_bid_size
    total_ask = book.total_ask_size
    if total_ask < 1e-9 or total_bid < 1e-9:
        return 0.0
    return float(np.log(total_bid / total_ask))

def feat_yes_near_book_imbalance(ctx: FeatureContext) -> float:
    book = ctx.market_hist[-1].yes_book
    mid = book.mid
    if mid <= 0.0:
        return 0.0
    thr = 0.05
    near_bids = sum(lvl.size for lvl in book.bids if abs(lvl.price - mid) <= thr)
    near_asks = sum(lvl.size for lvl in book.asks if abs(lvl.price - mid) <= thr)
    denom = near_bids + near_asks
    if denom < 1e-9:
        return 0.0
    return float((near_bids - near_asks) / denom)

def feat_yes_spread_norm(ctx: FeatureContext) -> float:
    m = ctx.market_hist[-1]
    price = m.yes_price
    if price <= 0.0:
        return 0.0
    return float((m.yes_ask - m.yes_bid) / price)

def feat_yes_spread_delta_30s(ctx: FeatureContext) -> float:
    h = ctx.market_hist
    mid = h[-1].yes_price
    if mid <= 0.0:
        return 0.0
    lookback = min(30, len(h) - 1)
    if lookback < 1:
        return 0.0
    curr = _yes_spread(h[-1])
    prev = _yes_spread(h[-1 - lookback])
    return float((curr - prev) / mid)

def feat_time_remaining_frac(ctx: FeatureContext) -> float:
    return float(ctx.market_hist[-1].time_remaining_frac)

def feat_log_time_remaining(ctx: FeatureContext) -> float:
    t = ctx.market_hist[-1].time_remaining_s
    return float(np.log(max(float(t), 1.0)))

def feat_btc_vs_poly_divergence(ctx: FeatureContext) -> float:
    mids = ctx.btc_mids
    if len(mids) < 2:
        return 0.0
    lookback = min(60, len(mids) - 1)
    prev = mids[-1 - lookback]
    ret = float((mids[-1] - prev) / prev) if prev != 0.0 else 0.0
    btc_signal  = float(np.tanh(ret * 500.0))
    poly_signal = 2.0 * float(ctx.market_hist[-1].yes_price) - 1.0
    return poly_signal - btc_signal

def feat_chainlink_vs_binance(ctx: FeatureContext) -> float:
    cl = ctx.chainlink_btc
    mids = ctx.btc_mids
    if len(cl) == 0 or len(mids) == 0:
        return 0.0
    cl_last  = float(cl[-1])
    mid_last = float(mids[-1])
    if mid_last <= 0.0 or cl_last <= 0.0:
        return 0.0
    return (cl_last - mid_last) / mid_last

def feat_btc_trend_consistency(ctx: FeatureContext) -> float:
    mids = ctx.btc_mids
    if len(mids) < 62:
        return 0.0
    ref = mids[-31]
    if ref == 0:
        return 0.0
    current_ret = (mids[-1] - ref) / ref
    if current_ret == 0:
        return 0.0
    direction = 1 if current_ret > 0 else -1
    diffs = np.diff(mids[-61:])
    same_dir = np.sum(np.sign(diffs) == direction)
    return float(same_dir / len(diffs))


FEATURE_REGISTRY = [
    feat_btc_return_30s,
    feat_btc_return_60s,
    feat_btc_return_300s,
    feat_btc_ema_cross_20_120,
    feat_btc_realized_vol_60s,
    feat_btc_spread_norm,
    feat_btc_spread_delta_30s,
    feat_yes_price,
    feat_yes_price_mom_30s,
    feat_yes_price_mom_60s,
    feat_yes_obi_level1,
    feat_yes_obi_multilevel,
    feat_yes_microprice_dev,
    feat_yes_depth_ratio,
    feat_yes_near_book_imbalance,
    feat_yes_spread_norm,
    feat_yes_spread_delta_30s,
    feat_time_remaining_frac,
    feat_log_time_remaining,
    feat_btc_vs_poly_divergence,
    feat_chainlink_vs_binance,
    feat_btc_trend_consistency,
]

try:
    from _candidate import CANDIDATE_FEATURES as _cands
    FEATURE_REGISTRY = FEATURE_REGISTRY + _cands
except ImportError:
    pass

# ── Model weights (from trader.py) ───────────────────────────────────────────
_weights_file = Path(__file__).parent / "weights" / "current.npz"
if _weights_file.exists():
    _wf      = np.load(_weights_file)
    _W_COEF  = _wf["coef"]
    _W_BIAS  = float(_wf["intercept"][0])
    _W_MEAN  = _wf["scaler_mean"]
    _W_SCALE = _wf["scaler_scale"]
else:
    _W_COEF = np.array([
         0.07487681, -0.28664297, -0.16029155,  0.14720130, -0.07334121,
         0.18334326, -0.10993217,  2.14291915, -0.05242122, -0.16313906,
         0.08220650,  0.10749001, -0.14586830,  0.31657409, -0.10730961,
        -1.31421537, -0.00592992,  0.09930648, -0.21187793, -0.56199010,
        -0.12081161,
    ], dtype=np.float64)
    _W_BIAS  = -0.02062640
    _W_MEAN  = np.array([
         3.69316e-06,  1.16809e-05,  6.79982e-05,  1.18581e-05,  7.18809e-05,
         2.71477e-04, -1.99346e-06,  4.79128e-01, -1.91226e-02, -2.44702e-02,
        -2.52417e-03,  1.04581e-02,  3.16245e-03,  4.80257e-02,  8.39626e-03,
         4.41521e-02, -1.00450e-01,  4.22768e-01,  4.53135e+00, -3.55764e-02,
        -3.91807e-05,
    ], dtype=np.float64)
    _W_SCALE = np.array([
         4.60932e-04,  6.85190e-04,  1.55798e-03,  3.82506e-04,  6.30333e-05,
         1.68602e-04,  1.91981e-04,  3.00781e-01,  1.91424e-01,  2.38049e-01,
         5.85738e-01,  4.90306e-01,  4.68766e-02,  7.86748e-01,  5.14779e-01,
         6.64674e-01,  1.76321e+00,  2.56844e-01,  9.80149e-01,  5.23748e-01,
         2.68236e-04,
    ], dtype=np.float64)


def _predict_yes_prob(ctx: FeatureContext) -> float:
    x = np.array([fn(ctx) for fn in FEATURE_REGISTRY], dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    # Slice to match the baked weight vector length if candidate features are appended
    n = len(_W_COEF)
    x_sc = (x[:n] - _W_MEAN[:n]) / _W_SCALE[:n]
    logit = float(_W_COEF @ x_sc) + _W_BIAS
    logit = max(-20.0, min(20.0, logit))
    return 1.0 / (1.0 + np.exp(-logit))


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: our model, devang/ball execution
# ──────────────────────────────────────────────────────────────────────────────

class HybridStrategy(BaseStrategy):
    """
    Prediction: trader.py 22-feature logistic regression (_predict_yes_prob).
    Execution:  devang/ball logic — fixed trade size, limit orders, cash check.
    """

    def __init__(self) -> None:
        self.btc_mids:      deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.btc_spreads:   deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.chainlink_btc: deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.timestamps:    deque[int]   = deque(maxlen=_GLOBAL_HISTORY)
        self.market_history: dict[str, deque[MarketView]] = {}
        # (slug, Token) -> cumulative shares held (mirrors devang position tracking)
        self._positions: dict[tuple[str, Token], float] = {}

    def _build_context(self, slug: str) -> FeatureContext:
        return FeatureContext(
            market_hist=list(self.market_history[slug]),
            btc_mids=np.array(self.btc_mids),
            btc_spreads=np.array(self.btc_spreads),
            chainlink_btc=np.array(self.chainlink_btc),
            timestamps=np.array(self.timestamps),
        )

    def on_tick(self, state: MarketState) -> list[Order]:
        # Record global series
        self.btc_mids.append(state.btc_mid)
        self.btc_spreads.append(state.btc_spread)
        self.chainlink_btc.append(state.chainlink_btc)
        self.timestamps.append(state.timestamp)

        # Record per-market history
        for slug, market in state.markets.items():
            if slug not in self.market_history:
                capacity = _INTERVAL_TICKS.get(market.interval, 3600)
                self.market_history[slug] = deque(maxlen=capacity)
            self.market_history[slug].append(market)

        orders: list[Order] = []

        for slug, market in state.markets.items():
            # Devang: only btc-* markets
            if not (slug.startswith("btc-") or slug.startswith("bitcoin-")):
                continue
            # Devang: skip when near expiry
            if market.time_remaining_frac < MIN_TIME_FRAC:
                continue
            if market.yes_ask <= 0 and market.no_ask <= 0:
                continue
            # Our model: require warmup
            if len(self.market_history[slug]) < _MIN_WARMUP:
                continue

            ctx   = self._build_context(slug)
            p_yes = _predict_yes_prob(ctx)

            yes_ask = market.yes_ask
            yes_bid = market.yes_bid
            no_ask  = market.no_ask

            # ── BUY YES (devang execution) ────────────────────────────────────
            if yes_ask > 0 and p_yes > yes_ask + THRESHOLD:
                key  = (slug, Token.YES)
                held = self._positions.get(key, 0.0)
                size = min(TRADE_SIZE, MAX_POSITION - held)
                if size > 0 and state.cash >= size * yes_ask:
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.YES,
                        side=Side.BUY,
                        size=size,
                        limit_price=min(yes_ask + LIMIT_PAD, 0.99),
                    ))
                    self._positions[key] = held + size

            # ── BUY NO (devang execution) ─────────────────────────────────────
            elif yes_bid > 0 and p_yes < yes_bid - THRESHOLD:
                key  = (slug, Token.NO)
                held = self._positions.get(key, 0.0)
                size = min(TRADE_SIZE, MAX_POSITION - held)
                if size > 0 and no_ask > 0 and state.cash >= size * no_ask:
                    orders.append(Order(
                        market_slug=slug,
                        token=Token.NO,
                        side=Side.BUY,
                        size=size,
                        limit_price=min(no_ask + LIMIT_PAD, 0.99),
                    ))
                    self._positions[key] = held + size

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        slug = settlement.market_slug
        self._positions.pop((slug, Token.YES), None)
        self._positions.pop((slug, Token.NO), None)
        self.market_history.pop(slug, None)
