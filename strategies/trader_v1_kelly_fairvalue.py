from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    MarketView,
    Order,
    OrderBookSnapshot,
    PositionView,
    Settlement,
    Side,
    Token,
)

# How many ticks of global BTC data to retain (1 hour at 1s cadence).
_GLOBAL_HISTORY = 3600

# Per-market capacity matches the market's full lifetime in seconds.
_INTERVAL_TICKS: dict[str, int] = {"5m": 300, "15m": 900, "hourly": 3600}

# Minimum ticks of per-market history required before features are computed.
_MIN_WARMUP = 30


# ──────────────────────────────────────────────────────────────────────────────
# FeatureContext
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureContext:
    market_hist:   list        # list[MarketView]; [-1] = current tick
    btc_mids:      np.ndarray  # global BTC Binance mid-price history; [-1] = current
    btc_spreads:   np.ndarray  # global BTC Binance spread history
    chainlink_btc: np.ndarray  # Chainlink oracle BTC price history
    timestamps:    np.ndarray  # unix second timestamps (aligns with btc_mids)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers (module-level; not features)
# ──────────────────────────────────────────────────────────────────────────────

def _ema(series: np.ndarray, span: int) -> float:
    """Compute exponentially weighted mean (last value) of a series."""
    if len(series) == 0:
        return 0.0
    alpha = 2.0 / (span + 1)
    val = float(series[0])
    for x in series[1:]:
        val = alpha * float(x) + (1.0 - alpha) * val
    return val


def _btc_return(mids: np.ndarray, lookback: int) -> float:
    """Return the fractional BTC price change over `lookback` ticks.

    Falls back to whatever history is available when the full window is not yet
    filled; returns 0.0 if fewer than 2 ticks exist.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group A — BTC price momentum)
# ──────────────────────────────────────────────────────────────────────────────

def feat_btc_return_30s(ctx: FeatureContext) -> float:
    """Fractional BTC mid-price change over the last 30 seconds."""
    return _btc_return(ctx.btc_mids, 30)


def feat_btc_return_60s(ctx: FeatureContext) -> float:
    """Fractional BTC mid-price change over the last 60 seconds."""
    return _btc_return(ctx.btc_mids, 60)


def feat_btc_return_300s(ctx: FeatureContext) -> float:
    """Fractional BTC mid-price change over the last 300 seconds (5 min).

    This matches the prediction horizon, so it encodes whether the current
    trend is continuing or has already run its course.
    """
    return _btc_return(ctx.btc_mids, 300)


def feat_btc_ema_cross_20_120(ctx: FeatureContext) -> float:
    """(EMA20 - EMA120) / EMA120 of the BTC mid-price series.

    Positive → short-term trend above long-term trend (bullish).
    Falls back gracefully: uses all available history regardless of window size.
    """
    mids = ctx.btc_mids
    if len(mids) == 0:
        return 0.0
    ema20  = _ema(mids, 20)
    ema120 = _ema(mids, 120)
    if ema120 == 0.0:
        return 0.0
    return float((ema20 - ema120) / ema120)


def feat_btc_realized_vol_60s(ctx: FeatureContext) -> float:
    """Standard deviation of 1-second BTC returns over the last 60 seconds.

    High volatility reduces directional predictability and increases the
    variance of logistic regression's probability output.
    """
    mids = ctx.btc_mids
    n = min(len(mids), 61)
    if n < 2:
        return 0.0
    window = mids[-n:]
    rets = np.diff(window) / np.where(window[:-1] != 0, window[:-1], 1.0)
    return float(np.std(rets))


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group B — Binance BTC spread)
# ──────────────────────────────────────────────────────────────────────────────

def feat_btc_spread_norm(ctx: FeatureContext) -> float:
    """Binance BTC bid-ask spread divided by mid-price (relative spread).

    Wide relative spread signals elevated uncertainty or low liquidity.
    """
    mids = ctx.btc_mids
    spreads = ctx.btc_spreads
    if len(mids) == 0 or len(spreads) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    return float(spreads[-1] / mid)


def feat_btc_spread_delta_30s(ctx: FeatureContext) -> float:
    """Change in Binance BTC spread over the last 30 seconds, normalized by mid.

    Positive → spread widening (uncertainty increasing); this often precedes a
    directional move.
    """
    spreads = ctx.btc_spreads
    mids = ctx.btc_mids
    if len(spreads) < 2 or len(mids) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    lookback = min(30, len(spreads) - 1)
    return float((spreads[-1] - spreads[-1 - lookback]) / mid)


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group C — Polymarket implied probability)
# ──────────────────────────────────────────────────────────────────────────────

def feat_yes_price(ctx: FeatureContext) -> float:
    """Raw Polymarket YES mid-price — the market's consensus P(BTC goes up).

    This is the strongest single baseline feature; other features should add
    orthogonal signal relative to this.
    """
    return float(ctx.market_hist[-1].yes_price)


def feat_yes_price_mom_30s(ctx: FeatureContext) -> float:
    """Change in Polymarket YES price over the last 30 seconds.

    Rising yes_price → smart money is buying up → bullish.
    """
    h = ctx.market_hist
    lookback = min(30, len(h) - 1)
    if lookback < 1:
        return 0.0
    return float(h[-1].yes_price - h[-1 - lookback].yes_price)


def feat_yes_price_mom_60s(ctx: FeatureContext) -> float:
    """Change in Polymarket YES price over the last 60 seconds."""
    h = ctx.market_hist
    lookback = min(60, len(h) - 1)
    if lookback < 1:
        return 0.0
    return float(h[-1].yes_price - h[-1 - lookback].yes_price)


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group D — Polymarket order book imbalance)
# ──────────────────────────────────────────────────────────────────────────────

def feat_yes_obi_level1(ctx: FeatureContext) -> float:
    """Best-level order book imbalance on the YES book.

    OBI = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size).
    Range [-1, 1]. Positive → more resting buy interest → upward price pressure.
    This is the most widely validated single-tick microstructure predictor.
    """
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
    """Multi-level order book imbalance on the YES book, decayed by price distance.

    Each level's bid/ask volume is weighted by exp(-lambda * |price - mid|),
    so orders near the inside carry more weight than far-from-money resting
    orders. Extends the level-1 OBI across the full depth.
    """
    book = ctx.market_hist[-1].yes_book
    mid = book.mid
    if mid <= 0.0:
        return 0.0

    lam = 10.0  # decay rate in probability units; 0.1 distance → weight ≈ 0.37
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
    """Signed deviation of the YES microprice from the quoted mid, normalized.

    Microprice = (best_ask * best_bid_size + best_bid * best_ask_size)
                 / (best_bid_size + best_ask_size).

    Positive → microprice > mid → book is skewed toward buying → bullish.
    This captures the same information as OBI but expressed as a price level
    rather than a volume ratio, and can be decorrelated from feat_yes_obi_level1.
    """
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
    """Log ratio of total YES bid depth to total YES ask depth.

    log(total_bid_size / total_ask_size). Zero = balanced. Positive = more
    inventory resting on the buy side. Uses log to handle large asymmetries
    symmetrically and keep the feature roughly Gaussian.
    """
    book = ctx.market_hist[-1].yes_book
    total_bid = book.total_bid_size
    total_ask = book.total_ask_size
    if total_ask < 1e-9 or total_bid < 1e-9:
        return 0.0
    return float(np.log(total_bid / total_ask))


def feat_yes_near_book_imbalance(ctx: FeatureContext) -> float:
    """Order book imbalance restricted to levels within 0.05 of the YES mid.

    Filters out far-from-money orders that are unlikely to fill, isolating
    the near-money depth that most directly determines short-term price pressure.
    Complements feat_yes_obi_multilevel by using a hard threshold instead of
    exponential decay.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group E — Polymarket spread / uncertainty)
# ──────────────────────────────────────────────────────────────────────────────

def feat_yes_spread_norm(ctx: FeatureContext) -> float:
    """YES bid-ask spread divided by YES mid-price (relative spread).

    Wide relative spread → market makers uncertain → do not overtrade.
    Useful as a feature because high-uncertainty markets are harder to predict
    and the logistic regression should weight them lower.
    """
    m = ctx.market_hist[-1]
    price = m.yes_price
    if price <= 0.0:
        return 0.0
    return float((m.yes_ask - m.yes_bid) / price)


def feat_yes_spread_delta_30s(ctx: FeatureContext) -> float:
    """Change in YES bid-ask spread over 30 seconds, normalized by current mid.

    Positive → spread is widening → uncertainty increasing, often before a move.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group F — Market timing)
# ──────────────────────────────────────────────────────────────────────────────

def feat_time_remaining_frac(ctx: FeatureContext) -> float:
    """Fraction of the market's duration that remains (1.0 at open, 0.0 at close).

    Near expiry the YES price rapidly converges to 0 or 1, so predictions
    made with little time left are less actionable and riskier.
    """
    return float(ctx.market_hist[-1].time_remaining_frac)


def feat_log_time_remaining(ctx: FeatureContext) -> float:
    """Natural log of seconds remaining until settlement.

    Captures non-linear time decay: early in the market there is lots of room
    for the price to move; near expiry it collapses to the terminal value fast.
    """
    t = ctx.market_hist[-1].time_remaining_s
    return float(np.log(max(float(t), 1.0)))


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group G — Cross-signal / divergence)
# ──────────────────────────────────────────────────────────────────────────────

def feat_btc_vs_poly_divergence(ctx: FeatureContext) -> float:
    """Polymarket implied signal minus BTC momentum signal, both in [-1, 1].

    BTC momentum: tanh(500 * ret_60s) maps ±0.2% return → ±0.76.
    Poly signal: 2 * yes_price - 1 maps [0, 1] → [-1, 1].
    Positive → poly is more bullish than recent BTC price action suggests.
    Negative → BTC is moving up but poly hasn't repriced yet (lead-lag alpha).
    """
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
    """(Chainlink oracle price - Binance mid) / Binance mid.

    Chainlink updates on a slower heartbeat than Binance spot, so when spot
    has moved recently the oracle lags. A positive value means the oracle
    hasn't caught up to a downward Binance move yet — the next Chainlink update
    will be lower, which is bearish for YES settlement.
    """
    cl = ctx.chainlink_btc
    mids = ctx.btc_mids
    if len(cl) == 0 or len(mids) == 0:
        return 0.0
    cl_last  = float(cl[-1])
    mid_last = float(mids[-1])
    if mid_last <= 0.0 or cl_last <= 0.0:
        return 0.0
    return (cl_last - mid_last) / mid_last


# ──────────────────────────────────────────────────────────────────────────────
# Model weights
# ──────────────────────────────────────────────────────────────────────────────
# Dev mode:         loads from weights/current.npz (updated by autofeature.py).
# Competition mode: file absent → uses baked fallback below.
# Run `python bake_weights.py` before submission to sync the baked values.

_weights_file = Path(__file__).parent / "weights" / "current.npz"
if _weights_file.exists():
    _wf      = np.load(_weights_file)
    _W_COEF  = _wf["coef"]
    _W_BIAS  = float(_wf["intercept"][0])
    _W_MEAN  = _wf["scaler_mean"]
    _W_SCALE = _wf["scaler_scale"]
else:
    # ── BEGIN BAKED WEIGHTS (generated by bake_weights.py — do not edit) ──
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
    # ── END BAKED WEIGHTS ──

# Trading hyperparameters
_MIN_EDGE   = 0.04   # minimum |p_yes - 0.50| required to open or hold a position
_HALF_KELLY = 0.50   # fraction of full Kelly to wager (conservative)
_MAX_SHARES = 500    # hard position cap per token per market
_T_LO       = 0.15   # entry gate: no new buys when time_remaining_frac < _T_LO
_T_HI       = 0.85   # entry gate: no new buys when time_remaining_frac > _T_HI


def _predict_yes_prob(ctx: FeatureContext) -> float:
    """Return P(YES) from the hardcoded logistic regression model."""
    x = np.array([fn(ctx) for fn in FEATURE_REGISTRY], dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x_sc = (x - _W_MEAN) / _W_SCALE
    logit = float(_W_COEF @ x_sc) + _W_BIAS
    logit = max(-20.0, min(20.0, logit))
    return 1.0 / (1.0 + np.exp(-logit))


def _kelly_shares(p_win: float, ask: float, cash: float) -> int:
    """Half-Kelly integer share count for a binary contract priced at `ask`."""
    if ask <= 0.0 or ask >= 1.0 or p_win <= ask:
        return 0
    full_kelly_frac = (p_win - ask) / (1.0 - ask)
    target_cash = _HALF_KELLY * full_kelly_frac * cash
    return min(_MAX_SHARES, int(target_cash / ask))


# ──────────────────────────────────────────────────────────────────────────────
# Feature registry
# ──────────────────────────────────────────────────────────────────────────────
# Order is fixed: the logistic regression weight vector is positional.
# Never reorder existing entries. Always append new features at the end.

FEATURE_REGISTRY = [
    # Group A — BTC price momentum
    feat_btc_return_30s,           # 0
    feat_btc_return_60s,           # 1
    feat_btc_return_300s,          # 2
    feat_btc_ema_cross_20_120,     # 3
    feat_btc_realized_vol_60s,     # 4
    # Group B — Binance spread
    feat_btc_spread_norm,          # 5
    feat_btc_spread_delta_30s,     # 6
    # Group C — Polymarket implied probability
    feat_yes_price,                # 7
    feat_yes_price_mom_30s,        # 8
    feat_yes_price_mom_60s,        # 9
    # Group D — Polymarket order book imbalance
    feat_yes_obi_level1,           # 10
    feat_yes_obi_multilevel,       # 11
    feat_yes_microprice_dev,       # 12
    feat_yes_depth_ratio,          # 13
    feat_yes_near_book_imbalance,  # 14
    # Group E — Polymarket spread / uncertainty
    feat_yes_spread_norm,          # 15
    feat_yes_spread_delta_30s,     # 16
    # Group F — Market timing
    feat_time_remaining_frac,      # 17
    feat_log_time_remaining,       # 18
    # Group G — Cross-signal
    feat_btc_vs_poly_divergence,   # 19
    feat_chainlink_vs_binance,     # 20
]

# ── Accepted features (appended by autofeature.py — do not edit manually) ────
# ── END ACCEPTED FEATURES ────────────────────────────────────────────────────

# ── Candidate injection (autofeature.py only — never commit _candidate.py) ───
try:
    from _candidate import CANDIDATE_FEATURES as _cands
    FEATURE_REGISTRY = FEATURE_REGISTRY + _cands
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────

class MyStrategy(BaseStrategy):

    def __init__(self) -> None:
        # ── Snapshot-wide history ─────────────────────────────────────────
        self.btc_mids:      deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.btc_spreads:   deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.chainlink_btc: deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.timestamps:    deque[int]   = deque(maxlen=_GLOBAL_HISTORY)

        # ── Per-market snapshot history ───────────────────────────────────
        # Keyed by market_slug. Each value is a deque of MarketView objects,
        # one per tick that market was active. Access any field via list comp:
        #   h = list(self.market_history[slug])
        #   yes_prices = np.array([m.yes_price for m in h])
        self.market_history: dict[str, deque[MarketView]] = {}
        self.closed_markets: set[str] = set()


    # ─────────────────────────────────────────────────────────────────────────
    #  on_tick
    # ─────────────────────────────────────────────────────────────────────────

    def on_tick(self, state: MarketState) -> list[Order]:
        # 1. Record snapshot-wide data.
        self.btc_mids.append(state.btc_mid)
        self.btc_spreads.append(state.btc_spread)
        self.chainlink_btc.append(state.chainlink_btc)
        self.timestamps.append(state.timestamp)

        # 2. Record per-market data.
        for slug, market in state.markets.items():
            if slug not in self.market_history:
                capacity = _INTERVAL_TICKS.get(market.interval, 3600)
                self.market_history[slug] = deque(maxlen=capacity)
            self.market_history[slug].append(market)

        # 3. Trading logic: BTC 5m markets only, after warmup.
        orders: list[Order] = []
        for slug, market in state.markets.items():
            if not slug.startswith("btc-") or market.interval != "5m":
                continue
            if len(self.market_history[slug]) < _MIN_WARMUP:
                continue

            ctx = self._build_context(slug)
            p = _predict_yes_prob(ctx)
            pos = state.positions.get(slug)
            yes_sh = pos.yes_shares if pos else 0.0
            no_sh  = pos.no_shares  if pos else 0.0

            # ── EXIT ─────────────────────────────────────────────────────────
            if yes_sh > 0.5:
                # Fair-value exit: market pays more than model thinks YES is worth.
                # Stop-loss: model has flipped to favour NO.
                if market.yes_bid > p or p < 0.5:
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_sh, limit_price=None,
                    ))
                    self.closed_markets.add(slug)
                continue

            if no_sh > 0.5:
                # Fair-value exit: market pays more than model thinks NO is worth.
                # Stop-loss: model has flipped to favour YES.
                if market.no_bid > 1.0 - p or p > 0.5:
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_sh, limit_price=None,
                    ))
                    self.closed_markets.add(slug)
                continue

            # ── ENTRY ─────────────────────────────────────────────────────────
            if slug in self.closed_markets:
                continue
            if not (_T_LO < market.time_remaining_frac < _T_HI):
                continue

            if p > 0.5 + _MIN_EDGE and market.yes_ask > 0:
                size = min(_kelly_shares(p, market.yes_ask, state.cash),
                           int(_MAX_SHARES - yes_sh))
                if size >= 1:
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.BUY,
                        size=float(size), limit_price=None,
                    ))
            elif p < 0.5 - _MIN_EDGE and market.no_ask > 0:
                size = min(_kelly_shares(1.0 - p, market.no_ask, state.cash),
                           int(_MAX_SHARES - no_sh))
                if size >= 1:
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.BUY,
                        size=float(size), limit_price=None,
                    ))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_context(self, slug: str) -> FeatureContext:
        """Assemble a FeatureContext for the given market slug."""
        return FeatureContext(
            market_hist=list(self.market_history[slug]),
            btc_mids=np.array(self.btc_mids),
            btc_spreads=np.array(self.btc_spreads),
            chainlink_btc=np.array(self.chainlink_btc),
            timestamps=np.array(self.timestamps),
        )

    def _compute_feature_vector(self, ctx: FeatureContext) -> np.ndarray:
        """Evaluate every registered feature and return a 1-D float array."""
        return np.array([fn(ctx) for fn in FEATURE_REGISTRY], dtype=np.float64)

    def _get_market_series(self, slug: str, field: str) -> np.ndarray:
        """Extract a named scalar field from a market's snapshot history."""
        h = self.market_history.get(slug)
        if not h:
            return np.array([])
        return np.array([getattr(m, field) for m in h])

    # ─────────────────────────────────────────────────────────────────────────
    #  on_fill
    # ─────────────────────────────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    #  on_settlement
    # ─────────────────────────────────────────────────────────────────────────

    def on_settlement(self, settlement: Settlement) -> None:
        slug = settlement.market_slug
        self.market_history.pop(slug, None)
        self.closed_markets.discard(slug)
