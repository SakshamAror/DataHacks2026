from collections import deque
from dataclasses import dataclass, field
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
    # Primary asset mid/spread/chainlink — equals btc_* for BTC markets,
    # eth_* for ETH markets, sol_* for SOL markets. Use these in asset-agnostic features.
    primary_mids:      np.ndarray = field(default_factory=lambda: np.array([]))
    primary_spreads:   np.ndarray = field(default_factory=lambda: np.array([]))
    chainlink_primary: np.ndarray = field(default_factory=lambda: np.array([]))
    # Raw ETH and SOL feeds (available for cross-asset features)
    eth_mids:      np.ndarray = field(default_factory=lambda: np.array([]))
    eth_spreads:   np.ndarray = field(default_factory=lambda: np.array([]))
    chainlink_eth: np.ndarray = field(default_factory=lambda: np.array([]))
    sol_mids:      np.ndarray = field(default_factory=lambda: np.array([]))
    sol_spreads:   np.ndarray = field(default_factory=lambda: np.array([]))
    chainlink_sol: np.ndarray = field(default_factory=lambda: np.array([]))


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

    This matches the 5m prediction horizon, and serves as a sub-interval signal
    for 15m markets.
    """
    return _btc_return(ctx.btc_mids, 300)


def feat_btc_return_900s(ctx: FeatureContext) -> float:
    """Fractional BTC mid-price change over the last 900 seconds (15 min).

    Matches the 15m prediction horizon — encodes whether the trend over the full
    market duration is continuing or has reversed.
    """
    return _btc_return(ctx.btc_mids, 900)


def feat_btc_return_3600s(ctx: FeatureContext) -> float:
    """Fractional BTC mid-price change over the last 3600 seconds (1 hour).

    Full-duration signal for hourly markets — captures whether the macro BTC
    trend over the entire market lifetime favors YES or NO settlement.
    """
    return _btc_return(ctx.btc_mids, 3600)


# ──────────────────────────────────────────────────────────────────────────────
# Feature functions  (Group A' — Primary-asset price momentum)
# These mirror Group A but use ctx.primary_mids / primary_spreads / chainlink_primary.
# For BTC markets primary == btc; for ETH markets primary == eth; for SOL == sol.
# Use these in ETH/SOL registries so the model learns from the asset's own price.
# ──────────────────────────────────────────────────────────────────────────────

def feat_primary_return_30s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.primary_mids, 30)

def feat_primary_return_60s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.primary_mids, 60)

def feat_primary_return_300s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.primary_mids, 300)

def feat_primary_return_900s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.primary_mids, 900)

def feat_primary_return_3600s(ctx: FeatureContext) -> float:
    return _btc_return(ctx.primary_mids, 3600)

def feat_primary_ema_cross_20_120(ctx: FeatureContext) -> float:
    mids = ctx.primary_mids
    if len(mids) == 0:
        return 0.0
    ema20  = _ema(mids, 20)
    ema120 = _ema(mids, 120)
    if ema120 == 0.0:
        return 0.0
    return float((ema20 - ema120) / ema120)

def feat_primary_realized_vol_60s(ctx: FeatureContext) -> float:
    mids = ctx.primary_mids
    n = min(len(mids), 61)
    if n < 2:
        return 0.0
    window = mids[-n:]
    rets = np.diff(window) / np.where(window[:-1] != 0, window[:-1], 1.0)
    return float(np.std(rets))

def feat_primary_spread_norm(ctx: FeatureContext) -> float:
    mids    = ctx.primary_mids
    spreads = ctx.primary_spreads
    if len(mids) == 0 or len(spreads) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    return float(spreads[-1] / mid)

def feat_primary_spread_delta_30s(ctx: FeatureContext) -> float:
    spreads = ctx.primary_spreads
    mids    = ctx.primary_mids
    if len(spreads) < 2 or len(mids) == 0:
        return 0.0
    mid = mids[-1]
    if mid <= 0.0:
        return 0.0
    lookback = min(30, len(spreads) - 1)
    return float((spreads[-1] - spreads[-1 - lookback]) / mid)

def feat_primary_vs_poly_divergence(ctx: FeatureContext) -> float:
    mids = ctx.primary_mids
    if len(mids) < 2:
        return 0.0
    lookback = min(60, len(mids) - 1)
    prev = mids[-1 - lookback]
    ret = float((mids[-1] - prev) / prev) if prev != 0.0 else 0.0
    asset_signal = float(np.tanh(ret * 500.0))
    poly_signal  = 2.0 * float(ctx.market_hist[-1].yes_price) - 1.0
    return poly_signal - asset_signal

def feat_chainlink_primary_vs_binance(ctx: FeatureContext) -> float:
    cl   = ctx.chainlink_primary
    mids = ctx.primary_mids
    if len(cl) == 0 or len(mids) == 0:
        return 0.0
    cl_last  = float(cl[-1])
    mid_last = float(mids[-1])
    if mid_last <= 0.0 or cl_last <= 0.0:
        return 0.0
    return (cl_last - mid_last) / mid_last


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
# Model registry — one entry per (asset, interval) market type we trade
# ──────────────────────────────────────────────────────────────────────────────
# Dev mode:   loads from models/weights/{asset}_{interval}_current.npz
# Submit mode: file absent → uses baked fallback (run models/bake_weights.py first)

_WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"


def _load_weights(asset: str, interval: str) -> dict | None:
    p = _WEIGHTS_DIR / f"{asset}_{interval}_current.npz"
    if not p.exists():
        return None
    wf = np.load(p)
    intercept = wf["intercept"]
    return {
        "coef": wf["coef"],
        "bias": float(intercept.flat[0]),
        "mean": wf["scaler_mean"],
        "scale": wf["scaler_scale"],
    }


# ── BTC 5m baked fallback ─────────────────────────────────────────────────────
# <<BAKED_WEIGHTS_BTC_5M_START>>
_BAKED_BTC_5M: dict = {
    "coef": np.array([
        +7.46811140e-02, -2.81352429e-01, -1.59619664e-01, +1.45081159e-01, -7.26858479e-02,
        +1.83000176e-01, -1.09769347e-01, +2.13961393e+00, -5.21527735e-02, -1.65886812e-01,
        +8.21320711e-02, +1.05212865e-01, -1.46708337e-01, +3.15208357e-01, -1.05335804e-01,
        -1.31159656e+00, +4.82966580e-03, +9.80865431e-02, -2.10600860e-01, -5.57749527e-01,
        -1.20373742e-01,
    ], dtype=np.float64),
    "bias": -0.02076125,
    "mean": np.array([
        +3.69315799e-06, +1.16809261e-05, +6.79981972e-05, +1.18580974e-05, +7.18808685e-05,
        +2.71476782e-04, -1.99345836e-06, +4.79128333e-01, -1.91226349e-02, -2.44701840e-02,
        -2.52417423e-03, +1.04580779e-02, +3.16244559e-03, +4.80257470e-02, +8.39626131e-03,
        +4.41521210e-02, -1.00450336e-01, +4.22768281e-01, +4.53135423e+00, -3.55764202e-02,
        -3.91807161e-05,
    ], dtype=np.float64),
    "scale": np.array([
        +4.60932382e-04, +6.85190141e-04, +1.55797774e-03, +3.82505957e-04, +6.30332546e-05,
        +1.68601514e-04, +1.91981007e-04, +3.00781267e-01, +1.91424158e-01, +2.38048847e-01,
        +5.85738172e-01, +4.90306399e-01, +4.68765593e-02, +7.86747526e-01, +5.14779342e-01,
        +6.64673560e-01, +1.76320844e+00, +2.56843654e-01, +9.80149449e-01, +5.23748473e-01,
        +2.68235602e-04,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_BTC_5M_END>>

# ── BTC 15m baked fallback (placeholder — run models/bake_weights.py after training) ──
# <<BAKED_WEIGHTS_BTC_15M_START>>
_BAKED_BTC_15M: dict = {
    "coef": np.array([
        -2.67442231e-02, +2.14266073e-01, +6.01547754e-02, -1.07979116e-01, -1.31753486e-01,
        +3.21268689e-01, -1.88576542e-01, +1.22228624e+00, +1.02848490e-02, +3.81435518e-02,
        +4.74483966e-02, +3.22256240e-01, -2.31378174e-01, +1.67464770e-01, -2.28405316e-01,
        -1.28232984e+00, +7.80156179e-02, +1.42639613e-01, -2.54877849e-01, +3.71510079e-01,
        -3.13154345e-01, +8.51756786e-02,
    ], dtype=np.float64),
    "bias": -0.18267595,
    "mean": np.array([
        +3.85134592e-06, +8.74108083e-06, +7.06348423e-05, +8.45654635e-06, +7.06009384e-05,
        +2.71225303e-04, -1.98299098e-06, +4.87866860e-01, -9.60145345e-03, -1.38978203e-02,
        +2.92418536e-03, +5.07730188e-03, +5.64640065e-03, -1.77089723e-03, +1.42078043e-02,
        +6.68981055e-02, -1.31671159e-01, +4.59293895e-01, +5.71298761e+00, -1.50238223e-02,
        -3.36997613e-05, +2.05909959e-04,
    ], dtype=np.float64),
    "scale": np.array([
        +4.57168846e-04, +6.77339276e-04, +1.54285914e-03, +3.76225581e-04, +6.15648198e-05,
        +1.65312349e-04, +1.88489984e-04, +3.14853985e-01, +1.47897236e-01, +1.91764616e-01,
        +6.40320365e-01, +4.07793787e-01, +6.05104975e-02, +9.18787460e-01, +4.34045531e-01,
        +5.99508954e-01, +1.08776998e+00, +2.70595315e-01, +9.94692954e-01, +5.99619858e-01,
        +2.61808122e-04, +2.61047231e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_BTC_15M_END>>

# ── BTC hourly baked fallback ─────────────────────────────────────────────────
# <<BAKED_WEIGHTS_BTC_HOURLY_START>>
_BAKED_BTC_HOURLY: dict = {
    "coef": np.array([
        +2.93558484e-02, -1.84249344e-01, -2.47686364e-01, -1.41242001e-01, -5.83808706e-02,
        +1.03652890e+00, -5.72080217e-01, +2.25159829e+00, +4.16129841e-03, -1.46462777e-01,
        +2.52774097e-01, -1.21372849e+00, -3.26254227e-01, +9.28683640e-01, +2.52183517e-01,
        -2.97491783e+00, +1.21987233e+00, +1.18695345e+00, -1.93213433e+00, -3.26198798e-01,
        -2.04348341e-01, -6.44732164e-01,
    ], dtype=np.float64),
    "bias": +1.38210605,
    "mean": np.array([
        +2.20323734e-05, +4.87048248e-05, +2.43910151e-04, +4.04443754e-05, +9.98476409e-05,
        +2.60579781e-04, -1.84494184e-06, +5.95077718e-01, -1.67717741e-03, -2.17906857e-03,
        -2.32221258e-02, -9.78676002e-02, +6.66787918e-03, +2.51963170e-01, -6.88680436e-02,
        +2.59254088e-02, -3.33147106e-02, +4.89742377e-01, +7.15584891e+00, +2.07188621e-01,
        -1.53256738e-04, +1.97606649e-03,
    ], dtype=np.float64),
    "scale": np.array([
        +5.88705976e-04, +8.58036791e-04, +1.92510105e-03, +4.76077793e-04, +8.60119712e-05,
        +2.37848402e-04, +2.59257195e-04, +3.05802057e-01, +6.28672264e-02, +8.72051279e-02,
        +7.16775532e-01, +4.22432556e-01, +4.38007949e-02, +8.06944844e-01, +4.06426244e-01,
        +3.66685044e-01, +5.17340440e-01, +2.88350729e-01, +1.01552984e+00, +5.72660829e-01,
        +2.94282942e-04, +4.10287306e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_BTC_HOURLY_END>>

# ── ETH baked fallbacks ───────────────────────────────────────────────────────
# <<BAKED_WEIGHTS_ETH_5M_START>>
_BAKED_ETH_5M: dict = {
    "coef": np.array([
        +5.96632175e-02, +9.21140303e-02, -2.02464442e-01, +1.36983414e-01, +4.02446217e-02,
        +2.18182572e-01, -1.25381717e-01, +1.06461545e+00, -4.76380517e-02, +1.42066834e-03,
        +7.15289132e-02, -6.79152056e-02, -1.76682770e-01, +5.67003435e-01, +3.52383462e-02,
        -1.37328025e+00, -5.61195008e-02, +1.15657302e-01, -2.17819007e-01, +2.70313289e-01,
        -4.66149198e-02,
    ], dtype=np.float64),
    "bias": -0.10589108,
    "mean": np.array([
        +4.42011881e-06, +1.34063710e-05, +7.59858113e-05, +1.33110061e-05, +7.18053824e-05,
        +2.71070020e-04, -2.00343295e-06, +4.75763579e-01, -2.20025452e-02, -2.75499279e-02,
        -5.59513005e-03, +1.18633817e-02, +4.35080526e-03, +4.42933655e-02, +1.47845151e-02,
        +6.26152082e-02, -1.26329601e-01, +4.21381500e-01, +4.52907293e+00, -4.23956911e-02,
        -4.54878031e-05,
    ], dtype=np.float64),
    "scale": np.array([
        +4.60607856e-04, +6.83395403e-04, +1.55035074e-03, +3.80707677e-04, +6.32545556e-05,
        +1.69319287e-04, +1.93031214e-04, +3.10177428e-01, +1.96584046e-01, +2.46279738e-01,
        +6.14340542e-01, +4.98487039e-01, +5.58434925e-02, +9.90433735e-01, +5.23717341e-01,
        +7.02496728e-01, +1.68860138e+00, +2.54773522e-01, +9.79061523e-01, +5.59652472e-01,
        +2.64237635e-04,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_ETH_5M_END>>

# <<BAKED_WEIGHTS_ETH_15M_START>>
_BAKED_ETH_15M: dict = {
    "coef": np.array([
        +6.72449963e-02, +2.82387139e-01, -1.00542666e-02, -1.82795731e-01, +2.31398296e-03,
        +3.24734147e-01, -1.88293228e-01, +7.21654134e-01, -4.42632864e-02, +2.54859252e-02,
        +7.55309173e-02, -2.24024831e-01, -2.68942691e-01, +6.77658574e-01, +2.32036557e-02,
        -1.55457344e+00, +2.52252234e-01, +5.05821578e-02, -1.74167953e-01, +4.81621932e-01,
        -9.16507082e-02, +1.45605735e-01,
    ], dtype=np.float64),
    "bias": -0.26034979,
    "mean": np.array([
        +3.69024225e-06, +8.68782525e-06, +7.51971220e-05, +8.94423238e-06, +7.03316414e-05,
        +2.70607340e-04, -1.88282469e-06, +4.88263998e-01, -1.06391857e-02, -1.59126630e-02,
        +3.74645158e-04, -2.24773372e-03, +1.06067502e-02, +5.27341203e-02, +1.06116256e-02,
        +1.00987414e-01, -1.76314461e-01, +4.58703184e-01, +5.71045869e+00, -1.64141305e-02,
        -4.05275167e-05, +2.18847917e-04,
    ], dtype=np.float64),
    "scale": np.array([
        +4.55979686e-04, +6.75845113e-04, +1.53419441e-03, +3.74906893e-04, +6.14076090e-05,
        +1.66083635e-04, +1.87458503e-04, +3.20056954e-01, +1.41723541e-01, +1.86639522e-01,
        +6.60997032e-01, +4.54163687e-01, +8.61779669e-02, +1.13122140e+00, +4.93738019e-01,
        +6.13812514e-01, +1.33030982e+00, +2.70801490e-01, +9.96638857e-01, +6.17663468e-01,
        +2.57080224e-04, +2.59904102e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_ETH_15M_END>>

# <<BAKED_WEIGHTS_ETH_HOURLY_START>>
_BAKED_ETH_HOURLY: dict = {
    "coef": np.array([
        -9.61647898e-04, -5.67840005e-02, -8.43790947e-01, -1.47880248e-01, -9.89454697e-02,
        +7.45598590e-01, -4.16124633e-01, +2.25505860e+00, -8.96416111e-02, -1.16100856e-01,
        -1.85697110e-02, +4.19825541e-01, +7.75697389e-02, +5.34032363e+00, +3.62391915e-01,
        -3.23795328e+00, +7.89422484e-01, +7.94207771e-01, -1.36030997e+00, +1.32714966e-02,
        -2.23649094e-01, +2.06798441e+00,
    ], dtype=np.float64),
    "bias": +2.33987514,
    "mean": np.array([
        +2.19362836e-05, +4.86614736e-05, +2.44394465e-04, +4.04975568e-05, +9.98451169e-05,
        +2.60729196e-04, -1.84850053e-06, +5.38965000e-01, -2.35255074e-03, -4.04585681e-03,
        +4.86733378e-03, -2.64258528e-02, +4.83480152e-03, +2.59048244e-01, -2.34758704e-02,
        +2.93929857e-02, -4.05103485e-02, +4.89749706e-01, +7.15585928e+00, +9.82472031e-02,
        -1.53193284e-04, +1.97597326e-03,
    ], dtype=np.float64),
    "scale": np.array([
        +5.88758827e-04, +8.57925661e-04, +1.92522583e-03, +4.76106886e-04, +8.60069905e-05,
        +2.38559803e-04, +2.59723659e-04, +3.05007904e-01, +6.52499497e-02, +8.87021404e-02,
        +6.79821813e-01, +4.02742266e-01, +3.94402269e-02, +1.44119724e+00, +3.80751965e-01,
        +5.26011057e-01, +7.98321767e-01, +2.88346246e-01, +1.01562733e+00, +5.76719601e-01,
        +2.94643775e-04, +4.10245360e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_ETH_HOURLY_END>>

# ── SOL baked fallbacks ───────────────────────────────────────────────────────
# <<BAKED_WEIGHTS_SOL_5M_START>>
_BAKED_SOL_5M: dict = {
    "coef": np.array([
        +1.28994218e-01, -7.36395651e-02, -2.25173341e-01, -8.97049049e-02, +2.36655998e-02,
        +1.44718900e-01, -7.97318778e-02, +2.46505551e+00, -8.37683045e-02, -1.22044699e-01,
        -1.07420916e-02, +4.38098049e-01, -8.09013714e-02, +1.07105079e-01, +2.22777871e-02,
        -1.15088788e+00, +3.91707603e-01, -4.10904748e-03, -9.83043303e-02, -1.13434915e-01,
        -9.85952104e-02,
    ], dtype=np.float64),
    "bias": -0.17877096,
    "mean": np.array([
        +4.07707337e-06, +1.32482042e-05, +7.82815474e-05, +1.32659934e-05, +7.15249628e-05,
        +2.70469122e-04, -1.84323050e-06, +4.79569640e-01, -1.84766339e-02, -2.40314827e-02,
        +1.86386818e-03, -1.12119266e-02, +8.59487773e-03, +2.98497364e-02, +1.25817673e-02,
        +1.08738146e-01, -1.85526084e-01, +4.18564351e-01, +4.52144677e+00, -3.60464166e-02,
        -5.22864372e-05,
    ], dtype=np.float64),
    "scale": np.array([
        +4.59515619e-04, +6.80817091e-04, +1.54322968e-03, +3.79401450e-04, +6.32411916e-05,
        +1.70344682e-04, +1.93169836e-04, +3.02777485e-01, +1.91408468e-01, +2.40364379e-01,
        +5.36427933e-01, +5.69613387e-01, +7.01725934e-02, +9.78480485e-01, +5.72631159e-01,
        +6.76270725e-01, +1.10303710e+00, +2.52872091e-01, +9.80494452e-01, +5.49964861e-01,
        +2.58988775e-04,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_SOL_5M_END>>

# <<BAKED_WEIGHTS_SOL_15M_START>>
_BAKED_SOL_15M: dict = {
    "coef": np.array([
        +3.50132065e-02, +1.59402105e-01, -2.76569569e-02, -1.86877894e-01, -1.68472283e-01,
        +3.40692857e-01, -1.80386225e-01, +1.26452731e+00, +3.03054894e-02, +1.43490600e-02,
        -6.11362064e-04, +2.17878054e-01, +1.11217409e-01, +1.01022562e-01, -8.55772436e-02,
        -1.26650067e+00, +2.29374243e-01, +8.94238835e-02, -2.36701228e-01, +2.56641905e-01,
        -1.71840437e-01, +2.28733165e-01,
    ], dtype=np.float64),
    "bias": -0.09371349,
    "mean": np.array([
        +4.61874102e-06, +1.01906832e-05, +7.58182299e-05, +9.36155262e-06, +7.01117410e-05,
        +2.69926363e-04, -2.05871835e-06, +4.85026733e-01, -1.10686006e-02, -1.61957632e-02,
        +8.75833002e-03, +8.04874405e-03, +1.11173771e-02, +2.17907869e-02, +2.51955003e-02,
        +1.11777394e-01, -2.10445196e-01, +4.57283332e-01, +5.70532175e+00, -2.35398817e-02,
        -4.45229115e-05, +2.32287541e-04,
    ], dtype=np.float64),
    "scale": np.array([
        +4.53702587e-04, +6.72057774e-04, +1.53024913e-03, +3.73110650e-04, +6.12745067e-05,
        +1.64447000e-04, +1.86685552e-04, +3.07589718e-01, +1.40331102e-01, +1.83176102e-01,
        +6.32413423e-01, +4.61104896e-01, +9.06060359e-02, +1.22122779e+00, +5.90304629e-01,
        +5.88354751e-01, +1.98691883e+00, +2.70811708e-01, +1.00025097e+00, +6.00513817e-01,
        +2.53917203e-04, +2.59243603e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_SOL_15M_END>>

# <<BAKED_WEIGHTS_SOL_HOURLY_START>>
_BAKED_SOL_HOURLY: dict = {
    "coef": np.array([
        -8.03388498e-02, +5.34735965e-01, -8.00455002e-01, -1.68991580e-01, +2.54149098e-01,
        +2.05063353e-01, -1.43700344e-01, +3.97264526e+00, -3.69282910e-02, -9.53203392e-02,
        +3.93662257e-01, +2.27634573e-01, -1.12385646e+00, +5.62230680e-01, +1.62535110e-01,
        -6.41208306e-01, -2.40401438e-01, +4.29155442e-01, -9.89291574e-02, +9.54176759e-01,
        -5.07183512e-01, +1.01330759e+00,
    ], dtype=np.float64),
    "bias": +0.15755726,
    "mean": np.array([
        +2.22527104e-05, +4.87926022e-05, +2.44086908e-04, +4.04666686e-05, +9.98464249e-05,
        +2.60594081e-04, -2.05127806e-06, +5.35500450e-01, -2.81214141e-03, -4.67607482e-03,
        +1.98564702e-03, -1.27873686e-02, +9.91229481e-03, +2.71690988e-01, +3.25071643e-02,
        +7.87416162e-02, -4.75019709e-02, +4.89744206e-01, +7.15570727e+00, +8.88416992e-02,
        -1.53324280e-04, +1.97605948e-03,
    ], dtype=np.float64),
    "scale": np.array([
        +5.89092633e-04, +8.57984340e-04, +1.92535211e-03, +4.76094090e-04, +8.60030032e-05,
        +2.38338725e-04, +2.60575774e-04, +3.20419127e-01, +6.02632630e-02, +8.09472689e-02,
        +7.06684347e-01, +5.00279806e-01, +7.49613648e-02, +1.75393606e+00, +5.29470454e-01,
        +4.56138724e-01, +5.92151742e-01, +2.88355369e-01, +1.01620735e+00, +6.10761444e-01,
        +2.94686407e-04, +4.10298651e-03,
    ], dtype=np.float64),
}
# <<BAKED_WEIGHTS_SOL_HOURLY_END>>

# ── Build active model table ──────────────────────────────────────────────────
_MODELS: dict[tuple[str, str], dict] = {}

for (_asset, _interval), _baked in [
    (("btc", "5m"),     _BAKED_BTC_5M),
    (("btc", "15m"),    _BAKED_BTC_15M),
    (("btc", "hourly"), _BAKED_BTC_HOURLY),
    (("eth", "5m"),     _BAKED_ETH_5M),
    (("eth", "15m"),    _BAKED_ETH_15M),
    (("eth", "hourly"), _BAKED_ETH_HOURLY),
    (("sol", "5m"),     _BAKED_SOL_5M),
    (("sol", "15m"),    _BAKED_SOL_15M),
    (("sol", "hourly"), _BAKED_SOL_HOURLY),
]:
    _m = _load_weights(_asset, _interval) or _baked
    if _m is not None:
        _MODELS[(_asset, _interval)] = _m

del _asset, _interval, _baked, _m


def _slug_to_asset(slug: str) -> str:
    s = slug.lower()
    if s.startswith(("btc-", "bitcoin-")):
        return "btc"
    if s.startswith(("eth-", "ethereum-")):
        return "eth"
    if s.startswith(("sol-", "solana-")):
        return "sol"
    return ""


# Trading hyperparameters
_ENTRY_THRESH = 0.04  # min (p_pred - market_ask) edge to enter
_STOP_MARGIN  = 0.05  # stop-loss: exit YES if p < 0.5 - STOP_MARGIN, NO if p > 0.5 + STOP_MARGIN
_HALF_KELLY   = 0.50  # fraction of full Kelly to wager
_MAX_SHARES   = 500   # hard position cap per token per market
_T_LO         = 0.15  # no new entries when time_remaining_frac < _T_LO
_T_HI         = 0.85  # no new entries when time_remaining_frac > _T_HI


def _predict_yes_prob(ctx: FeatureContext, model: dict, registry: list) -> float:
    """Return P(YES) from the given logistic regression model."""
    n = len(model["coef"])
    x = np.array([fn(ctx) for fn in registry[:n]], dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x_sc = (x - model["mean"]) / model["scale"]
    logit = float(model["coef"] @ x_sc) + model["bias"]
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
# ── Feature registry
# ──────────────────────────────────────────────────────────────────────────────
# Order is fixed: the logistic regression weight vector is positional.
# Never reorder existing entries. Always append new features at the end.

# 5m feature set — positional order is fixed; weights are indexed by position.
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

# 15m feature set — same as 5m plus the full-duration BTC return.
# feat_btc_return_300s (index 2) stays as a sub-interval signal.
FEATURE_REGISTRY_15M = FEATURE_REGISTRY + [
    feat_btc_return_900s,          # 21 — full 15m horizon return
]

# Hourly (BTC) feature set — 5m base + 1-hour horizon return.
FEATURE_REGISTRY_HOURLY = FEATURE_REGISTRY + [
    feat_btc_return_3600s,         # 21 — full 1-hour horizon return
]

# ── Primary-asset registries for ETH and SOL markets ─────────────────────────
# These mirror the BTC registries but use feat_primary_* functions so that the
# model learns from ETH/SOL price dynamics instead of BTC price dynamics.
# ETH 5m and SOL 5m share the same registry; their weight vectors differ.

FEATURE_REGISTRY_PRIMARY_5M = [
    # Group A' — primary asset momentum
    feat_primary_return_30s,           # 0
    feat_primary_return_60s,           # 1
    feat_primary_return_300s,          # 2
    feat_primary_ema_cross_20_120,     # 3
    feat_primary_realized_vol_60s,     # 4
    # Group B' — primary asset spread
    feat_primary_spread_norm,          # 5
    feat_primary_spread_delta_30s,     # 6
    # Group C — Polymarket (asset-agnostic)
    feat_yes_price,                    # 7
    feat_yes_price_mom_30s,            # 8
    feat_yes_price_mom_60s,            # 9
    # Group D — Order book imbalance (asset-agnostic)
    feat_yes_obi_level1,               # 10
    feat_yes_obi_multilevel,           # 11
    feat_yes_microprice_dev,           # 12
    feat_yes_depth_ratio,              # 13
    feat_yes_near_book_imbalance,      # 14
    # Group E — Polymarket spread (asset-agnostic)
    feat_yes_spread_norm,              # 15
    feat_yes_spread_delta_30s,         # 16
    # Group F — Market timing (asset-agnostic)
    feat_time_remaining_frac,          # 17
    feat_log_time_remaining,           # 18
    # Group G' — cross-signal divergence (primary asset vs poly)
    feat_primary_vs_poly_divergence,   # 19
    feat_chainlink_primary_vs_binance, # 20
]

FEATURE_REGISTRY_PRIMARY_15M = FEATURE_REGISTRY_PRIMARY_5M + [
    feat_primary_return_900s,          # 21 — full 15m horizon return
]

FEATURE_REGISTRY_PRIMARY_HOURLY = FEATURE_REGISTRY_PRIMARY_5M + [
    feat_primary_return_3600s,         # 21 — full 1-hour horizon return
]

# Registry lookup by (asset, interval) — all 9 market types
_REGISTRIES: dict[tuple[str, str], list] = {
    # BTC: uses BTC-specific feat_btc_* functions (existing trained models)
    ("btc", "5m"):     FEATURE_REGISTRY,
    ("btc", "15m"):    FEATURE_REGISTRY_15M,
    ("btc", "hourly"): FEATURE_REGISTRY_HOURLY,
    # ETH: uses feat_primary_* → primary_mids = eth_mids at inference time
    ("eth", "5m"):     FEATURE_REGISTRY_PRIMARY_5M,
    ("eth", "15m"):    FEATURE_REGISTRY_PRIMARY_15M,
    ("eth", "hourly"): FEATURE_REGISTRY_PRIMARY_HOURLY,
    # SOL: uses feat_primary_* → primary_mids = sol_mids at inference time
    ("sol", "5m"):     FEATURE_REGISTRY_PRIMARY_5M,
    ("sol", "15m"):    FEATURE_REGISTRY_PRIMARY_15M,
    ("sol", "hourly"): FEATURE_REGISTRY_PRIMARY_HOURLY,
}

# ── Candidate injection (used by autofeature.py during evaluation only) ───────
# autofeature.py writes _candidate.py to the strategies/ directory before running
# training/CV. This block picks it up so the candidate appears as the last entry
# in FEATURE_REGISTRY. Cleanup removes _candidate.py after evaluation.
try:
    from _candidate import CANDIDATE_FEATURES as _cands  # type: ignore[import]
    for _fn in _cands:
        if _fn not in FEATURE_REGISTRY:
            FEATURE_REGISTRY.append(_fn)
except ImportError:
    pass
# ── END ACCEPTED FEATURES


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
        self.eth_mids:      deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.eth_spreads:   deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.chainlink_eth: deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.sol_mids:      deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.sol_spreads:   deque[float] = deque(maxlen=_GLOBAL_HISTORY)
        self.chainlink_sol: deque[float] = deque(maxlen=_GLOBAL_HISTORY)

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
        self.eth_mids.append(state.eth_mid)
        self.eth_spreads.append(state.eth_spread)
        self.chainlink_eth.append(state.chainlink_eth)
        self.sol_mids.append(state.sol_mid)
        self.sol_spreads.append(state.sol_spread)
        self.chainlink_sol.append(state.chainlink_sol)

        # 2. Record per-market data.
        for slug, market in state.markets.items():
            if slug not in self.market_history:
                capacity = _INTERVAL_TICKS.get(market.interval, 3600)
                self.market_history[slug] = deque(maxlen=capacity)
            self.market_history[slug].append(market)

        # 3. Trading logic: route each market to its model by (asset, interval).
        orders: list[Order] = []
        for slug, market in state.markets.items():
            asset = _slug_to_asset(slug)
            model_key = (asset, market.interval)
            model = _MODELS.get(model_key)
            if model is None:
                continue
            if len(self.market_history[slug]) < _MIN_WARMUP:
                continue

            registry = _REGISTRIES[model_key]
            ctx = self._build_context(slug)
            p = _predict_yes_prob(ctx, model, registry)
            pos = state.positions.get(slug)
            yes_sh = pos.yes_shares if pos else 0.0
            no_sh  = pos.no_shares  if pos else 0.0

            # ── STOP-LOSS ─────────────────────────────────────────────────────
            # Exit only when model has meaningfully reversed against the position.
            # Otherwise hold to settlement and collect the full $1 payout.
            if yes_sh > 0.5:
                if p < 0.5 - _STOP_MARGIN:
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.SELL,
                        size=yes_sh, limit_price=None,
                    ))
                    self.closed_markets.add(slug)
                continue

            if no_sh > 0.5:
                if p > 0.5 + _STOP_MARGIN:
                    orders.append(Order(
                        market_slug=slug, token=Token.NO, side=Side.SELL,
                        size=no_sh, limit_price=None,
                    ))
                    self.closed_markets.add(slug)
                continue

            # ── ENTRY: model vs market comparison ─────────────────────────────
            # Buy when the model thinks the market is underpricing the token.
            # Edge = p_pred - market_ask; Kelly sizes proportionally to edge.
            if slug in self.closed_markets:
                continue
            if not (_T_LO < market.time_remaining_frac < _T_HI):
                continue

            if p - market.yes_ask > _ENTRY_THRESH and market.yes_ask > 0:
                size = min(_kelly_shares(p, market.yes_ask, state.cash),
                           int(_MAX_SHARES - yes_sh))
                if size >= 1:
                    orders.append(Order(
                        market_slug=slug, token=Token.YES, side=Side.BUY,
                        size=float(size), limit_price=None,
                    ))
            elif (1.0 - p) - market.no_ask > _ENTRY_THRESH and market.no_ask > 0:
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
        asset = _slug_to_asset(slug)
        btc_m  = np.array(self.btc_mids)
        eth_m  = np.array(self.eth_mids)
        sol_m  = np.array(self.sol_mids)
        if asset == "eth":
            primary_m  = eth_m
            primary_sp = np.array(self.eth_spreads)
            cl_primary = np.array(self.chainlink_eth)
        elif asset == "sol":
            primary_m  = sol_m
            primary_sp = np.array(self.sol_spreads)
            cl_primary = np.array(self.chainlink_sol)
        else:
            primary_m  = btc_m
            primary_sp = np.array(self.btc_spreads)
            cl_primary = np.array(self.chainlink_btc)
        return FeatureContext(
            market_hist=list(self.market_history[slug]),
            btc_mids=btc_m,
            btc_spreads=np.array(self.btc_spreads),
            chainlink_btc=np.array(self.chainlink_btc),
            timestamps=np.array(self.timestamps),
            primary_mids=primary_m,
            primary_spreads=primary_sp,
            chainlink_primary=cl_primary,
            eth_mids=eth_m,
            eth_spreads=np.array(self.eth_spreads),
            chainlink_eth=np.array(self.chainlink_eth),
            sol_mids=sol_m,
            sol_spreads=np.array(self.sol_spreads),
            chainlink_sol=np.array(self.chainlink_sol),
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

    def get_forecasts(self, state: MarketState) -> dict[str, float]:
        """Return {market_slug: P(YES)} for all active modeled markets with warmup."""
        forecasts: dict[str, float] = {}
        for slug, market in state.markets.items():
            asset = _slug_to_asset(slug)
            model_key = (asset, market.interval)
            model = _MODELS.get(model_key)
            if model is None:
                continue
            hist = self.market_history.get(slug)
            if hist is None or len(hist) < _MIN_WARMUP:
                continue
            registry = _REGISTRIES[model_key]
            ctx = self._build_context(slug)
            forecasts[slug] = _predict_yes_prob(ctx, model, registry)
        return forecasts

    def on_fill(self, fill: Fill) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    #  on_settlement
    # ─────────────────────────────────────────────────────────────────────────

    def on_settlement(self, settlement: Settlement) -> None:
        slug = settlement.market_slug
        self.market_history.pop(slug, None)
        self.closed_markets.discard(slug)
