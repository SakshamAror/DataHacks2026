"""
Offline logistic regression training pipeline for BTC Polymarket prediction markets.

Usage:
    cd /Users/devangpant/logreg-strategy
    python train_model.py

Reads data from the paths in .env, writes trained weights to WEIGHTS_PATH.
"""

from dotenv import load_dotenv
import os
import math
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

load_dotenv()

TRAIN_DB      = os.environ["TRAIN_DB"]
TRAIN_BINANCE = os.environ["TRAIN_BINANCE"]
TRAIN_BOOKS   = os.environ["TRAIN_BOOKS"]
VAL_DB        = os.environ["VAL_DB"]
VAL_BINANCE   = os.environ["VAL_BINANCE"]
VAL_BOOKS     = os.environ["VAL_BOOKS"]
WEIGHTS_PATH  = os.environ["WEIGHTS_PATH"]

N        = int(os.environ.get("N", "60"))
LR       = float(os.environ.get("LR", "0.1"))
LAMBDA_  = float(os.environ.get("LAMBDA_", "0.01"))
EPOCHS   = int(os.environ.get("EPOCHS", "100"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))

FEATURE_NAMES = [
    "chainlink_lag",
    "btc_spread",
    f"btc_momentum_{N}",
    f"chainlink_momentum_{N}",
    "btc_drift_from_open",
    "chainlink_drift_from_open",
    "time_remaining_frac",
    "yes_ask",
    "book_imbalance",
]


# ── Slug parsing ────────────────────────────────────────────────────────────────

_PREFIXES_5M  = ["btc-updown-5m", "sol-updown-5m", "eth-updown-5m"]
_PREFIXES_15M = ["btc-updown-15m", "sol-updown-15m", "eth-updown-15m"]
_HOURLY_PAT   = re.compile(
    r"^(bitcoin|solana|ethereum)-up-or-down-([a-z]+)-(\d+)-(\d{4})-(\d+)(am|pm)-et$"
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_slug(slug: str):
    """Return (start_ts, end_ts, interval) or None."""
    for prefix in _PREFIXES_5M:
        m = re.match(rf"^{re.escape(prefix)}-(\d+)$", slug)
        if m:
            s = int(m.group(1))
            return s, s + 300, "5m"
    for prefix in _PREFIXES_15M:
        m = re.match(rf"^{re.escape(prefix)}-(\d+)$", slug)
        if m:
            s = int(m.group(1))
            return s, s + 900, "15m"
    m = _HOURLY_PAT.match(slug)
    if m:
        _asset, month_name, day, year, hour, ampm = m.groups()
        month = _MONTHS.get(month_name)
        if month is None:
            return None
        h = int(hour)
        if ampm == "pm" and h != 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        from datetime import datetime
        try:
            import zoneinfo
            et = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            from datetime import timezone
            et = timezone.utc
        try:
            dt = datetime(int(year), month, int(day), h, 0, 0, tzinfo=et)
        except (ValueError, OverflowError):
            return None
        s = int(dt.timestamp())
        return s, s + 3600, "hourly"
    return None


# ── Logistic regression ─────────────────────────────────────────────────────────

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def predict(features, weights, bias):
    z = sum(w * f for w, f in zip(weights, features)) + bias
    return sigmoid(z)


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = max(eps, min(1 - eps, y_pred))
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def train(X, y, lr=LR, lambda_=LAMBDA_, epochs=EPOCHS):
    import random
    n = len(X[0])
    weights = [0.0] * n
    bias = 0.0
    data = list(zip(X, y))
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        for feats, label in data:
            p = predict(feats, weights, bias)
            err = p - label
            total_loss += binary_cross_entropy(label, p)
            for i in range(n):
                weights[i] -= lr * (err * feats[i] + lambda_ * weights[i])
            bias -= lr * err
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(X):.6f}")
    return weights, bias


def save_weights(weights, bias, feature_names, means, stds, path):
    with open(path, "w") as f:
        json.dump({
            "weights": weights,
            "bias": bias,
            "features": feature_names,
            "means": means,
            "stds": stds,
        }, f, indent=2)


def load_weights(path):
    with open(path) as f:
        d = json.load(f)
    return d["weights"], d["bias"], d["features"], d["means"], d["stds"]


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_books(books_path: str) -> pd.DataFrame:
    """Load orderbook CSVs from a file or directory."""
    p = Path(books_path)
    frames = []
    if p.is_dir():
        for f in sorted(p.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  warning: skipping {f.name}: {e}")
    elif p.is_file():
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp_us" in df.columns:
        df["ts_sec"] = df["timestamp_us"] // 1_000_000
    return df


def _load_binance(binance_path: str) -> pd.DataFrame:
    """Load Binance LOB from a parquet file or directory of parquets."""
    p = Path(binance_path)
    frames = []
    if p.is_dir():
        candidates = list(p.glob("btcusdt*.parquet")) or list(p.glob("*.parquet"))
        for f in candidates:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as e:
                print(f"  warning: skipping {f.name}: {e}")
    elif p.is_file() and p.exists():
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def build_features(db_path: str, binance_path: str, books_path: str) -> pd.DataFrame:
    """
    Load all raw data and join into a feature DataFrame.
    Each row = one (market_slug, ts_sec) observation.
    Only BTC markets are included; only resolved markets get a label.
    """
    print("  Loading market prices (BTC only)...")
    conn = sqlite3.connect(db_path)
    prices_df = pd.read_sql_query(
        """SELECT timestamp_us, interval, market_slug,
                  yes_price, no_price, yes_bid, yes_ask, no_bid, no_ask
           FROM market_prices
           WHERE market_slug LIKE 'btc-%' OR market_slug LIKE 'bitcoin-%'
           ORDER BY timestamp_us""",
        conn,
    )
    prices_df["ts_sec"] = prices_df["timestamp_us"] // 1_000_000

    print("  Loading Chainlink BTC prices...")
    chainlink_per_sec = pd.DataFrame(columns=["ts_sec", "chainlink_btc"])
    try:
        cl_df = pd.read_sql_query(
            """SELECT timestamp_us, price FROM rtds_prices
               WHERE source='chainlink' AND symbol='BTC/USD'
               ORDER BY timestamp_us""",
            conn,
        )
        if not cl_df.empty:
            cl_df["ts_sec"] = cl_df["timestamp_us"] // 1_000_000
            chainlink_per_sec = (
                cl_df.groupby("ts_sec")["price"].last()
                .reset_index()
                .rename(columns={"price": "chainlink_btc"})
            )
    except Exception as e:
        print(f"  warning: could not load Chainlink prices: {e}")

    print("  Loading market outcomes (DB table)...")
    db_outcomes: dict[str, str] = {}
    try:
        cur = conn.execute(
            "SELECT market_slug, outcome FROM market_outcomes WHERE status='resolved'"
        )
        db_outcomes = {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        pass
    conn.close()

    # Derive outcomes from Chainlink for every market whose slug we can parse.
    # This is the same logic the backtester uses: close >= open → YES.
    print("  Deriving outcomes from Chainlink prices...")
    chainlink_idx = chainlink_per_sec.set_index("ts_sec")["chainlink_btc"]
    outcomes: dict[str, str] = {}
    all_slugs = prices_df["market_slug"].unique()
    for slug in all_slugs:
        # Use DB table first if available and slug matches this time window
        if slug in db_outcomes:
            outcomes[slug] = db_outcomes[slug]
            continue
        lc = parse_slug(slug)
        if lc is None:
            continue
        start_ts, end_ts, _ = lc
        # Find nearest Chainlink price within ±5s of start and end
        def _nearest(ts):
            if chainlink_idx.empty:
                return None
            candidates = chainlink_idx.loc[max(ts-5, chainlink_idx.index.min()):ts+5]
            if candidates.empty:
                candidates = chainlink_idx
            diff = (candidates.index.to_series() - ts).abs()
            return float(candidates.iloc[diff.argmin()])
        open_p  = _nearest(start_ts)
        close_p = _nearest(end_ts)
        if open_p is None or close_p is None:
            continue
        outcomes[slug] = "YES" if close_p >= open_p else "NO"

    print(f"  {len(prices_df):,} price rows, {len(outcomes):,} outcomes derived "
          f"({len(db_outcomes):,} from DB, {len(outcomes)-len(db_outcomes):,} from Chainlink)")

    print("  Loading Binance LOB...")
    binance_df = _load_binance(binance_path)
    if not binance_df.empty and "bid_price_1" in binance_df.columns:
        binance_df["ts_sec"] = binance_df["timestamp_us"] // 1_000_000
        # Keep only BTC rows (bid > $10k distinguishes BTC from ETH/SOL)
        binance_df = binance_df[binance_df["bid_price_1"] > 10_000].copy()
        binance_df["btc_mid"] = (binance_df["bid_price_1"] + binance_df["ask_price_1"]) / 2
        binance_df["_spread"] = binance_df["ask_price_1"] - binance_df["bid_price_1"]
        binance_per_sec = (
            binance_df.groupby("ts_sec")
            .agg(btc_mid=("btc_mid", "last"), btc_spread=("_spread", "last"))
            .reset_index()
        )
        print(f"  {len(binance_per_sec):,} Binance seconds loaded")
    else:
        print("  warning: no Binance data found; btc_mid/spread will be 0")
        binance_per_sec = pd.DataFrame(columns=["ts_sec", "btc_mid", "btc_spread"])

    print("  Loading orderbooks...")
    books_df = _load_books(books_path)
    if not books_df.empty and "market_slug" in books_df.columns:
        books_df = books_df[
            books_df["market_slug"].str.startswith("btc-") |
            books_df["market_slug"].str.startswith("bitcoin-")
        ].copy()
        bid_col = books_df.get("yes_total_bid_size", pd.Series(dtype=float)).reindex(books_df.index, fill_value=0.0)
        ask_col = books_df.get("yes_total_ask_size", pd.Series(dtype=float)).reindex(books_df.index, fill_value=0.0)
        if "yes_total_bid_size" in books_df.columns:
            bid_col = books_df["yes_total_bid_size"].fillna(0.0)
            ask_col = books_df["yes_total_ask_size"].fillna(0.0)
        books_df["book_imbalance"] = (bid_col - ask_col) / (bid_col + ask_col + 1e-9)
        books_per_mkt = books_df[["ts_sec", "market_slug", "book_imbalance"]].copy()
        print(f"  {len(books_per_mkt):,} orderbook rows loaded")
    else:
        print("  warning: no orderbook data; book_imbalance will be 0")
        books_per_mkt = pd.DataFrame(columns=["ts_sec", "market_slug", "book_imbalance"])

    # ── Global per-second timeline: Binance + Chainlink ─────────────────────────
    print("  Building global timeline...")
    if not binance_per_sec.empty and not chainlink_per_sec.empty:
        global_ts = pd.merge_asof(
            binance_per_sec.sort_values("ts_sec"),
            chainlink_per_sec.sort_values("ts_sec"),
            on="ts_sec",
            direction="backward",
        )
    elif not binance_per_sec.empty:
        global_ts = binance_per_sec.copy()
        global_ts["chainlink_btc"] = float("nan")
    elif not chainlink_per_sec.empty:
        global_ts = chainlink_per_sec.copy()
        global_ts["btc_mid"] = float("nan")
        global_ts["btc_spread"] = 0.0
    else:
        print("ERROR: no Binance or Chainlink data at all.")
        return pd.DataFrame()

    global_ts = global_ts.sort_values("ts_sec").reset_index(drop=True)

    # Rolling momentum (N seconds back)
    global_ts[f"btc_momentum_{N}"] = (
        global_ts["btc_mid"] - global_ts["btc_mid"].shift(N)
    )
    global_ts[f"chainlink_momentum_{N}"] = (
        global_ts["chainlink_btc"] - global_ts["chainlink_btc"].shift(N)
    )
    global_ts["chainlink_lag"] = global_ts["btc_mid"] - global_ts["chainlink_btc"]

    global_cols = [
        "ts_sec", "btc_mid", "chainlink_btc",
        "chainlink_lag", "btc_spread",
        f"btc_momentum_{N}", f"chainlink_momentum_{N}",
    ]

    # ── Merge global features onto market price rows ─────────────────────────────
    print("  Merging global features onto market prices...")
    merged = pd.merge_asof(
        prices_df.sort_values("ts_sec"),
        global_ts[global_cols].sort_values("ts_sec"),
        on="ts_sec",
        direction="backward",
    )

    # ── Merge per-market orderbook imbalance ─────────────────────────────────────
    if not books_per_mkt.empty:
        merged = pd.merge_asof(
            merged.sort_values("ts_sec"),
            books_per_mkt.sort_values("ts_sec"),
            on="ts_sec",
            by="market_slug",
            direction="backward",
        )
    else:
        merged["book_imbalance"] = 0.0
    merged["book_imbalance"] = merged["book_imbalance"].fillna(0.0)

    # ── Per-market derived features ──────────────────────────────────────────────
    merged["market_mispricing"] = merged["yes_ask"] - 0.5

    # Vectorised time_remaining_frac via slug lifecycle lookup
    slug_lc: dict[str, tuple[int, int]] = {}
    for slug in merged["market_slug"].unique():
        lc = parse_slug(slug)
        if lc:
            slug_lc[slug] = (lc[0], lc[1])

    merged["_start_ts"] = merged["market_slug"].map(
        lambda s: slug_lc.get(s, (0, 0))[0]
    )
    merged["_end_ts"] = merged["market_slug"].map(
        lambda s: slug_lc.get(s, (0, 0))[1]
    )
    duration = (merged["_end_ts"] - merged["_start_ts"]).replace(0, 1)
    merged["time_remaining_frac"] = (
        (merged["_end_ts"] - merged["ts_sec"]) / duration
    ).clip(0.0, 1.0)
    merged.drop(columns=["_start_ts", "_end_ts"], inplace=True)

    # ── Drift from market open ────────────────────────────────────────────────────
    # For each market, find btc_mid and chainlink_btc at the earliest tick.
    # Drift = current - open: directly encodes whether BTC is up/down from open,
    # which is exactly what determines YES/NO settlement.
    merged = merged.sort_values(["market_slug", "ts_sec"])
    open_prices = (
        merged.groupby("market_slug")[["btc_mid", "chainlink_btc"]]
        .first()
        .rename(columns={"btc_mid": "_btc_open", "chainlink_btc": "_cl_open"})
    )
    merged = merged.join(open_prices, on="market_slug")
    merged["btc_drift_from_open"]      = merged["btc_mid"]      - merged["_btc_open"]
    merged["chainlink_drift_from_open"] = merged["chainlink_btc"] - merged["_cl_open"]
    merged.drop(columns=["_btc_open", "_cl_open"], inplace=True)

    # ── Labels ───────────────────────────────────────────────────────────────────
    merged["label"] = merged["market_slug"].map(outcomes).map({"YES": 1, "NO": 0})

    # Drop rows with no resolved label; fill missing features with 0
    before = len(merged)
    merged = merged.dropna(subset=["label"])
    merged[FEATURE_NAMES] = merged[FEATURE_NAMES].fillna(0.0)
    print(f"  {len(merged):,} rows kept ({before - len(merged):,} dropped for no-outcome)")
    print(f"  {merged['market_slug'].nunique():,} unique markets, "
          f"label balance: {merged['label'].mean():.3f}")

    return merged


# ── Normalization ───────────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame, feature_cols: list[str], means=None, stds=None):
    """Z-score normalize. Compute means/stds from df if not provided."""
    X = df[feature_cols].values.astype(float)
    if means is None:
        means = X.mean(axis=0).tolist()
        stds = X.std(axis=0).tolist()
    safe_stds = [s if s > 1e-9 else 1.0 for s in stds]
    X_norm = (X - np.array(means)) / np.array(safe_stds)
    return X_norm.tolist(), means, safe_stds


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate(X, y, weights, bias):
    correct = 0
    brier = 0.0
    for feats, label in zip(X, y):
        p = predict(feats, weights, bias)
        correct += int((p >= 0.5) == bool(label))
        brier += (p - label) ** 2
    n = len(y)
    return correct / n, brier / n


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading training data ===")
    df_train = build_features(TRAIN_DB, TRAIN_BINANCE, TRAIN_BOOKS)

    if df_train.empty:
        print("\nERROR: no training data. Run 'python download_data.py' first.")
        return

    if len(df_train) > MAX_ROWS:
        print(f"\nSubsampling {MAX_ROWS:,} rows from {len(df_train):,} (set MAX_ROWS to change)")
        df_train = df_train.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

    X_train, means, stds = normalize(df_train, FEATURE_NAMES)
    y_train = df_train["label"].astype(int).tolist()

    print(f"\n=== Training logistic regression ===")
    print(f"  rows={len(X_train):,}  N={N}  lr={LR}  lambda={LAMBDA_}  epochs={EPOCHS}")
    weights, bias = train(X_train, y_train)

    train_acc, train_brier = evaluate(X_train, y_train, weights, bias)
    print(f"\n  train  accuracy={train_acc:.4f}  Brier={train_brier:.4f}")

    print("\n=== Loading validation data ===")
    df_val = build_features(VAL_DB, VAL_BINANCE, VAL_BOOKS)

    if not df_val.empty:
        if len(df_val) > MAX_ROWS:
            df_val = df_val.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
        X_val, _, _ = normalize(df_val, FEATURE_NAMES, means=means, stds=stds)
        y_val = df_val["label"].astype(int).tolist()
        val_acc, val_brier = evaluate(X_val, y_val, weights, bias)
        print(f"  val    accuracy={val_acc:.4f}  Brier={val_brier:.4f}")
    else:
        print("  WARNING: no validation data found; skipping eval")

    print(f"\n=== Saving weights to {WEIGHTS_PATH} ===")
    save_weights(weights, bias, FEATURE_NAMES, means, stds, WEIGHTS_PATH)

    print("\nFeature weights:")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  {name:30s}  {w:+.6f}")
    print(f"  {'bias':30s}  {bias:+.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
