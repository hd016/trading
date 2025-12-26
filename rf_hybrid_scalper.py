"""
RF + Rule-Based Hybrid Crypto Scalper (LONG + SHORT) — Binance USDT-M Futures (1m)
CONFIG-ONLY VERSION (NO PARAMETERS IN CODE)

- All parameters are loaded from YAML only (config.yaml).
- If a required key is missing -> hard error with clear message.
- Downloads OHLCV from Binance Futures (public endpoint, no keys needed)
- Builds Pine-like regime filters (pre-alert + entry score) for LONG and SHORT
- Walk-forward RandomForest predicts "TP hit before SL within horizon"
- Hybrid signal = rules (+ optional prealert) (+ optional RF confirmation)
- Backtest with TP/SL (RR=1:2 default), fees/slippage, cooldown,
  and heuristic when TP & SL hit in same bar.

Requirements:
    pip install pandas numpy scikit-learn requests pyyaml

Run:
    python rf_hybrid_scalper.py --config config.yaml
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

import os

def cache_path(cfg: dict) -> Path:
    cache_dir = Path(c(cfg, "cache.dir", str))
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbol = c(cfg, "data.symbol", str)
    interval = c(cfg, "data.interval", str)
    days = c(cfg, "data.days", int)
    fmt = c(cfg, "cache.format", str).lower()

    fname = f"{symbol}_{interval}_{days}d.{ 'parquet' if fmt=='parquet' else 'csv' }"
    return cache_dir / fname

def load_cached_ohlcv(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
        # Expect timestamp column saved
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            # fallback: first column might be index
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[["open","high","low","close","volume"]].sort_index()
    else:
        raise ValueError(f"Unsupported cache.format: {fmt}")

def save_cached_ohlcv(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "parquet":
        # parquet saves index fine, but to be safe keep index
        df.to_parquet(path)
    elif fmt == "csv":
        out = df.copy()
        out = out.reset_index().rename(columns={"index": "timestamp"})
        out.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported cache.format: {fmt}")

def load_or_fetch_ohlcv(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    enabled = bool(c(cfg, "cache.enabled"))
    fmt = c(cfg, "cache.format", str).lower()

    symbol = c(cfg, "data.symbol", str)
    interval = c(cfg, "data.interval", str)
    days = c(cfg, "data.days", int)

    if not enabled:
        print("Cache disabled -> fetching...")
        return load_binance_futures_range(symbol, interval, days)

    path = cache_path(cfg)

    if path.exists() and not refresh:
        print("Loading cached bars:", path.resolve())
        df = load_cached_ohlcv(path, fmt)
        # basic sanity
        if len(df) > 0 and set(["open","high","low","close","volume"]).issubset(df.columns):
            return df
        else:
            print("Cache file invalid -> refetching...")

    print("Fetching bars from Binance...")
    df = load_binance_futures_range(symbol, interval, days)
    print("Saving cache:", path.resolve())
    save_cached_ohlcv(df, path, fmt)
    return df


# ============================================================
# YAML CONFIG (NO DEFAULTS)
# ============================================================
REQUIRED_KEYS = [
    "data.symbol", "data.interval", "data.days",
    "labeling.horizon_bars", "labeling.atr_len", "labeling.rr", "labeling.sl_atr_mult",
    "filters.warningPeriod", "filters.momentum_dump_th", "filters.momentum_pump_th",
    "filters.volumeClusterMult", "filters.minEntryScore", "filters.rsiDumpTh", "filters.rsiPumpTh",
    "filters.useMTF", "filters.prealert_lookback",
    "rf_confirm.require_rf", "rf_confirm.p_long_th", "rf_confirm.p_short_th", "rf_confirm.require_prealert",
    "walkforward.train_bars", "walkforward.test_bars", "walkforward.step_bars",
    "model.n_estimators", "model.max_depth", "model.min_samples_leaf", "model.random_state",
    "execution.fee_bps", "execution.slippage_bps", "execution.cooldown_bars",
    "output.backtest_csv",
]


def load_yaml_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_cfg(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: '{dotted}'")
        cur = cur[part]
    return cur


def validate_cfg(cfg: dict):
    missing = []
    for k in REQUIRED_KEYS:
        try:
            get_cfg(cfg, k)
        except KeyError:
            missing.append(k)
    if missing:
        msg = "config.yaml is missing required keys:\n" + "\n".join(f"- {k}" for k in missing)
        raise ValueError(msg)


def c(cfg: dict, dotted: str, cast_type=None):
    v = get_cfg(cfg, dotted)
    if cast_type is None:
        return v
    if v is None:
        return None
    return cast_type(v)


# ============================================================
# BINANCE DATA (USDT-M FUTURES)
# ============================================================
def _binance_futures_klines(symbol="BTCUSDT", interval="1m", start_ms=None, end_ms=None, limit=1000) -> pd.DataFrame:
    base = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()


def load_binance_futures_range(symbol: str, interval: str, days: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(days * 24 * 60 * 60 * 1000)

    out = []
    cur = start_ms
    while True:
        df = _binance_futures_klines(symbol, interval, start_ms=cur, end_ms=now_ms, limit=1000)
        if df.empty:
            break
        out.append(df)

        last_ts = df.index[-1]
        nxt = int(last_ts.timestamp() * 1000) + 60_000  # next minute
        if nxt >= now_ms:
            break
        cur = nxt
        time.sleep(0.12)

    df_all = pd.concat(out).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    return df_all


# ============================================================
# INDICATORS
# ============================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / (ad + 1e-12)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c_ = df["high"], df["low"], df["close"]
    pc = c_.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(close: pd.Series, fast=8, slow=21, signal=5) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m - s


# ============================================================
# FEATURES (RF)
# ============================================================
def make_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    atr_len = c(cfg, "labeling.atr_len", int)

    out = df.copy()
    out["ret1"] = out["close"].pct_change(1)
    out["ret3"] = out["close"].pct_change(3)
    out["ret8"] = out["close"].pct_change(8)

    out["vol10"] = out["ret1"].rolling(10).std()
    out["vol30"] = out["ret1"].rolling(30).std()

    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["ema_diff"] = (out["ema20"] - out["ema50"]) / (out["close"] + 1e-12)

    out["rsi14"] = rsi(out["close"], 14) / 100.0

    out["atr"] = atr(out, atr_len)
    out["atr_n"] = out["atr"] / (out["close"] + 1e-12)

    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["volume"] / (out["vol_sma20"] + 1e-12)

    out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)
    out["body"] = (out["close"] - out["open"]) / (out["close"] + 1e-12)
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / (out["close"] + 1e-12)
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / (out["close"] + 1e-12)

    return out.dropna()


# ============================================================
# PINE-LIKE FILTERS (PREALERT + ENTRY SCORE) LONG & SHORT
# ============================================================
def add_pine_like_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    wp = c(cfg, "filters.warningPeriod", int)
    momentum_dump_th = c(cfg, "filters.momentum_dump_th", float)
    momentum_pump_th = c(cfg, "filters.momentum_pump_th", float)
    volumeClusterMult = c(cfg, "filters.volumeClusterMult", float)
    minEntryScore = c(cfg, "filters.minEntryScore", int)
    rsiDumpTh = c(cfg, "filters.rsiDumpTh", float)
    rsiPumpTh = c(cfg, "filters.rsiPumpTh", float)
    useMTF = bool(c(cfg, "filters.useMTF"))

    out = df.copy()

    # 1m core
    out["rsi10"] = rsi(out["close"], 10)
    out["macd_hist_1m"] = macd_hist(out["close"], 8, 21, 5)
    out["atr14"] = atr(out, 14)
    out["atr_sma14"] = out["atr14"].rolling(14).mean()

    # RSI trend over wp bars
    out["rsi_trend"] = out["rsi10"] - out["rsi10"].shift(wp - 1)
    out["rsi_degrade"] = out["rsi_trend"] < 0
    out["rsi_improve"] = out["rsi_trend"] > 0

    # Momentum over wp
    first_close = out["close"].shift(wp - 1)
    out["avg_momentum_pct"] = ((out["close"] - first_close) / (first_close + 1e-12)) * 100.0

    # Volume cluster
    out["avg_vol_last3"] = out["volume"].rolling(3).mean()
    out["volume_cluster"] = out["avg_vol_last3"] > (out["vol_sma20"] * volumeClusterMult)

    # MACD hist trend
    out["macd_trend"] = out["macd_hist_1m"] - out["macd_hist_1m"].shift(wp - 1)
    out["macd_deteriorating"] = out["macd_trend"] <= 0
    out["macd_improving"] = out["macd_trend"] >= 0

    # MTF context
    if useMTF:
        ohlc15 = out[["open", "high", "low", "close", "volume"]].resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        ohlc15["mtf_rsi14"] = rsi(ohlc15["close"], 14)
        ohlc15["mtf_ema50"] = ema(ohlc15["close"], 50)

        out["mtf_rsi14"] = ohlc15["mtf_rsi14"].reindex(out.index, method="ffill")
        out["mtf_ema50"] = ohlc15["mtf_ema50"].reindex(out.index, method="ffill")

        out["mtf_bear"] = (out["mtf_rsi14"] < 50) & (out["close"] < out["mtf_ema50"])
        out["mtf_bull"] = (out["mtf_rsi14"] > 50) & (out["close"] > out["mtf_ema50"])
    else:
        out["mtf_bear"] = True
        out["mtf_bull"] = True

    # PRE-ALERT (dump vs pump)
    out["preAlertScore_dump"] = (
        out["rsi_degrade"].astype(int)
        + out["volume_cluster"].astype(int)
        + out["macd_deteriorating"].astype(int)
        + out["mtf_bear"].astype(int)
    )
    out["preAlert_dump"] = (out["preAlertScore_dump"] >= 2) & (out["avg_momentum_pct"] < momentum_dump_th)

    out["preAlertScore_pump"] = (
        out["rsi_improve"].astype(int)
        + out["volume_cluster"].astype(int)
        + out["macd_improving"].astype(int)
        + out["mtf_bull"].astype(int)
    )
    out["preAlert_pump"] = (out["preAlertScore_pump"] >= 2) & (out["avg_momentum_pct"] > momentum_pump_th)

    # ENTRY building blocks
    out["volSpike"] = out["volume"] > (out["vol_sma20"] * 2.2)
    out["atrSpike"] = out["atr14"] > (out["atr_sma14"] * 1.3)

    low_prev = out["low"].shift(1)
    high_prev = out["high"].shift(1)
    out["supportLvl"] = low_prev.rolling(20).min()
    out["resistLvl"] = high_prev.rolling(20).max()

    out["supportBreak"] = (out["close"] < out["supportLvl"]) & (out["close"].shift(1) >= out["supportLvl"])
    out["resistBreak"] = (out["close"] > out["resistLvl"]) & (out["close"].shift(1) <= out["resistLvl"])

    out["bearCandle"] = (out["open"] > out["close"]) & ((out["open"] - out["close"]) > (out["atr14"] * 0.8))
    out["bullCandle"] = (out["close"] > out["open"]) & ((out["close"] - out["open"]) > (out["atr14"] * 0.8))

    out["rsiDump"] = out["rsi10"] < rsiDumpTh
    out["rsiPump"] = out["rsi10"] > rsiPumpTh

    out["entryScore_short"] = (
        out["volSpike"].astype(int) * 2
        + out["atrSpike"].astype(int) * 1
        + out["rsiDump"].astype(int) * 1
        + out["supportBreak"].astype(int) * 2
        + out["bearCandle"].astype(int) * 1
    )

    out["entryScore_long"] = (
        out["volSpike"].astype(int) * 2
        + out["atrSpike"].astype(int) * 1
        + out["rsiPump"].astype(int) * 1
        + out["resistBreak"].astype(int) * 2
        + out["bullCandle"].astype(int) * 1
    )

    out["shortEntry_rule"] = (out["entryScore_short"] >= minEntryScore) & out["mtf_bear"]
    out["longEntry_rule"] = (out["entryScore_long"] >= minEntryScore) & out["mtf_bull"]

    return out.dropna()


# ============================================================
# LABELING (TP before SL) — ENTRY AT NEXT OPEN
# ============================================================
def tp_sl_labels(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    horizon = c(cfg, "labeling.horizon_bars", int)
    rr = c(cfg, "labeling.rr", float)
    sl_atr_mult = c(cfg, "labeling.sl_atr_mult", float)

    a = df["atr"]
    open_next = df["open"].shift(-1)
    high = df["high"]
    low = df["low"]

    sl_dist = sl_atr_mult * a
    tp_dist = rr * sl_dist

    n = len(df)
    y_long = np.zeros(n, dtype=np.int8)
    y_short = np.zeros(n, dtype=np.int8)
    valid = np.ones(n, dtype=np.int8)

    highs_fwd = np.vstack([high.shift(-i).to_numpy() for i in range(1, horizon + 1)])
    lows_fwd = np.vstack([low.shift(-i).to_numpy() for i in range(1, horizon + 1)])

    entry = open_next.to_numpy()
    slv = sl_dist.to_numpy()
    tpv = tp_dist.to_numpy()

    tail = horizon + 1
    valid[-tail:] = 0

    def first_idx(mask: np.ndarray) -> Optional[int]:
        idx = np.where(mask)[0]
        return int(idx[0]) if idx.size else None

    for t in range(n - tail):
        if not np.isfinite(entry[t]) or not np.isfinite(slv[t]) or not np.isfinite(tpv[t]) or slv[t] <= 0:
            valid[t] = 0
            continue

        long_tp = entry[t] + tpv[t]
        long_sl = entry[t] - slv[t]
        short_tp = entry[t] - tpv[t]
        short_sl = entry[t] + slv[t]

        tp_hit_l = highs_fwd[:, t] >= long_tp
        sl_hit_l = lows_fwd[:, t] <= long_sl
        i_tp = first_idx(tp_hit_l)
        i_sl = first_idx(sl_hit_l)
        if i_tp is not None and (i_sl is None or i_tp < i_sl):
            y_long[t] = 1

        tp_hit_s = lows_fwd[:, t] <= short_tp
        sl_hit_s = highs_fwd[:, t] >= short_sl
        i_tp = first_idx(tp_hit_s)
        i_sl = first_idx(sl_hit_s)
        if i_tp is not None and (i_sl is None or i_tp < i_sl):
            y_short[t] = 1

    return pd.Series(y_long, index=df.index), pd.Series(y_short, index=df.index), pd.Series(valid, index=df.index)


# ============================================================
# WALK-FORWARD RF
# ============================================================
def walk_forward_rf(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train_bars = c(cfg, "walkforward.train_bars", int)
    test_bars = c(cfg, "walkforward.test_bars", int)
    step_bars = c(cfg, "walkforward.step_bars", int)

    n_estimators = c(cfg, "model.n_estimators", int)
    max_depth = c(cfg, "model.max_depth")  # can be None
    min_samples_leaf = c(cfg, "model.min_samples_leaf", int)
    random_state = c(cfg, "model.random_state", int)

    yL, yS, valid = tp_sl_labels(df, cfg)
    out = df.copy()
    out["y_long"] = yL
    out["y_short"] = yS
    out["valid"] = valid

    feature_cols = [
        "ret1", "ret3", "ret8",
        "vol10", "vol30",
        "ema_diff",
        "rsi14",
        "atr_n",
        "vol_ratio",
        "hl_range", "body", "upper_wick", "lower_wick",
        "preAlertScore_dump", "preAlertScore_pump",
        "entryScore_short", "entryScore_long",
    ]

    p_long = np.full(len(out), np.nan, dtype=float)
    p_short = np.full(len(out), np.nan, dtype=float)

    i = 0
    while True:
        train_start = i
        train_end = train_start + train_bars
        test_end = train_end + test_bars
        if test_end > len(out):
            break

        train = out.iloc[train_start:train_end]
        test = out.iloc[train_end:test_end]

        train = train[train["valid"] == 1]
        test_valid = (test["valid"] == 1).to_numpy()

        X_train = train[feature_cols].to_numpy()
        y_long = train["y_long"].to_numpy()
        y_short = train["y_short"].to_numpy()

        mL = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        mS = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state + 1,
            class_weight="balanced_subsample",
        )

        mL.fit(X_train, y_long)
        mS.fit(X_train, y_short)

        X_test = test[feature_cols].to_numpy()
        probaL = mL.predict_proba(X_test)[:, 1]
        probaS = mS.predict_proba(X_test)[:, 1]

        base = train_end
        for k in range(len(test)):
            if test_valid[k]:
                p_long[base + k] = float(probaL[k])
                p_short[base + k] = float(probaS[k])

        i += step_bars

    out["p_long"] = p_long
    out["p_short"] = p_short
    return out


# ============================================================
# HYBRID SIGNALS
# ============================================================
def make_hybrid_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    prelook = c(cfg, "filters.prealert_lookback", int)
    require_rf = bool(c(cfg, "rf_confirm.require_rf"))
    p_long_th = c(cfg, "rf_confirm.p_long_th", float)
    p_short_th = c(cfg, "rf_confirm.p_short_th", float)
    require_prealert = bool(c(cfg, "rf_confirm.require_prealert"))

    out = df.copy()

    out["preAlert_recent_dump"] = out["preAlert_dump"].rolling(prelook).max().fillna(0).astype(bool)
    out["preAlert_recent_pump"] = out["preAlert_pump"].rolling(prelook).max().fillna(0).astype(bool)

    long_ok = out["longEntry_rule"].fillna(False)
    short_ok = out["shortEntry_rule"].fillna(False)

    if require_prealert:
        long_ok &= out["preAlert_recent_pump"]
        short_ok &= out["preAlert_recent_dump"]

    if require_rf:
        long_ok &= (out["p_long"] >= p_long_th)
        short_ok &= (out["p_short"] >= p_short_th)

    sig = pd.Series(0, index=out.index, dtype=int)
    sig[long_ok] = 1
    sig[short_ok] = -1

    both = long_ok & short_ok
    if both.any():
        choose_long = out.loc[both, "p_long"] >= out.loc[both, "p_short"]
        sig.loc[both] = np.where(choose_long.to_numpy(), 1, -1)

    out["signal"] = sig
    return out


# ============================================================
# BACKTEST
# ============================================================
def backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rr = c(cfg, "labeling.rr", float)
    sl_atr_mult = c(cfg, "labeling.sl_atr_mult", float)
    fee_bps = c(cfg, "execution.fee_bps", float)
    slippage_bps = c(cfg, "execution.slippage_bps", float)
    cooldown_bars = c(cfg, "execution.cooldown_bars", int)

    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    position = 0
    entry_price = np.nan
    sl_price = np.nan
    tp_price = np.nan
    cooldown = 0

    equity = 1.0
    eq_curve = []

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    atrv = df["atr"].to_numpy()
    sig = df["signal"].fillna(0).to_numpy()

    for t in range(len(df)):
        eq_curve.append(equity)

        if cooldown > 0:
            cooldown -= 1
            continue

        if position == 0:
            if t + 1 >= len(df):
                continue

            if sig[t] == 1:
                position = 1
                entry_price = o[t + 1] * (1 + slip)
                sl_dist = sl_atr_mult * atrv[t]
                tp_dist = rr * sl_dist
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
                equity *= (1 - fee)

            elif sig[t] == -1:
                position = -1
                entry_price = o[t + 1] * (1 - slip)
                sl_dist = sl_atr_mult * atrv[t]
                tp_dist = rr * sl_dist
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                equity *= (1 - fee)

        else:
            if position == 1:
                hit_tp = h[t] >= tp_price
                hit_sl = l[t] <= sl_price

                if hit_tp and hit_sl:
                    dist_tp = abs(tp_price - o[t])
                    dist_sl = abs(o[t] - sl_price)
                    exit_price = (tp_price if dist_tp < dist_sl else sl_price) * (1 - slip)
                elif hit_tp:
                    exit_price = tp_price * (1 - slip)
                elif hit_sl:
                    exit_price = sl_price * (1 - slip)
                else:
                    exit_price = None

                if exit_price is not None:
                    equity *= (exit_price / entry_price)
                    equity *= (1 - fee)
                    position = 0
                    cooldown = cooldown_bars

            elif position == -1:
                hit_tp = l[t] <= tp_price
                hit_sl = h[t] >= sl_price

                if hit_tp and hit_sl:
                    dist_tp = abs(o[t] - tp_price)
                    dist_sl = abs(sl_price - o[t])
                    exit_price = (tp_price if dist_tp < dist_sl else sl_price) * (1 + slip)
                elif hit_tp:
                    exit_price = tp_price * (1 + slip)
                elif hit_sl:
                    exit_price = sl_price * (1 + slip)
                else:
                    exit_price = None

                if exit_price is not None:
                    equity *= (entry_price / exit_price)
                    equity *= (1 - fee)
                    position = 0
                    cooldown = cooldown_bars

    out = df.copy()
    out["equity"] = eq_curve
    return out


def summarize(bt: pd.DataFrame) -> None:
    eq = bt["equity"].dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    changes = eq.pct_change().fillna(0)
    exits = int((changes != 0).sum())

    trade_rets = changes[changes != 0]
    if len(trade_rets) > 0:
        winrate = float((trade_rets > 0).mean())
        avg_win = float(trade_rets[trade_rets > 0].mean()) if (trade_rets > 0).any() else 0.0
        avg_loss = float(trade_rets[trade_rets < 0].mean()) if (trade_rets < 0).any() else 0.0
        expectancy = winrate * avg_win + (1 - winrate) * avg_loss
    else:
        winrate = avg_win = avg_loss = expectancy = 0.0

    print("==== Summary ====")
    print(f"Bars: {len(bt):,}")
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Max drawdown: {dd * 100:.2f}%")
    print(f"Exit count (approx): {exits:,}")
    print(
        f"Winrate (approx): {winrate * 100:.2f}% | "
        f"Avg win: {avg_win * 100:.3f}% | Avg loss: {avg_loss * 100:.3f}% | "
        f"Exp: {expectancy * 100:.3f}%"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and refetch from Binance")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    validate_cfg(cfg)

    symbol = c(cfg, "data.symbol", str)
    interval = c(cfg, "data.interval", str)
    days = c(cfg, "data.days", int)
    out_csv = c(cfg, "output.backtest_csv", str)

    print("Loaded config:", Path(args.config).resolve())
    print(f"Symbol/TF: {symbol} {interval} | Days: {days}")

    df = load_or_fetch_ohlcv(cfg, refresh=args.refresh)
    print("Bars loaded:", len(df))

    df = make_features(df, cfg)
    df = add_pine_like_filters(df, cfg)

    wf = walk_forward_rf(df, cfg)
    wf = make_hybrid_signals(wf, cfg)

    print("Signals distribution:", wf["signal"].value_counts(dropna=False).to_dict())

    bt = backtest(wf, cfg)
    summarize(bt)

    bt.to_csv(out_csv)
    print("Saved:", out_csv)

main()