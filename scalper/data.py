"""Data loading and caching utilities for Binance USDT-M futures."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import c


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
    if fmt == "csv":
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    raise ValueError(f"Unsupported cache.format: {fmt}")


def save_cached_ohlcv(df: pd.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(path)
        return
    if fmt == "csv":
        out = df.copy()
        out = out.reset_index().rename(columns={"index": "timestamp"})
        out.to_csv(path, index=False)
        return
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
        if len(df) > 0 and set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
            return df
        print("Cache file invalid -> refetching...")

    print("Fetching bars from Binance...")
    df = load_binance_futures_range(symbol, interval, days)
    print("Saving cache:", path.resolve())
    save_cached_ohlcv(df, path, fmt)
    return df


def _binance_futures_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    base = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    response = requests.get(base, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()


def load_binance_futures_range(symbol: str, interval: str, days: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(days * 24 * 60 * 60 * 1000)

    frames = []
    cur = start_ms
    while True:
        df = _binance_futures_klines(symbol, interval, start_ms=cur, end_ms=now_ms, limit=1000)
        if df.empty:
            break
        frames.append(df)

        last_ts = df.index[-1]
        next_start = int(last_ts.timestamp() * 1000) + 60_000
        if next_start >= now_ms:
            break
        cur = next_start
        time.sleep(0.12)

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    return df_all
