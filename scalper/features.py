"""Feature engineering for the RandomForest model."""
from __future__ import annotations

import pandas as pd

from .config import c
from .indicators import atr, ema, rsi


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
