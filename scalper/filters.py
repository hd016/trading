"""Pine-like filters for signal generation."""
from __future__ import annotations

import pandas as pd

from .config import c
from .indicators import atr, ema, macd_hist, rsi


def add_pine_like_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    warning_period = c(cfg, "filters.warningPeriod", int)
    momentum_dump_th = c(cfg, "filters.momentum_dump_th", float)
    momentum_pump_th = c(cfg, "filters.momentum_pump_th", float)
    volume_cluster_mult = c(cfg, "filters.volumeClusterMult", float)
    min_entry_score = c(cfg, "filters.minEntryScore", int)
    rsi_dump_th = c(cfg, "filters.rsiDumpTh", float)
    rsi_pump_th = c(cfg, "filters.rsiPumpTh", float)
    use_mtf = bool(c(cfg, "filters.useMTF"))

    out = df.copy()

    out["rsi10"] = rsi(out["close"], 10)
    out["macd_hist_1m"] = macd_hist(out["close"], 8, 21, 5)
    out["atr14"] = atr(out, 14)
    out["atr_sma14"] = out["atr14"].rolling(14).mean()

    out["rsi_trend"] = out["rsi10"] - out["rsi10"].shift(warning_period - 1)
    out["rsi_degrade"] = out["rsi_trend"] < 0
    out["rsi_improve"] = out["rsi_trend"] > 0

    first_close = out["close"].shift(warning_period - 1)
    out["avg_momentum_pct"] = ((out["close"] - first_close) / (first_close + 1e-12)) * 100.0

    out["avg_vol_last3"] = out["volume"].rolling(3).mean()
    out["volume_cluster"] = out["avg_vol_last3"] > (out["vol_sma20"] * volume_cluster_mult)

    out["macd_trend"] = out["macd_hist_1m"] - out["macd_hist_1m"].shift(warning_period - 1)
    out["macd_deteriorating"] = out["macd_trend"] <= 0
    out["macd_improving"] = out["macd_trend"] >= 0

    if use_mtf:
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

    out["rsiDump"] = out["rsi10"] < rsi_dump_th
    out["rsiPump"] = out["rsi10"] > rsi_pump_th

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

    out["shortEntry_rule"] = (out["entryScore_short"] >= min_entry_score) & out["mtf_bear"]
    out["longEntry_rule"] = (out["entryScore_long"] >= min_entry_score) & out["mtf_bull"]

    return out.dropna()
