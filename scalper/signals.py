"""Hybrid signal construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import c


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
        long_ok &= out["p_long"] >= p_long_th
        short_ok &= out["p_short"] >= p_short_th

    sig = pd.Series(0, index=out.index, dtype=int)
    sig[long_ok] = 1
    sig[short_ok] = -1

    both = long_ok & short_ok
    if both.any():
        choose_long = out.loc[both, "p_long"] >= out.loc[both, "p_short"]
        sig.loc[both] = np.where(choose_long.to_numpy(), 1, -1)

    out["signal"] = sig
    return out
