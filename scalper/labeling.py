"""Labeling utilities for TP/SL evaluation."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import c


def tp_sl_labels(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series, pd.Series]:
    horizon = c(cfg, "labeling.horizon_bars", int)
    rr = c(cfg, "labeling.rr", float)
    sl_atr_mult = c(cfg, "labeling.sl_atr_mult", float)

    atr_values = df["atr"]
    open_next = df["open"].shift(-1)
    high = df["high"]
    low = df["low"]

    sl_dist = sl_atr_mult * atr_values
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
