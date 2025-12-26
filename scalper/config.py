"""Configuration utilities for the hybrid scalper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = [
    "data.symbol",
    "data.interval",
    "data.days",
    "labeling.horizon_bars",
    "labeling.atr_len",
    "labeling.rr",
    "labeling.sl_atr_mult",
    "filters.warningPeriod",
    "filters.momentum_dump_th",
    "filters.momentum_pump_th",
    "filters.volumeClusterMult",
    "filters.minEntryScore",
    "filters.rsiDumpTh",
    "filters.rsiPumpTh",
    "filters.useMTF",
    "filters.prealert_lookback",
    "rf_confirm.require_rf",
    "rf_confirm.p_long_th",
    "rf_confirm.p_short_th",
    "rf_confirm.require_prealert",
    "walkforward.train_bars",
    "walkforward.test_bars",
    "walkforward.step_bars",
    "model.n_estimators",
    "model.max_depth",
    "model.min_samples_leaf",
    "model.random_state",
    "execution.fee_bps",
    "execution.slippage_bps",
    "execution.cooldown_bars",
    "output.backtest_csv",
]


def load_yaml_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_cfg(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: '{dotted}'")
        cur = cur[part]
    return cur


def validate_cfg(cfg: dict) -> None:
    missing = []
    for key in REQUIRED_KEYS:
        try:
            get_cfg(cfg, key)
        except KeyError:
            missing.append(key)
    if missing:
        msg = "config.yaml is missing required keys:\n" + "\n".join(f"- {k}" for k in missing)
        raise ValueError(msg)


def c(cfg: dict, dotted: str, cast_type=None):
    value = get_cfg(cfg, dotted)
    if cast_type is None:
        return value
    if value is None:
        return None
    return cast_type(value)
